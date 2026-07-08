# Copyright (c) Panocular AI.
#
# On-GPU trainer actor for synchronous DiLoCo RL: wraps the model + inner
# optimizer in torchft's ``local_sgd.DiLoCo`` (stock DiLoCo, Douillard et al.,
# 2311.08105), which hooks the inner optimizer's post-step and drives the
# pseudo-gradient all-reduce + outer Nesterov step automatically every
# ``sync_every`` steps -- no parameter server, no custom outer optimizer.

import logging
from datetime import timedelta

import torch
import torch.distributed as dist
from monarch.actor import endpoint

from torchft.checkpointing.http_transport import HTTPTransport
from torchft.local_sgd import DiLoCo
from torchft.manager import Manager
from torchft.process_group import ProcessGroupGloo

from torchtitan.experiments.rl.actors.trainer import PolicyTrainer

logger = logging.getLogger(__name__)


class DiLoCoManagerTrainer(PolicyTrainer):
    """PolicyTrainer that syncs across replicas via torchft ``local_sgd.DiLoCo``.

    Each replica is a single-GPU trainer (TP=1, world_size=1): the model params
    are unsharded on this rank, so DiLoCo's pseudo-gradient all-reduce moves whole
    tensors across replicas (the small-model transport assumption). The base
    ``forward_backward`` / ``optim_step`` are reused unchanged -- the DiLoCo sync
    is driven entirely by the inner-optimizer post-step hook.
    """

    def _diloco_state_dict(self) -> dict:
        # Consulted only by the Manager's recovery path (a replica that falls
        # behind after a failed all-reduce). Not exercised in a clean run.
        return {
            "model": self.model.state_dict(),
            "inner_optim": self.optimizers.optimizers[0].state_dict(),
        }

    def _diloco_load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.optimizers.optimizers[0].load_state_dict(state_dict["inner_optim"])

    @endpoint
    async def setup_diloco(
        self,
        *,
        lighthouse_address: str,
        replica_id: int,
        num_replicas: int,
        sync_every: int,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
    ) -> None:
        """Build the torchft Manager and enter the DiLoCo context.

        ``min_replica_size = num_replicas`` and ``use_async_quorum=False`` make the
        quorum a synchronous barrier over all workers. ``init_sync=False`` because
        every replica loads the SAME HF checkpoint deterministically, so weights
        already agree at step 0 (skips a fragile DTensor-over-HTTP broadcast).
        """
        # Each replica owns a private TCPStore (single rank; the Manager needs one
        # even at world_size=1). port=0 -> OS-assigned free port.
        self._diloco_store = dist.TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        pg = ProcessGroupGloo(timeout=timedelta(seconds=60))
        transport = HTTPTransport(timeout=timedelta(seconds=300), num_chunks=0)

        self._diloco_manager = Manager(
            pg=pg,
            min_replica_size=num_replicas,
            use_async_quorum=False,
            load_state_dict=self._diloco_load_state_dict,
            state_dict=self._diloco_state_dict,
            replica_id=f"diloco_{replica_id}",
            store_addr="localhost",
            store_port=self._diloco_store.port,
            rank=0,
            world_size=1,
            lighthouse_addr=lighthouse_address,
            port=19530 + replica_id,
            connect_timeout=timedelta(seconds=120),
            quorum_timeout=timedelta(seconds=900),
            timeout=timedelta(seconds=900),
            checkpoint_transport=transport,
            init_sync=False,
        )

        inner_optimizer = self.optimizers.optimizers[0]
        outer_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=outer_lr,
            momentum=outer_momentum,
            nesterov=True,
        )
        self._diloco = DiLoCo(
            self._diloco_manager,
            [self.model_parts[0]],
            inner_optimizer,
            outer_optimizer,
            backup_device=self.device,
            sync_every=sync_every,
        )
        self._diloco.__enter__()
        logger.info(
            "DiLoCoManagerTrainer replica=%d connected to lighthouse=%s "
            "(sync_every=%d, outer_lr=%.3g, momentum=%.3g)",
            replica_id,
            lighthouse_address,
            sync_every,
            outer_lr,
            outer_momentum,
        )

    @endpoint
    async def diloco_step_info(self) -> dict:
        """Expose the Manager's current step / participant count for logging."""
        return {
            "current_step": int(self._diloco_manager.current_step()),
            "num_participants": int(self._diloco_manager.num_participants()),
        }

    @endpoint
    async def close_diloco(self) -> None:
        if getattr(self, "_diloco", None) is not None:
            self._diloco.__exit__(None, None, None)
            self._diloco = None
        if getattr(self, "_diloco_manager", None) is not None:
            self._diloco_manager.shutdown(wait=False)
            self._diloco_manager = None
