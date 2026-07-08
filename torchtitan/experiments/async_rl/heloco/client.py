# Copyright (c) Panocular AI.
#
# torchft client for HeLoCo RL, driven explicitly at the H-window boundary on
# a CPU state dict rather than an nn.Module.

import logging
import socket
import threading
import uuid

import torch

from torchft.async_diloco import AsyncDiLoCo

logger: logging.Logger = logging.getLogger(__name__)


class HeLoCoRLClient(AsyncDiLoCo):
    """AsyncDiLoCo driven explicitly at the H-window boundary.

    Operates on a CPU state dict (server parameter names) rather than an
    ``nn.Module``, so it does not care how the trainer shards its weights.

    The instance holds the authoritative *window-start* global parameters in
    ``self._global_params``; :meth:`push` computes the pseudo-gradient
    ``global - local`` against that snapshot, so the orchestrator never has to
    track theta_0 itself.

    The wire transfer itself is the parent's :meth:`_session_roundtrip`
    (one HTTP ``POST /sync`` per cycle, flat coalesced buffers -- see
    ``AsyncDiLoCoServer`` for the format); only the model-facing side
    (state dicts instead of an ``nn.Module``, explicit drivers instead of an
    optimizer post-step hook) is replaced here.

    Args:
        server_address: HTTP ``/sync`` URL from
            :py:meth:`AsyncDiLoCoServer.address`.
        param_names: Ordered parameter names, matching the server's
            ``named_parameters()`` order exactly. This ordering defines the
            flat wire layout on both ends -- get it wrong and the transfer
            silently scrambles tensors.
        param_shapes: ``{name: shape}`` for unflattening the received
            global parameters.
        param_dtypes: ``{name: dtype}`` for the local global-parameter
            snapshot. Should be the server's storage dtype (fp32).
        heartbeat_address: Optional ``/heartbeat`` URL from
            :py:meth:`AsyncDiLoCoServer.heartbeat_address`.
        heartbeat_interval: Seconds between heartbeat pings.
        should_quantize: Upload pseudo-gradients as blockwise symmetric int8
            (the parameter download stays float32). Must match the server.
        sync_timeout: Socket timeout per sync request. Must exceed the
            server's ``grace_period``.
    """

    def __init__(
        self,
        server_address: str,
        param_names: list[str],
        param_shapes: dict[str, torch.Size],
        param_dtypes: dict[str, torch.dtype],
        *,
        heartbeat_address: str | None = None,
        heartbeat_interval: float = 2.0,
        should_quantize: bool = False,
        sync_timeout: float = 60.0,
    ) -> None:
        # Intentionally do NOT call super().__init__: it requires an nn.Module
        # and an inner optimizer, neither of which exists here. The parent
        # methods used are _session_roundtrip (needs _server_address,
        # _baseline_revision, _quantize, _param_numels, _total_numel,
        # _sync_timeout) and _run_heartbeat (needs _heartbeat_url/_stop/
        # _interval), whose state is set up below; everything else is
        # overridden.
        self._server_address = server_address
        self._quantize = should_quantize
        self._sync_timeout = sync_timeout

        self._param_names: list[str] = list(param_names)
        self._param_shapes: dict[str, torch.Size] = dict(param_shapes)
        self._global_params: dict[str, torch.Tensor] = {
            name: torch.zeros(
                param_shapes[name], dtype=param_dtypes[name], device="cpu"
            )
            for name in self._param_names
        }
        self._param_numels: list[int] = [
            self._global_params[name].numel() for name in self._param_names
        ]
        self._total_numel: int = sum(self._param_numels)

        # Revision of the server global model our snapshot is based on; sent
        # with every push so the server can reject a pseudo-gradient computed
        # against a baseline it lost continuity with (checkpoint restore).
        self._baseline_revision: int = 0

        #: DyLU recommendation from the most recent pull/push (0 = no change).
        self.last_dylu_steps: int = 0

        self._heartbeat_interval = heartbeat_interval
        if heartbeat_address is not None:
            # Unique per instance: every replica must register under a
            # distinct id, hostname-prefixed for readable server logs.
            worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
            self._heartbeat_url: str | None = (
                f"{heartbeat_address}?worker_id={worker_id}"
            )
        else:
            self._heartbeat_url = None
        self._heartbeat_stop: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None

    @property
    def revision(self) -> int:
        """Server global-model revision this client's snapshot is based on
        (updated by every :meth:`pull`/:meth:`push`)."""
        return self._baseline_revision

    # ------------------------------------------------------------------ #
    # Explicit drivers (replace the context-manager + post-step hook).
    # ------------------------------------------------------------------ #

    def pull(self) -> dict[str, torch.Tensor]:
        """Pull the current global parameters without sending a pseudo-gradient.

        Updates ``self._global_params`` in place and returns a clone (CPU tensors).
        """
        flat_params, new_steps, revision, _ = self._session_roundtrip(
            flag=0.0, speed=0.0, flat_grads=None
        )
        self._adopt_flat(flat_params, revision, new_steps)
        return {name: t.clone() for name, t in self._global_params.items()}

    def push(
        self, local_state_dict: dict[str, torch.Tensor], speed: float = 0.0
    ) -> dict[str, torch.Tensor]:
        """Push the pseudo-gradient and pull the updated global parameters.

        ``pseudo_grad[name] = self._global_params[name] - local_state_dict[name]``
        (computed in fp32 on CPU). Sends ``speed`` for DyLU, receives the new
        global theta, updates ``self._global_params``, and returns a clone of it.

        If the server rejects the push (stale baseline revision, e.g. after
        the server restored from a checkpoint), the window's pseudo-gradient
        is dropped and the response is adopted as a pure re-baseline.

        Args:
            local_state_dict: theta_local at the window end, keyed by server
                parameter names, CPU. Upcast to fp32 here if needed.
            speed: inner steps/sec over the window, for DyLU.
        """
        grad_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for name in self._param_names:
                local = local_state_dict[name].detach().to("cpu", torch.float32)
                grad_chunks.append((self._global_params[name] - local).reshape(-1))
        flat_grads = torch.cat(grad_chunks)

        flat_params, new_steps, revision, applied = self._session_roundtrip(
            flag=1.0, speed=speed, flat_grads=flat_grads
        )
        if not applied:
            logger.warning(
                "HeLoCo push rejected by server (baseline revision %d); "
                "re-baselining to server revision %d",
                self._baseline_revision,
                revision,
            )
        self._adopt_flat(flat_params, revision, new_steps)
        return {name: t.clone() for name, t in self._global_params.items()}

    # ------------------------------------------------------------------ #
    # Heartbeat lifecycle (explicit, not tied to __enter__/__exit__).
    # ------------------------------------------------------------------ #

    def start_heartbeat(self) -> None:
        if self._heartbeat_url is None or self._heartbeat_thread is not None:
            return
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._run_heartbeat, daemon=True
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self._heartbeat_interval * 2)
        self._heartbeat_thread = None

    # ------------------------------------------------------------------ #
    # Adoption of a pulled flat parameter buffer.
    # ------------------------------------------------------------------ #

    def _adopt_flat(
        self, flat_params: torch.Tensor, revision: int, new_steps: int
    ) -> None:
        """Unflatten a received fp32 buffer into ``_global_params``.

        Replaces the parent's ``_adopt_global`` (which installs into an
        ``nn.Module``); the orchestrator adopts the returned state dict into
        the trainer itself.
        """
        with torch.no_grad():
            offset = 0
            for name in self._param_names:
                target = self._global_params[name]
                n = target.numel()
                target.copy_(flat_params[offset : offset + n].view(target.shape))
                offset += n
        self._baseline_revision = revision
        self.last_dylu_steps = new_steps
