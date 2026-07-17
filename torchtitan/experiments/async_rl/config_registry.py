# Copyright (c) Panocular AI.
#
# Config entry points for the async_rl coordination strategies, discoverable
# by torchtitan's ConfigManager via ``--module async_rl --config
# <function_name>`` (or the fully-qualified ``--module
# torchtitan.experiments.async_rl``).
#
# ``base_rl_config`` + ``wrap_replica`` are the shared building blocks: a plain
# RLTrainer.Config with the common loss/data choices, and a helper that
# copies it into one of the coordinator Configs (each a strict superset of
# RLTrainer.Config) plus the coordinator-specific extras. The ``rl_*``
# functions below are the full-size entry points, one per strategy x model
# (each with a "_0_6b" and, where a checkpoint is available, a larger preset).
#
# Adding a new RL model: add it to ``_MODEL_REGISTRY_BY_MODEL`` and
# ``_RENDERER_NAME_BY_MODEL`` below (and, if it should have a default
# checkpoint, ``_DEFAULT_HF_ASSETS_PATH``) -- no changes needed anywhere else
# in this package. GPU count is not fixed either: trainer/generator
# tensor_parallel_degree are real parameters here (default 1), and
# num_replicas / GPUS_PER_REPLICA (launch script arg) flow through
# independently -- nothing below assumes a specific machine's GPU count, only
# its own defaults do.
#
# ConfigManager calls the ``--config`` function with NO arguments (CLI flags
# then overlay onto the resulting dataclass's fields), so a size/flavor/model
# switch needs its own named entry point to be reachable from the CLI --
# passing ``model="llama3"`` to ``rl_heloco_qwen3_0_6b`` only works from
# Python (as the ``rl_heloco_llama3_8b`` wrapper below does).

import dataclasses
import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments import rl as _rl_pkg
from torchtitan.experiments.async_rl.async_inference.trainer import (
    AsyncInferenceReplica,
)
from torchtitan.experiments.async_rl.async_inference.worker import AsyncInferenceWorker

from torchtitan.experiments.async_rl.diloco.trainer import DiLoCoRLReplica
from torchtitan.experiments.async_rl.heloco.trainer import HeLoCoRLReplica
from torchtitan.experiments.async_rl.heloco_async_inference.trainer import (
    HeLoCoAsyncInferenceReplica,
)
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.examples.alphabet_sort import AlphabetSortRollouter
from torchtitan.experiments.rl.generator_router import (
    GeneratorRouter,
    RoundRobinRoutingStrategy,
)
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.trainer import GRPOLoss, RLTrainer
from torchtitan.models.llama3 import model_registry as _llama3_model_registry
from torchtitan.models.qwen3 import model_registry as _qwen3_model_registry

def _hf_model_registry(flavor, *, attn_backend, hf_assets_path):
    # Lazy: pulls transformers, only needed when the "hf" model is selected.
    from torchtitan.experiments.transformers_modeling_backend.rl import (
        model_registry,
    )

    return model_registry(
        flavor, attn_backend=attn_backend, hf_assets_path=hf_assets_path
    )


#: Entries share one signature: (flavor, *, attn_backend, hf_assets_path).
#: Native registries ignore hf_assets_path (they load weights via their own
#: state-dict adapters); the "hf" backend resolves its ARCHITECTURE from it.
_MODEL_REGISTRY_BY_MODEL = {
    "qwen3": lambda flavor, *, attn_backend, hf_assets_path: _qwen3_model_registry(
        flavor, attn_backend=attn_backend
    ),
    "llama3": lambda flavor, *, attn_backend, hf_assets_path: _llama3_model_registry(
        flavor, attn_backend=attn_backend
    ),
    "hf": _hf_model_registry,
}

#: Renderer (chat-template) name per model -- see
#: torchtitan.experiments.rl.renderer._RENDERER_BY_MODEL. llama3 has no
#: dedicated entry there, so it resolves via the "default" key.
_RENDERER_NAME_BY_MODEL = {
    "qwen3": "qwen3",
    "llama3": "default",
    # Arbitrary HF architectures: resolve the chat template from the
    # checkpoint's own tokenizer (renderers "auto" path).
    "hf": "auto",
}

_EXAMPLE_CHECKPOINT_DIR = os.path.join(_rl_pkg.__path__[0], "example_checkpoint")

#: Default hf_assets_path per (model, flavor). "qwen3"/"0.6B" ships inside
#: torchtitan's rl experiment (no download needed); the rest are conventional
#: paths under the same example_checkpoint dir (matching
#: torchtitan.experiments.rl.config_registry's own qwen3_1_7b/qwen3_14b
#: presets) that the caller downloads a checkpoint into, or pass
#: hf_assets_path explicitly.
_DEFAULT_HF_ASSETS_PATH = {
    ("qwen3", "0.6B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3-0.6B"),
    ("qwen3", "1.7B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3-1.7B"),
    ("llama3", "8B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Llama-3.1-8B"),
    # HF backend A/B baselines: same checkpoints as the native qwen3 presets.
    ("hf", "Qwen3-0.6B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3-0.6B"),
    ("hf", "Qwen3-1.7B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3-1.7B"),
}


def base_rl_config(
    gpu_memory_limit: float = 0.35,
    hf_assets_path: str | None = None,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    trainer_tensor_parallel_degree: int = 1,
    generator_tensor_parallel_degree: int = 1,
    rollouter=None,
) -> RLTrainer.Config:
    """Shared base RLTrainer config: stock GRPO loss, wandb off.

    ``model``/``flavor`` select the model spec (via ``_MODEL_REGISTRY_BY_MODEL``)
    and the default checkpoint path (via ``_DEFAULT_HF_ASSETS_PATH``);
    hf_assets_path overrides the default (e.g. for a fine-tuned checkpoint of
    the same flavor, or any (model, flavor) with no default). gpu_memory_limit
    is lowered from the base 0.9 so workers (and other users on a shared box)
    coexist without vLLM grabbing ~90% of each GPU for KV cache.
    trainer/generator tensor_parallel_degree default to 1; raise either for a
    model too large to fit one GPU per role -- the GPU mesh spawned by
    torchtitan.experiments.async_rl.train (and therefore the GPU count a
    launch script needs to provision) scales with these automatically.
    ``rollouter`` is the task bundle (dataset + reward rubric + environment, a
    ``Rollouter.Config`` subclass instance); it defaults to the alphabet-sort
    example task and flows through ``wrap_replica`` unchanged, so a different
    task/reward needs no coordinator changes.
    """
    if model not in _MODEL_REGISTRY_BY_MODEL:
        raise ValueError(
            f"unknown RL model {model!r} (known: "
            f"{sorted(_MODEL_REGISTRY_BY_MODEL)}); add it to "
            "_MODEL_REGISTRY_BY_MODEL/_RENDERER_NAME_BY_MODEL in this file"
        )
    resolved_hf_assets_path = hf_assets_path or _DEFAULT_HF_ASSETS_PATH.get(
        (model, flavor)
    )
    if resolved_hf_assets_path is None:
        raise ValueError(
            f"no default hf_assets_path for {model} {flavor!r}; add one to "
            "_DEFAULT_HF_ASSETS_PATH in this file or pass hf_assets_path "
            "explicitly"
        )
    model_registry = _MODEL_REGISTRY_BY_MODEL[model]
    return RLTrainer.Config(
        model_spec=model_registry(
            flavor,
            attn_backend="varlen",
            hf_assets_path=resolved_hf_assets_path,
        ),
        hf_assets_path=resolved_hf_assets_path,
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=rollouter
        if rollouter is not None
        else AlphabetSortRollouter.Config(),
        group_size=8,
        renderer=RendererConfig(
            name=_RENDERER_NAME_BY_MODEL[model], enable_thinking=False
        ),
        generator_router=GeneratorRouter.Config(
            strategy=RoundRobinRoutingStrategy.Config()
        ),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            # Structured-logging JSONL traces are a debugging aid that costs
            # real disk (hundreds of MB/hour per actor); off by default here
            # since async_rl runs are typically many-hour, many-actor swarms.
            debug=DebugConfig(enable_structured_logging=False),
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=trainer_tensor_parallel_degree,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            debug=DebugConfig(enable_structured_logging=False),
            model_dtype="bfloat16",
            gpu_memory_limit=gpu_memory_limit,
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=generator_tensor_parallel_degree,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def wrap_replica(cls, base: RLTrainer.Config, **kwargs):
    """Copy every field of a plain RLTrainer.Config ``base`` into ``cls.Config``
    (one of the coordinator Configs, each a strict superset of RLTrainer.Config),
    plus the coordinator-specific extras in ``kwargs`` (sync_every,
    num_replicas, should_quantize, ...). Avoids repeating the full field list
    in every config function."""
    base_fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    base_fields.update(kwargs)
    return cls.Config(**base_fields)


def rl_heloco_qwen3_0_6b(
    gpu_memory_limit: float = 0.35,
    hf_assets_path: str | None = None,
    sync_every: int = 4,
    train_seconds: float = 3600.0,
    num_outer_steps: int = 0,
    should_quantize: bool = True,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    trainer_tensor_parallel_degree: int = 1,
    generator_tensor_parallel_degree: int = 1,
) -> HeLoCoRLReplica.Config:
    """Async multi-worker GRPO (2 GPUs/worker at the default TP=1: 1 generator
    + 1 trainer GPU each, so two workers fit on a 4-GPU node alongside the CPU
    parameter server -- raise the tensor_parallel_degree params for a model
    that needs more than one GPU per role; the launch script's
    GPUS_PER_REPLICA must then grow to match).

    N workers each run a full RL loop and sync pseudo-gradients through the CPU
    parameter server (HeLoCo outer optimizer) with no barrier. Loss is stock GRPO.
    """
    return wrap_replica(
        HeLoCoRLReplica,
        base_rl_config(
            gpu_memory_limit,
            hf_assets_path,
            model=model,
            flavor=flavor,
            trainer_tensor_parallel_degree=trainer_tensor_parallel_degree,
            generator_tensor_parallel_degree=generator_tensor_parallel_degree,
        ),
        sync_every=sync_every,
        train_seconds=0.0 if num_outer_steps else train_seconds,
        num_outer_steps=num_outer_steps,
        should_quantize=should_quantize,
    )


def rl_heloco_qwen3_1_7b(**kwargs) -> HeLoCoRLReplica.Config:
    """1.7B preset; see rl_heloco_qwen3_0_6b for the strategy docstring
    (same 2-GPUs/worker layout -- 1.7B still fits comfortably at TP=1)."""
    kwargs.setdefault("flavor", "1.7B")
    return rl_heloco_qwen3_0_6b(**kwargs)


def rl_heloco_llama3_8b(**kwargs) -> HeLoCoRLReplica.Config:
    """Llama3-8B preset -- a second model family, proving the
    _MODEL_REGISTRY_BY_MODEL extension point works end to end with no other
    changes. No RL checkpoint ships for llama3; download one to
    example_checkpoint/Llama-3.1-8B first, or pass hf_assets_path explicitly.
    See rl_heloco_qwen3_0_6b for the strategy docstring."""
    kwargs.setdefault("model", "llama3")
    kwargs.setdefault("flavor", "8B")
    return rl_heloco_qwen3_0_6b(**kwargs)


def rl_heloco_hf_qwen3_0_6b(**kwargs) -> HeLoCoRLReplica.Config:
    """Qwen3-0.6B on the HF transformers modeling backend (see
    rl_diloco_hf_qwen3_0_6b for the backend notes; rl_heloco_qwen3_0_6b for
    the strategy docstring)."""
    kwargs.setdefault("model", "hf")
    kwargs.setdefault("flavor", "Qwen3-0.6B")
    return rl_heloco_qwen3_0_6b(**kwargs)


def rl_heloco_hf_qwen3_1_7b(**kwargs) -> HeLoCoRLReplica.Config:
    """1.7B HF-backend preset; see rl_heloco_hf_qwen3_0_6b."""
    kwargs.setdefault("flavor", "Qwen3-1.7B")
    return rl_heloco_hf_qwen3_0_6b(**kwargs)


def rl_heloco_hf(**kwargs) -> HeLoCoRLReplica.Config:
    """Model-agnostic HF-backend preset; see rl_diloco_hf."""
    if os.environ.get("RL_HF_ASSETS_PATH"):
        # ConfigManager calls config fns with NO args and only overlays CLI
        # flags onto the returned dataclass's FIELDS — too late for the
        # model_spec built here. Launchers that fetch a HF repo export the
        # checkpoint dir so the architecture resolves from it at call time.
        kwargs.setdefault("hf_assets_path", os.environ["RL_HF_ASSETS_PATH"])
    return rl_heloco_hf_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_qwen3_0_6b(
    gpu_memory_limit: float = 0.35,
    hf_assets_path: str | None = None,
    sync_every: int = 4,
    train_seconds: float = 3600.0,
    num_outer_steps: int = 0,
    should_quantize: bool = True,
    max_staleness: int = 4,
    rollout_queue_address: str = "",
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    trainer_tensor_parallel_degree: int = 1,
    generator_tensor_parallel_degree: int = 1,
) -> HeLoCoAsyncInferenceReplica.Config:
    """Prime-rl-style decoupled generation (arXiv:2505.07291) scaled to
    MULTIPLE trainers: N PURE-LEARNER HeLoCo trainer replicas (1 GPU each at
    the default TP=1 -- no local generation, no vLLM on the trainer) plus a
    separate pool of rl_heloco_async_inference_worker_* generator processes on
    their own machines that free-run rollouts into a hub-hosted shared queue.
    Each trainer pops rollouts from that queue, trains, and pushes its
    pseudo-gradient to the HeLoCo parameter server (no barrier); any trainer
    may consume any worker's rollouts. The hub
    (torchtitan.experiments.async_rl.heloco_async_inference.server)
    publishes the CURRENT global theta (the consensus weights, not any one
    trainer's copy) to a relay process for the generator pool to pull. Start
    the coordination plane first: async_inference.relay,
    async_inference.rollout_queue, then heloco_async_inference.server.
    rollout_queue_address is required (usually $ROLLOUT_QUEUE_ADDR, the same
    queue workers' rollout_queue_address points at). Loss is stock GRPO.
    """
    return wrap_replica(
        HeLoCoAsyncInferenceReplica,
        base_rl_config(
            gpu_memory_limit,
            hf_assets_path,
            model=model,
            flavor=flavor,
            trainer_tensor_parallel_degree=trainer_tensor_parallel_degree,
            generator_tensor_parallel_degree=generator_tensor_parallel_degree,
        ),
        sync_every=sync_every,
        train_seconds=0.0 if num_outer_steps else train_seconds,
        num_outer_steps=num_outer_steps,
        should_quantize=should_quantize,
        max_staleness=max_staleness,
        rollout_queue_address=rollout_queue_address,
    )


def rl_heloco_async_inference_qwen3_1_7b(
    **kwargs,
) -> HeLoCoAsyncInferenceReplica.Config:
    """1.7B preset; see rl_heloco_async_inference_qwen3_0_6b for the strategy
    docstring."""
    kwargs.setdefault("flavor", "1.7B")
    return rl_heloco_async_inference_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_llama3_8b(**kwargs) -> HeLoCoAsyncInferenceReplica.Config:
    """Llama3-8B preset -- a second model family, proving the
    _MODEL_REGISTRY_BY_MODEL extension point works end to end with no other
    changes. See rl_heloco_async_inference_qwen3_0_6b for the strategy
    docstring."""
    kwargs.setdefault("model", "llama3")
    kwargs.setdefault("flavor", "8B")
    return rl_heloco_async_inference_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_hf_qwen3_0_6b(
    **kwargs,
) -> HeLoCoAsyncInferenceReplica.Config:
    """Qwen3-0.6B on the HF transformers modeling backend (see
    rl_diloco_hf_qwen3_0_6b for the backend notes;
    rl_heloco_async_inference_qwen3_0_6b for the strategy docstring)."""
    kwargs.setdefault("model", "hf")
    kwargs.setdefault("flavor", "Qwen3-0.6B")
    return rl_heloco_async_inference_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_hf_qwen3_1_7b(
    **kwargs,
) -> HeLoCoAsyncInferenceReplica.Config:
    """1.7B HF-backend preset; see rl_heloco_async_inference_hf_qwen3_0_6b."""
    kwargs.setdefault("flavor", "Qwen3-1.7B")
    return rl_heloco_async_inference_hf_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_hf(**kwargs) -> HeLoCoAsyncInferenceReplica.Config:
    """Model-agnostic HF-backend preset; see rl_diloco_hf."""
    if os.environ.get("RL_HF_ASSETS_PATH"):
        # ConfigManager calls config fns with NO args and only overlays CLI
        # flags onto the returned dataclass's FIELDS — too late for the
        # model_spec built here. Launchers that fetch a HF repo export the
        # checkpoint dir so the architecture resolves from it at call time.
        kwargs.setdefault("hf_assets_path", os.environ["RL_HF_ASSETS_PATH"])
    return rl_heloco_async_inference_hf_qwen3_0_6b(**kwargs)


def rl_diloco_qwen3_0_6b(
    gpu_memory_limit: float = 0.35,
    hf_assets_path: str | None = None,
    sync_every: int = 4,
    train_seconds: float = 3600.0,
    num_outer_steps: int = 0,
    num_replicas: int = 2,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    trainer_tensor_parallel_degree: int = 1,
    generator_tensor_parallel_degree: int = 1,
) -> DiLoCoRLReplica.Config:
    """Synchronous DiLoCo GRPO (2 GPUs/worker at the default TP=1: 1 generator
    + 1 trainer GPU each, so two workers fit on a 4-GPU node -- raise the
    tensor_parallel_degree params for a model that needs more than one GPU per
    role; the launch script's GPUS_PER_REPLICA must then grow to match).

    N workers coordinate through a torchft Lighthouse/Manager quorum and sync
    averaged pseudo-gradients + an outer Nesterov-SGD step every sync_every
    steps (stock DiLoCo, no parameter server). Loss is stock GRPO.
    """
    return wrap_replica(
        DiLoCoRLReplica,
        base_rl_config(
            gpu_memory_limit,
            hf_assets_path,
            model=model,
            flavor=flavor,
            trainer_tensor_parallel_degree=trainer_tensor_parallel_degree,
            generator_tensor_parallel_degree=generator_tensor_parallel_degree,
        ),
        sync_every=sync_every,
        train_seconds=0.0 if num_outer_steps else train_seconds,
        num_outer_steps=num_outer_steps,
        num_replicas=num_replicas,
    )


def rl_diloco_qwen3_1_7b(**kwargs) -> DiLoCoRLReplica.Config:
    """1.7B preset; see rl_diloco_qwen3_0_6b for the strategy docstring."""
    kwargs.setdefault("flavor", "1.7B")
    return rl_diloco_qwen3_0_6b(**kwargs)


def rl_diloco_hf_qwen3_0_6b(**kwargs) -> DiLoCoRLReplica.Config:
    """Qwen3-0.6B via the HF transformers modeling backend — the architecture
    comes from the checkpoint's config.json, the modules ARE transformers
    modules, and weights load through the near-identity HF adapter. Same
    GRPO/DiLoCo recipe as rl_diloco_qwen3_0_6b (its A/B baseline), including
    trainer-side per-layer compile (aot_eager) and generator-side CUDA-graph
    capture. Generator-side per-layer compile is auto-disabled by the vLLM
    wrapper for HF backends: the transformers AttentionInterface dict dispatch
    (attention_instances[layer_idx]) specializes dynamo per layer x shape and
    exceeds any fullgraph recompile budget."""
    kwargs.setdefault("model", "hf")
    kwargs.setdefault("flavor", "Qwen3-0.6B")
    return rl_diloco_qwen3_0_6b(**kwargs)


def rl_diloco_hf_qwen3_1_7b(**kwargs) -> DiLoCoRLReplica.Config:
    """1.7B HF-backend preset; see rl_diloco_hf_qwen3_0_6b."""
    kwargs.setdefault("flavor", "Qwen3-1.7B")
    return rl_diloco_hf_qwen3_0_6b(**kwargs)


def rl_diloco_hf(**kwargs) -> DiLoCoRLReplica.Config:
    """Model-agnostic HF-backend preset: the checkpoint (and therefore the
    architecture) comes entirely from --hf_assets_path — pass it explicitly
    (launchers that fetch a HF repo always do). The flavor default is only the
    fallback checkpoint for bare CLI runs. Same for the rl_heloco_hf /
    rl_async_inference_hf / rl_heloco_async_inference_hf (+_worker_) family."""
    if os.environ.get("RL_HF_ASSETS_PATH"):
        # ConfigManager calls config fns with NO args and only overlays CLI
        # flags onto the returned dataclass's FIELDS — too late for the
        # model_spec built here. Launchers that fetch a HF repo export the
        # checkpoint dir so the architecture resolves from it at call time.
        kwargs.setdefault("hf_assets_path", os.environ["RL_HF_ASSETS_PATH"])
    return rl_diloco_hf_qwen3_0_6b(**kwargs)


def rl_diloco_llama3_8b(**kwargs) -> DiLoCoRLReplica.Config:
    """Llama3-8B preset; see rl_heloco_llama3_8b for the extension-point
    note and rl_diloco_qwen3_0_6b for the strategy docstring."""
    kwargs.setdefault("model", "llama3")
    kwargs.setdefault("flavor", "8B")
    return rl_diloco_qwen3_0_6b(**kwargs)


def rl_async_inference_qwen3_0_6b(
    gpu_memory_limit: float = 0.35,
    hf_assets_path: str | None = None,
    sync_every: int = 4,
    train_seconds: float = 3600.0,
    num_outer_steps: int = 0,
    max_staleness: int = 4,
    relay_addresses: str = "",
    rollout_queue_address: str = "",
    num_shards: int = 4,
    publish_every: int = 1,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    trainer_tensor_parallel_degree: int = 1,
    generator_tensor_parallel_degree: int = 1,
) -> AsyncInferenceReplica.Config:
    """Trainer role of prime-rl (arXiv:2505.07291): ONE pure-learner trainer
    (1 GPU, no local vLLM) fed entirely by a pool of remote generator workers
    (rl_async_inference_worker_*) on their own machines. The trainer pops
    rollouts from the standalone queue process at rollout_queue_address under
    a max_staleness bound, trains, and shards + publishes its weights to
    relay_addresses every publish_every windows (SHARDCAST-style) -- plus an
    initial publish at startup so the workers can bootstrap. Both addresses
    are required -- start the servers first:
    ``python -m torchtitan.experiments.async_rl.async_inference.relay`` and
    ``python -m torchtitan.experiments.async_rl.async_inference.rollout_queue``.
    Workers push rollouts to the same queue
    ($ASYNC_INFERENCE_ROLLOUT_QUEUE_ADDR) and pull weights from the relay
    ($ASYNC_INFERENCE_RELAY_ADDRS).
    """
    return wrap_replica(
        AsyncInferenceReplica,
        base_rl_config(
            gpu_memory_limit,
            hf_assets_path,
            model=model,
            flavor=flavor,
            trainer_tensor_parallel_degree=trainer_tensor_parallel_degree,
            generator_tensor_parallel_degree=generator_tensor_parallel_degree,
        ),
        sync_every=sync_every,
        train_seconds=0.0 if num_outer_steps else train_seconds,
        num_outer_steps=num_outer_steps,
        max_staleness=max_staleness,
        relay_addresses=relay_addresses,
        rollout_queue_address=rollout_queue_address,
        num_shards=num_shards,
        publish_every=publish_every,
    )


def rl_async_inference_qwen3_1_7b(**kwargs) -> AsyncInferenceReplica.Config:
    """1.7B preset; see rl_async_inference_qwen3_0_6b for the strategy docstring."""
    kwargs.setdefault("flavor", "1.7B")
    return rl_async_inference_qwen3_0_6b(**kwargs)


def rl_async_inference_llama3_8b(**kwargs) -> AsyncInferenceReplica.Config:
    """Llama3-8B preset -- a second model family, proving the
    _MODEL_REGISTRY_BY_MODEL extension point works end to end with no other
    changes. See rl_async_inference_qwen3_0_6b for the strategy docstring."""
    kwargs.setdefault("model", "llama3")
    kwargs.setdefault("flavor", "8B")
    return rl_async_inference_qwen3_0_6b(**kwargs)


def rl_async_inference_hf_qwen3_0_6b(**kwargs) -> AsyncInferenceReplica.Config:
    """Qwen3-0.6B on the HF transformers modeling backend (see
    rl_diloco_hf_qwen3_0_6b for the backend notes;
    rl_async_inference_qwen3_0_6b for the strategy docstring)."""
    kwargs.setdefault("model", "hf")
    kwargs.setdefault("flavor", "Qwen3-0.6B")
    return rl_async_inference_qwen3_0_6b(**kwargs)


def rl_async_inference_hf_qwen3_1_7b(**kwargs) -> AsyncInferenceReplica.Config:
    """1.7B HF-backend preset; see rl_async_inference_hf_qwen3_0_6b."""
    kwargs.setdefault("flavor", "Qwen3-1.7B")
    return rl_async_inference_hf_qwen3_0_6b(**kwargs)


def rl_async_inference_hf(**kwargs) -> AsyncInferenceReplica.Config:
    """Model-agnostic HF-backend preset; see rl_diloco_hf."""
    if os.environ.get("RL_HF_ASSETS_PATH"):
        # ConfigManager calls config fns with NO args and only overlays CLI
        # flags onto the returned dataclass's FIELDS — too late for the
        # model_spec built here. Launchers that fetch a HF repo export the
        # checkpoint dir so the architecture resolves from it at call time.
        kwargs.setdefault("hf_assets_path", os.environ["RL_HF_ASSETS_PATH"])
    return rl_async_inference_hf_qwen3_0_6b(**kwargs)


def rl_async_inference_worker_qwen3_0_6b(
    hf_assets_path: str | None = None,
    relay_addresses: str = "",
    rollout_queue_address: str = "",
    worker_id: int = 0,
    group_size: int = 8,
    groups_per_round: int = 2,
    poll_interval_s: float = 2.0,
    num_rounds: int = 0,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    generator_tensor_parallel_degree: int = 1,
) -> AsyncInferenceWorker.Config:
    """Inference-worker role of the async-inference relay swarm: no trainer
    fields apply here (this role has no trainer actor -- see
    async_inference/worker.py), so this copies only what a generator needs
    out of base_rl_config() rather than going through wrap_replica (which
    assumes an RLTrainer.Config-shaped target). relay_addresses (weights in)
    and rollout_queue_address (rollouts out, the standalone queue process both
    this worker and the trainer talk to) are both required.
    """
    base = base_rl_config(
        model=model,
        flavor=flavor,
        generator_tensor_parallel_degree=generator_tensor_parallel_degree,
    )
    return AsyncInferenceWorker.Config(
        model_spec=base.model_spec,
        hf_assets_path=hf_assets_path or base.hf_assets_path,
        generator=base.generator,
        rollouter=base.rollouter,
        renderer=base.renderer,
        group_size=group_size,
        groups_per_round=groups_per_round,
        relay_addresses=relay_addresses,
        rollout_queue_address=rollout_queue_address,
        worker_id=worker_id,
        poll_interval_s=poll_interval_s,
        num_rounds=num_rounds,
    )


def rl_async_inference_worker_qwen3_1_7b(**kwargs) -> AsyncInferenceWorker.Config:
    """1.7B preset; see rl_async_inference_worker_qwen3_0_6b for the strategy
    docstring."""
    kwargs.setdefault("flavor", "1.7B")
    return rl_async_inference_worker_qwen3_0_6b(**kwargs)


def rl_async_inference_worker_hf_qwen3_0_6b(**kwargs) -> AsyncInferenceWorker.Config:
    """Qwen3-0.6B HF-backend worker (see rl_diloco_hf_qwen3_0_6b for the
    backend notes; rl_async_inference_worker_qwen3_0_6b for the role
    docstring)."""
    kwargs.setdefault("model", "hf")
    kwargs.setdefault("flavor", "Qwen3-0.6B")
    return rl_async_inference_worker_qwen3_0_6b(**kwargs)


def rl_async_inference_worker_hf_qwen3_1_7b(**kwargs) -> AsyncInferenceWorker.Config:
    """1.7B HF-backend worker; see rl_async_inference_worker_hf_qwen3_0_6b."""
    kwargs.setdefault("flavor", "Qwen3-1.7B")
    return rl_async_inference_worker_hf_qwen3_0_6b(**kwargs)


def rl_async_inference_worker_hf(**kwargs) -> AsyncInferenceWorker.Config:
    """Model-agnostic HF-backend worker; see rl_diloco_hf."""
    if os.environ.get("RL_HF_ASSETS_PATH"):
        # ConfigManager calls config fns with NO args and only overlays CLI
        # flags onto the returned dataclass's FIELDS — too late for the
        # model_spec built here. Launchers that fetch a HF repo export the
        # checkpoint dir so the architecture resolves from it at call time.
        kwargs.setdefault("hf_assets_path", os.environ["RL_HF_ASSETS_PATH"])
    return rl_async_inference_worker_hf_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_worker_qwen3_0_6b(
    hf_assets_path: str | None = None,
    relay_addresses: str = "",
    rollout_queue_address: str = "",
    worker_id: int = 0,
    group_size: int = 8,
    groups_per_round: int = 8,
    poll_interval_s: float = 2.0,
    num_rounds: int = 0,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    generator_tensor_parallel_degree: int = 1,
) -> AsyncInferenceWorker.Config:
    """Inference-worker (generator) role of the heloco_async_inference swarm:
    the exact same AsyncInferenceWorker process as rl_async_inference_worker_*
    (all workers free-run -- generate continuously at their current weights,
    upgrading opportunistically -- which is the definition of a decoupled
    async generator). These workers are the trainers' SOLE rollout source (no
    trainer here runs any local generation), and there can be many of them
    feeding many trainers through the one hub. Point relay_addresses
    (weights in) at the relay process and rollout_queue_address (rollouts out)
    at the shared queue process ($ROLLOUT_QUEUE_ADDR). ``groups_per_round`` defaults
    higher than the base preset so a small generator pool fills a trainer's
    per-window token target in a few rounds.
    """
    base = base_rl_config(
        model=model,
        flavor=flavor,
        generator_tensor_parallel_degree=generator_tensor_parallel_degree,
    )
    return AsyncInferenceWorker.Config(
        model_spec=base.model_spec,
        hf_assets_path=hf_assets_path or base.hf_assets_path,
        generator=base.generator,
        rollouter=base.rollouter,
        renderer=base.renderer,
        group_size=group_size,
        groups_per_round=groups_per_round,
        relay_addresses=relay_addresses,
        rollout_queue_address=rollout_queue_address,
        worker_id=worker_id,
        poll_interval_s=poll_interval_s,
        num_rounds=num_rounds,
    )


def rl_heloco_async_inference_worker_qwen3_1_7b(
    **kwargs,
) -> AsyncInferenceWorker.Config:
    """1.7B preset; see rl_heloco_async_inference_worker_qwen3_0_6b."""
    kwargs.setdefault("flavor", "1.7B")
    return rl_heloco_async_inference_worker_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_worker_hf_qwen3_0_6b(
    **kwargs,
) -> AsyncInferenceWorker.Config:
    """Qwen3-0.6B HF-backend worker (see rl_diloco_hf_qwen3_0_6b for the
    backend notes; rl_heloco_async_inference_worker_qwen3_0_6b for the role
    docstring)."""
    kwargs.setdefault("model", "hf")
    kwargs.setdefault("flavor", "Qwen3-0.6B")
    return rl_heloco_async_inference_worker_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_worker_hf_qwen3_1_7b(
    **kwargs,
) -> AsyncInferenceWorker.Config:
    """1.7B HF-backend worker; see
    rl_heloco_async_inference_worker_hf_qwen3_0_6b."""
    kwargs.setdefault("flavor", "Qwen3-1.7B")
    return rl_heloco_async_inference_worker_hf_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_worker_hf(**kwargs) -> AsyncInferenceWorker.Config:
    """Model-agnostic HF-backend worker; see rl_diloco_hf."""
    if os.environ.get("RL_HF_ASSETS_PATH"):
        # ConfigManager calls config fns with NO args and only overlays CLI
        # flags onto the returned dataclass's FIELDS — too late for the
        # model_spec built here. Launchers that fetch a HF repo export the
        # checkpoint dir so the architecture resolves from it at call time.
        kwargs.setdefault("hf_assets_path", os.environ["RL_HF_ASSETS_PATH"])
    return rl_heloco_async_inference_worker_hf_qwen3_0_6b(**kwargs)
