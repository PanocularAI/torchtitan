# Copyright (c) Panocular AI.
#
# Tests for the shared decentralized_rl infrastructure: config_registry.py (model/
# flavor/task/GPU-count resolution and Config validation), controller.py
# (the window runner and the template train() loop), train.py
# (PerHostProvisioner), and the package's CPU-light import guarantee.
# Per-strategy tests live next to their package (heloco/test_heloco.py etc).

import asyncio
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from torchtitan.experiments.decentralized_rl.config_registry import (
    _DEFAULT_HF_ASSETS_PATH,
    _MODEL_REGISTRY_BY_MODEL,
    _RENDERER_NAME_BY_MODEL,
    base_rl_config,
    rl_diloco_llama3_8b,
    rl_heloco_llama3_8b,
    rl_heloco_qwen3_1_7b,
    wrap_replica,
)
from torchtitan.experiments.decentralized_rl.controller import RLControllerMixin
from torchtitan.experiments.decentralized_rl.train import PerHostProvisioner
from torchtitan.experiments.decentralized_rl.trainers import DiLoCoRLReplica, HeLoCoRLReplica

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# === config_registry.py ====================================================


def test_model_and_flavor_resolution():
    """flavor selects the model spec and the default checkpoint (an explicit
    hf_assets_path still wins); llama3 resolves as a second model family
    through the SAME builder functions; unknown models fail clearly."""
    cfg_default = base_rl_config()
    cfg_large = base_rl_config(flavor="1.7B")
    assert cfg_default.model_spec.flavor == "0.6B"
    assert cfg_large.model_spec.flavor == "1.7B"
    assert cfg_large.hf_assets_path == _DEFAULT_HF_ASSETS_PATH[("qwen3", "1.7B")]
    assert cfg_large.hf_assets_path != cfg_default.hf_assets_path
    cfg_override = base_rl_config(hf_assets_path="/tmp/custom", flavor="1.7B")
    assert cfg_override.hf_assets_path == "/tmp/custom"
    # The named 1.7B preset routes through the same flavor resolution.
    assert rl_heloco_qwen3_1_7b().hf_assets_path == cfg_large.hf_assets_path

    # A second model family: no coordinator code is model-specific.
    for fn, cls in (
        (rl_heloco_llama3_8b, HeLoCoRLReplica),
        (rl_diloco_llama3_8b, DiLoCoRLReplica),
    ):
        cfg = fn()
        assert isinstance(cfg, cls.Config)
        assert cfg.model_spec.name == "llama3" and cfg.model_spec.flavor == "8B"
        # llama3 has no dedicated renderer entry -- resolved via the "default" key.
        assert cfg.renderer.name == "default"
        assert cfg.hf_assets_path.endswith("Llama-3.1-8B")

    with pytest.raises(ValueError, match="unknown RL model 'bogus_model'"):
        base_rl_config(model="bogus_model")


def test_new_model_needs_only_registry_dict_entries():
    """The extension contract in practice: register a fake model purely at
    runtime (no filesystem changes) by adding entries to the three module-
    level dicts, and confirm base_rl_config resolves it with no other
    decentralized_rl code touched. Without the hf_assets_path entry, resolution
    fails with a clear error rather than a silent bogus default."""
    from torchtitan.models.qwen3 import model_registry as qwen3_model_registry

    _MODEL_REGISTRY_BY_MODEL[
        "_fake_for_test"
    ] = lambda flavor, *, attn_backend, hf_assets_path: qwen3_model_registry(
        "0.6B", attn_backend=attn_backend
    )
    _RENDERER_NAME_BY_MODEL["_fake_for_test"] = "auto"
    try:
        with pytest.raises(ValueError, match="no default hf_assets_path"):
            base_rl_config(model="_fake_for_test", flavor="tiny")

        _DEFAULT_HF_ASSETS_PATH[("_fake_for_test", "tiny")] = "/fake/checkpoints/tiny"
        try:
            cfg = base_rl_config(model="_fake_for_test", flavor="tiny")
            assert cfg.hf_assets_path == "/fake/checkpoints/tiny"
            assert cfg.renderer.name == "auto"
        finally:
            del _DEFAULT_HF_ASSETS_PATH[("_fake_for_test", "tiny")]
    finally:
        del _MODEL_REGISTRY_BY_MODEL["_fake_for_test"]
        del _RENDERER_NAME_BY_MODEL["_fake_for_test"]


def test_tensor_parallel_degree_is_a_real_gpu_count_knob():
    """trainer/generator tensor_parallel_degree must not be hardcoded -- the
    number of GPUs a role needs is a function of these, so a model too big
    for one GPU per role must be expressible without editing decentralized_rl: as a
    Python kwarg AND as a CLI overlay on a named --config preset (the real
    ConfigManager path)."""
    from torchtitan.config import ConfigManager

    default_cfg = base_rl_config()
    assert default_cfg.trainer.parallelism.tensor_parallel_degree == 1
    assert default_cfg.generator.parallelism.tensor_parallel_degree == 1

    scaled_cfg = base_rl_config(
        trainer_tensor_parallel_degree=4, generator_tensor_parallel_degree=2
    )
    assert scaled_cfg.trainer.parallelism.tensor_parallel_degree == 4
    assert scaled_cfg.generator.parallelism.tensor_parallel_degree == 2

    cfg = ConfigManager().parse_args(
        [
            "--module",
            "decentralized_rl",
            "--config",
            "rl_heloco_qwen3_0_6b",
            "--trainer.parallelism.tensor_parallel_degree",
            "2",
            "--generator.parallelism.tensor_parallel_degree",
            "8",
        ]
    )
    assert cfg.trainer.parallelism.tensor_parallel_degree == 2
    assert cfg.generator.parallelism.tensor_parallel_degree == 8


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(sync_every=0, train_seconds=60.0), "sync_every"),
        (dict(train_seconds=0.0), "exactly one"),  # neither bound
        (dict(train_seconds=60.0, num_outer_steps=4), "exactly one"),  # both
    ],
)
def test_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        wrap_replica(HeLoCoRLReplica, base_rl_config(), **kwargs)


def test_rollouter_is_a_swappable_task():
    """The task bundle (dataset + reward rubric + environment) must be
    injectable without touching any coordinator code: a different
    Rollouter.Config passed to base_rl_config flows through wrap_replica into
    every strategy's Config unchanged."""
    from torchtitan.experiments.rl.examples.alphabet_sort import AlphabetSortRollouter

    custom = AlphabetSortRollouter.Config()  # stands in for any other task's Config
    base = base_rl_config(rollouter=custom)
    assert base.rollouter is custom
    cfg = wrap_replica(HeLoCoRLReplica, base, train_seconds=60.0)
    assert cfg.rollouter is custom
    # Default preserved when not passed.
    assert type(base_rl_config().rollouter) is AlphabetSortRollouter.Config


# === controller.py =========================================================


class _WindowHost(RLControllerMixin):
    """Mixin host with the generator/trainer work replaced by event recorders."""

    def __init__(self, *, losses=None, collect_s=0.002, train_s=0.004):
        self.config = SimpleNamespace(replica_id=0)
        self._policy_version = 0
        self.events = []
        self._losses = iter(losses or [])
        self._collect_s = collect_s
        self._train_s = train_s

    async def _collect_and_build(self, step):
        self.events.append(("collect_start", step))
        await asyncio.sleep(self._collect_s)
        self.events.append(("collect_end", step))
        return [f"mb{step}"], []

    async def _train_on(self, packed, rollout_groups):
        step = int(packed[0][2:])
        self.events.append(("train_start", step))
        await asyncio.sleep(self._train_s)
        self.events.append(("train_end", step))
        loss = next(self._losses, 0.1)
        return {
            "loss": loss,
            "reward_mean": 0.5,
            "policy_version": step,
            "num_rollouts": 8,
            "staleness": 0,
        }


def test_run_window_pipelined_overlap_and_divergence_cancel():
    # _run_window is always pipelined (one-step-ahead generation/training
    # overlap); there is no sequential path.
    host = _WindowHost()
    rewards, last, global_step, diverged = asyncio.run(host._run_window(3, 0))
    assert not diverged
    assert global_step == 3 and len(rewards) == 3
    ev = host.events
    # Collection for step h+1 starts before training on step h finishes.
    assert ev.index(("collect_start", 2)) < ev.index(("train_end", 1))
    assert ev.index(("collect_start", 3)) < ev.index(("train_end", 2))
    # No collection for a step beyond the window (nothing straddles the boundary).
    assert ("collect_start", 4) not in ev

    # A non-finite loss stops the window early (divergence).
    host = _WindowHost(losses=[0.1, math.inf])
    rewards, last, global_step, diverged = asyncio.run(host._run_window(4, 0))
    assert diverged and len(rewards) == 2

    # Regression: divergence must CANCEL the already-launched collection for
    # the next step, not leak it to run during shutdown.
    host = _WindowHost(losses=[math.nan], collect_s=0.05)

    async def run():
        result = await host._run_window(3, 0)
        # Give a leaked task time to run if one existed.
        await asyncio.sleep(0.1)
        return result

    rewards, last, global_step, diverged = asyncio.run(run())
    assert diverged and len(rewards) == 1
    # Step 2's collection was launched but cancelled before completing.
    assert ("collect_start", 2) in host.events
    assert ("collect_end", 2) not in host.events


class _LoopHost(RLControllerMixin):
    def __init__(
        self,
        *,
        num_outer_steps=0,
        train_seconds=0.0,
        sync_every=2,
        window_s=0.0,
        diverge_after=None,
    ):
        self.config = SimpleNamespace(
            replica_id=3,
            sync_every=sync_every,
            num_outer_steps=num_outer_steps,
            train_seconds=train_seconds,
        )
        self._policy_version = 0
        self.calls = []
        self._window_s = window_s
        self._diverge_after = diverge_after
        self.windows_run = 0

    def _build_sync_pipeline(self):
        # Provided by RLTrainer on real hosts; the loop tests fake the window
        # runner entirely, so the pipeline is never used. Not recorded in
        # self.calls to keep the asserted hook sequences focused on the loop.
        pass

    async def _run_window(self, sync_every, start_step):
        self.calls.append(("window", sync_every, start_step))
        self.windows_run += 1
        if self._window_s:
            await asyncio.sleep(self._window_s)
        diverged = (
            self._diverge_after is not None and self.windows_run > self._diverge_after
        )
        last = {
            "loss": math.nan if diverged else 0.1234,
            "reward_mean": 0.6,
            "policy_version": 1,
            "num_rollouts": 8,
            "staleness": 1,
        }
        return [0.5, 0.7], last, start_step + sync_every, diverged

    async def _validate_fixed(self, step):
        self.calls.append(("validate", step))
        return None

    def _aggregate_validation(self, metrics):
        return {"validation_reward/_mean": 0.25}

    async def _train_setup(self):
        self.calls.append(("setup",))

    async def _window_sync(self, t0):
        self.calls.append(("sync",))
        return "detail line"

    async def _after_validation(self):
        self.calls.append(("resume",))

    async def _train_teardown(self):
        self.calls.append(("teardown",))

    async def _train_cleanup(self):
        self.calls.append(("cleanup",))


def test_train_step_bound_hook_order():
    host = _LoopHost(num_outer_steps=2, sync_every=2)
    asyncio.run(host.train())
    assert host.calls == [
        ("validate", 0),  # pre-training validation before setup
        ("setup",),
        ("window", 2, 0),
        ("sync",),
        ("validate", 2),
        ("resume",),
        ("window", 2, 2),
        ("sync",),
        ("validate", 4),
        ("resume",),
        ("teardown",),
        ("validate", 4),  # post-training validation
        ("cleanup",),
    ]


def test_train_divergence_skips_teardown_but_cleans_up():
    host = _LoopHost(num_outer_steps=5, diverge_after=1)
    asyncio.run(host.train())
    assert host.windows_run == 2
    assert ("teardown",) not in host.calls
    assert host.calls[-1] == ("cleanup",)
    # Only the first (healthy) window was synced and validated.
    assert host.calls.count(("sync",)) == 1
    assert [c for c in host.calls if c[0] == "validate"] == [
        ("validate", 0),
        ("validate", 2),
    ]


def test_train_time_bound_and_sync_retarget():
    # Time bound: the loop stops at the deadline, still tears down + cleans up.
    host = _LoopHost(train_seconds=0.08, window_s=0.05)
    asyncio.run(host.train())
    assert 1 <= host.windows_run <= 3
    assert host.calls[-1] == ("cleanup",)
    assert ("teardown",) in host.calls

    # _window_sync may retarget sync_every (a DyLU recommendation): the next
    # window adopts it.
    class _DyLUHost(_LoopHost):
        async def _window_sync(self, t0):
            self._sync_every = 5
            return None

    host = _DyLUHost(num_outer_steps=2, sync_every=2)
    asyncio.run(host.train())
    windows = [c for c in host.calls if c[0] == "window"]
    assert windows == [("window", 2, 0), ("window", 5, 2)]


# === train.py (PerHostProvisioner) =========================================


def _bootstrap_devices(bootstrap, monkeypatch):
    """Run a bootstrap callable and return the CUDA_VISIBLE_DEVICES it set."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    bootstrap()
    return os.environ["CUDA_VISIBLE_DEVICES"]


def test_provisioner_slices_pool_and_rejects_over_allocation(monkeypatch):
    # Whitespace-tolerant parse, then multi- and single-GPU slices carved out
    # of the pool in order.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " 1 , 4 ,6,7")
    prov = PerHostProvisioner(total_gpus=4)
    assert _bootstrap_devices(prov.allocate(2), monkeypatch) == "1,4"
    assert _bootstrap_devices(prov.allocate(1), monkeypatch) == "6"
    assert _bootstrap_devices(prov.allocate(1), monkeypatch) == "7"
    # The pool is exhausted.
    with pytest.raises(RuntimeError, match="only 0 available"):
        prov.allocate(1)

    # A replica asked to spawn more meshes than its CUDA_VISIBLE_DEVICES slice
    # can hold (e.g. 1 trainer + 2 engines on a 2-GPU slice).
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    prov = PerHostProvisioner(total_gpus=3)
    prov.allocate(1)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES exposes"):
        prov.allocate(2)


# === __init__.py ============================================================


def test_package_import_stays_cpu_light():
    """The heloco parameter server and async_inference relay server are
    CPU-only processes that import their parent packages; a bare package
    import must not pull vLLM/monarch or the RL actor stack (which is why
    the __init__.py files re-export nothing)."""
    code = (
        "import sys; "
        "import torchtitan.experiments.decentralized_rl, "
        "torchtitan.experiments.decentralized_rl.server, "
        "torchtitan.experiments.decentralized_rl.relay, "
        "torchtitan.experiments.decentralized_rl.rollout_queue, "
        "torchtitan.experiments.decentralized_rl.heloco_client; "
        "heavy = [m for m in sys.modules if m == 'vllm' or m == 'monarch' "
        "or m.startswith('torchtitan.experiments.rl.actors')]; "
        "assert not heavy, heavy; print('light')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert out.returncode == 0, out.stderr
    assert "light" in out.stdout


def test_hf_backend_registry_resolution():
    """The HF transformers backend resolves through the same table-driven
    contract: dims come from the checkpoint's config.json (not derived), the
    titan-shaped layers view satisfies the trainer/generator assertion
    expression, and the near-identity state-dict adapter is wired."""
    from torchtitan.models.common.attention import FlexAttention

    cfg = base_rl_config(model="hf", flavor="Qwen3-0.6B")
    spec = cfg.model_spec
    # the exact expression asserted by rl/actors/trainer.py and generator.py
    # (the HF backend routes attention through its flex path)
    inner = spec.model.layers[0].attention.inner_attention
    assert isinstance(inner, FlexAttention.Config)
    attn = spec.model.layers[0].attention
    assert attn.head_dim == 128, "must come from config.json, not dim/n_heads"
    assert attn.n_kv_heads == 8
    assert spec.state_dict_adapter is not None
    assert cfg.renderer.name == "auto"
    assert cfg.hf_assets_path.endswith("Qwen3-0.6B")
    # trained untied (FSDP); the adapter aliases embeddings into lm_head at load
    assert spec.model.tie_word_embeddings is False


def test_hf_backend_covers_every_strategy():
    """Every coordination strategy (and both decoupled worker roles) has an HF
    preset that is a pure model/flavor redirect of its native counterpart —
    the strategies themselves are model-agnostic, so the redirect plus the
    registry tables is the whole integration surface. Resolve each preset and
    check the HF markers that distinguish it from a native config."""
    from torchtitan.experiments.decentralized_rl.config_registry import (
        rl_async_inference_hf_qwen3_0_6b,
        rl_async_inference_worker_hf_qwen3_0_6b,
        rl_diloco_hf_qwen3_0_6b,
        rl_heloco_async_inference_hf_qwen3_0_6b,
        rl_heloco_async_inference_worker_hf_qwen3_0_6b,
        rl_heloco_hf_qwen3_0_6b,
    )

    for preset in (
        rl_diloco_hf_qwen3_0_6b,
        rl_heloco_hf_qwen3_0_6b,
        rl_async_inference_hf_qwen3_0_6b,
        rl_heloco_async_inference_hf_qwen3_0_6b,
        rl_async_inference_worker_hf_qwen3_0_6b,
        rl_heloco_async_inference_worker_hf_qwen3_0_6b,
    ):
        cfg = preset()
        spec = cfg.model_spec
        assert spec.name == "hf_transformers_rl", preset.__name__
        assert spec.state_dict_adapter is not None, preset.__name__
        assert cfg.renderer.name == "auto", preset.__name__
        assert cfg.hf_assets_path.endswith("Qwen3-0.6B"), preset.__name__
        assert spec.model.tie_word_embeddings is False, preset.__name__


def test_hf_backend_1_7b_flavor_registered():
    """The Qwen3-1.7B HF flavor resolves dims from ITS checkpoint config (not
    0.6B's): 28 layers, hidden 2048, and the shared example_checkpoint dir."""
    cfg = base_rl_config(model="hf", flavor="Qwen3-1.7B")
    assert cfg.hf_assets_path.endswith("Qwen3-1.7B")
    assert cfg.model_spec.model.num_hidden_layers == 28
    assert cfg.model_spec.model.hidden_size == 2048
    assert ("hf", "Qwen3-1.7B") in _DEFAULT_HF_ASSETS_PATH
