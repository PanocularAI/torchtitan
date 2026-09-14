# FORK-DELTA — PanocularAI/torchtitan

What this fork adds on top of upstream `pytorch/torchtitan`, why it cannot all live outside
the fork, and which patches are candidates to send upstream.

Measured vs upstream base `a182e530` (2026-09-29): **12 files, +396 / −44** — and 11 of
those 12 are *modified upstream files*. The only added file is this one.
(It was 17 files / +766 on `1c7ab8089`, and 34 files / +5561 before everything RL moved to
the engine.)

> **This fork now adds no packages of its own.** The HeLoCo parameter server, the relay, the
> rollout queue, the RL presets, the whole of `decentralized_rl`, and the HF-backend RL glue
> all moved to panofabric-engine — including the compatibility shims, which live there now as
> `panoengine.train.rl`. Run specs stored in the control plane's database still say `--module
> decentralized_rl`; controld translates that to the engine path when it builds the argv, so
> neither repo ships a package under that name. `torchtitan/rl/` — the RL implementation
> itself, `experiments/rl/` until upstream #4773 promoted it — stays here, and the engine
> imports it from this fork.

## The cost model

A rebase conflicts only on files **both sides touched**. Added files never conflict. So the
carrying cost of this fork is not the 396 added lines — it is the **44 deleted lines
across 11 modified files**. Shrinking that set is the only thing that makes upstream syncs
cheap.

### Added — never conflicts, stays here

**Nothing but this file.** Every Panocular package that used to live here — the HeLoCo
parameter server, the relay, the rollout queue, the RL presets, all of `decentralized_rl`
(controller, Monarch trainer actors, replica strategies), and the HF-backend RL glue that sat
in `experiments/transformers_modeling_backend/rl/` — is in panofabric-engine now.

The control plane launches the engine's paths directly: `panoengine.train.rl.{train,worker}`
and `panoengine.decentralized.{parameter_server,relay,rollout_queue}`. Run specs already stored
in its database name `--module decentralized_rl`. That name is a **control-plane** value and
never was an importable module, so controld maps it to `panoengine.train.rl` on the way into
argv (`spec/runspec.py`'s `engine_module()`), and neither repo carries a shim package for it.
The spec keeps the value its author wrote; only the argv is translated.

Consequence worth knowing: `panofabric-engine` must be importable wherever a run is launched.
It already was — the presets have lived there for a while — but now the entry points do too, so
**controld and the engine image have to move together**. A controld that names
`panoengine.train.rl.train` cannot launch against an image predating it.

### Modified — the conflict surface

Two files that used to be here are gone from this list: `run_train.sh` (its `FT_ENABLE`
launch wrapper moved to the engine's own `run_train.sh`, so this one is byte-identical to
upstream) and `experiments/__init__.py` (it only registered `decentralized_rl`).


| File | Δ | What we changed |
|---|---|---|
| `torchtitan/experiments/torchft/manager.py` | +134 / −1 | FT manager wiring for the decentralized strategies: `heloco`, rank-0-only sync, PG checkpoint transport, lighthouse hostname. |
| `torchtitan/experiments/torchft/checkpoint.py` | +79 / −1 | Full-state-dict checkpointing for rank-0-only (heterogeneous-island) sync. |
| `torchtitan/experiments/torchft/optimizer.py` | +12 / −1 | `default_ft_adamw`. |
| `torchtitan/experiments/torchft/trainer.py` | +3 / −0 | Passes `pp_enabled` so `heloco` can refuse PP. |
| `torchtitan/config/manager.py` | +31 / −8 | `_import_registry` re-raises a real `ImportError` instead of reporting "module not found". |
| `torchtitan/observability/metrics.py` | +26 / −2 | `StdoutJsonLogger` fallback so runs with neither wandb nor TensorBoard still emit metrics. |
| `torchtitan/rl/controller.py` | +43 / −29 | `_start_rollout_producers` / `_stop_rollout_producers`, so the engine's windowed loop can drive the consume side. |
| `torchtitan/rl/components/work_buffer.py` | +18 / −1 | `pause()` / `resume()`, so rollouts never straddle an outer weight merge. |
| `torchtitan/distributed/utils.py` | +8 / −1 | `set_timeout` compat shim for torch nightlies that only have the private spelling. |
| `torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py` | +19 / −0 | Takes the tie flag from the checkpoint's `config.json`, so a tied checkpoint loads into the untied model the engine builds for every HF-backend preset (FSDP can't shard one weight across two groups). Not RL-specific. |
| `tests/unit_tests/cpu/test_config_manager.py` | +23 / −0 | Covers the `_import_registry` change. |

## The engine owns the decentralized primitives

`experiments/torchft/manager.py` imports `panoengine.decentralized.{async_diloco,heloco}` —
the algorithms live in panofabric-engine. The import is **deferred** (inside the HeLoCo code
path), so nothing here imports the engine at module-init time, and `panofabric-engine[decentralized]` pulls
torchft only, never torchtitan. The dependency direction stays one-way: this fork is an
adapter over the engine's primitives, not their home.

## Why `decentralized_rl` left the fork — and why `experiments/rl` did not

`decentralized_rl` subclassed `torchtitan.experiments.rl`'s `PolicyTrainer` and `Controller`.
Subclassing works across a package boundary, so that alone never required living in-tree —
and measured on the current code those classes override **zero** base methods between them;
every method they define is an addition. They needed those classes as bases, not as things to
patch. So they moved to `panoengine/train/rl/`, with shims left here at every
path a stored spec or a live image can name.

`experiments/rl` itself did NOT move, and a vendored copy of it in the engine was tried and
deleted. Duplicating a directory this fork already carries — including this fork's own two
modified files in it (+61/−30) — means every upstream rebase improves one copy and not the
other. One live copy, here, imported by the engine.

The bill that remains is real: the engine now eats upstream's experimental-API breaks
(`dae3961af compat(rl): adapt decentralized_rl to upstream's post-rebase APIs` is that bill
arriving once already), and it cannot patch upstream in place from over there. It is bounded by
the engine's SHA pin on this fork. If the drift gets expensive, moving the coordinators back
here is the escape hatch.

Keeping a fork is free. Carrying a diff on *modified* files is what costs. Optimize the latter.

## Upstreaming candidates

Each merged PR permanently deletes fork surface. All are small and independently reviewable:

| Patch | Size | Note |
|---|---|---|
| `_import_registry` re-raises the real `ImportError` | ~30 lines | Highest goodwill. Today a *broken* registry is indistinguishable from a *missing* one — "config function not found" when the truth is an ImportError three levels down. |
| `StdoutJsonLogger` metrics fallback | ~25 lines | Runs with neither wandb nor TensorBoard currently drop every metric but loss/tps/mfu. |
| Don't hard-require `rank0_synchronization_only` on the FT manager (`23a45fbb7`) | small | |
| Don't bound a multi-GB relay transfer with a total timeout (`fa05a8f05`) | small | |
| Give the replica/worker mains a stdout log handler (`6a0741b41`) | small | |

**Target: 11 modified files → 2–3.** Then the next upstream sync is minutes, not a branch.

## Rebased onto upstream `a182e530` (2026-09-30)

120 upstream commits on top of `1c7ab8089`. The six-commit stack replayed as six feature
commits plus this file. Everything it lost, it lost to upstream or to a decision:

- **Dropped: the HF transformers-backend RL glue** — the HF branches in
  `rl/model/{vllm_wrapper,vllm_registry,attention}.py`, and `skip_dp` in
  `transformers_modeling_backend/parallelize.py`, whose only caller was that vLLM branch.
  Upstream #4810 deleted `ModelSpec`, the seam both this glue and the engine's
  `hf_model_registry.py` compose at, and the glue only ever served the `rl_*_hf*` A/B
  baseline presets, which never built a model live. The engine retires those presets in
  its port. **Kept** from the same old `fix(hf-backend)` commit: the checkpoint-driven
  tie flag in `state_dict_adapter.py`. It was never RL-only — every engine
  `models/hf_transformers` preset and the tool-calling SFT example load tied checkpoints
  through it.
- **Dropped: a global `torch._dynamo.config.recompile_limit = 10`** in the vLLM wrapper —
  a stale copy of code upstream moved into `models/gpt_oss` in #3737 (June).
- **Dropped, already upstream at `1c7ab8089`:** three commits made on `main` after the
  last rebase — Qwen3.5 text-only special-token lookups and decoder-only
  (`vision_encoder=None`) builds (both upstream #4355/#3957), and wiring `lm_head` for
  `ChunkedLossWrapper` in the FT trainer (upstream did it; since #4773 it lives in the
  shared `TrainingEngine._initialize_model`, which the FT engine inherits).
- **Followed:** `experiments/rl` → `torchtitan/rl` (#4773), `ParallelDims` →
  `ParallelismContext` (#4905), and the FT trainer's move onto `TrainingEngine` (#4773).
  `default_ft_adamw` was rewritten onto the new `AdamW.Config` (#4904 removed
  `ParamGroupConfig` and upstream's own `default_adamw`).

**New required dependencies** since `1c7ab8089`: `renderers==0.1.11` (new) and
`attn-gym[linear]` 0.0.8 → 0.0.13.

## Rebased onto upstream `1c7ab8089` (2026-09-14)

The 31-commit stack was regrouped into 6 feature commits on the old base, then replayed
onto `1c7ab8089`. One patch was **dropped as obsolete**: *defer the `triton` import off
the model-description path*. Upstream's `fc0e81611` (#4627) deprecated
`minimal_async_ep` and deleted the directory (−2571 lines), gutting
`models/common/token_dispatcher.py`; no module-scope `import triton` survives on the
model path, so the patch had nothing left to patch.

Everything else was kept and re-applied. Three upstream refactors absorbed our code
rather than colliding with it:

- `components/checkpoint` → `components/checkpointer`, `components/metrics.py` →
  `observability/metrics.py`, `tests/unit_tests/` → `tests/unit_tests/cpu/` — followed.
- `_replace_vllm_layer_configs` (upstream extracted our inline vLLM attention rewrite
  into a helper **and** added `delta_net` handling for hybrid models) now serves the
  native path; our `_is_hf_backend` branch sits on top of it.
- `resolve_fsdp_mesh` / `resolve_sparse_fsdp_mesh` replaced the inline mesh-name
  construction inside our `skip_dp` guard.
- Upstream's own evolution won where it overlapped ours: the FT optimizer's
  `step`/`zero_grad` quorum dispatch (our stack never touched those), and the
  token-flat "fold batch dim" forward — our HF path re-adds the batch dim locally.

**New required dependencies** this update brings, which the engine image must carry:
`grain==0.2.18` (replaces `torchdata`), `spmd_types==0.2.5` (was 0.2.3),
`attn-gym[linear]==0.0.8` (new), and `torch_remat` — moved from an optional extra to a
**required** git-pinned dep. `flash-linear-attention` is no longer needed: upstream
`02a04013d` (#4389) moved the native `models/qwen3_5` GatedDeltaNet kernels to
Attention Gym.

Nothing in this repo blocks on a PR landing. The fork already carries every patch and keeps
carrying it; each merge is simply a free deletion whenever it happens.

## Consumers

This fork is pinned by SHA (never by branch) from
[panofabric-engine](https://github.com/PanocularAI/panofabric-engine)'s `[train]` extra.
A moving ref in a published dist is not reproducible and breaks outright when the branch is
deleted — which is exactly what happened to the old `@async_rl` ref.
