# FORK-DELTA — PanocularAI/torchtitan

What this fork adds on top of upstream `pytorch/torchtitan`, why it cannot all live outside
the fork, and which patches are candidates to send upstream.

Measured at `8dab0716` vs upstream base `0b2a804dd`: **34 files, +8032 / −119.**

## The cost model

A rebase conflicts only on files **both sides touched**. Added files never conflict. So the
carrying cost of this fork is not the 8,032 added lines — it is the **119 deleted lines
across 18 modified files**. Shrinking that set is the only thing that makes upstream syncs
cheap.

### Added — never conflicts, stays here

`torchtitan/experiments/decentralized_rl/` — 15 of our 16 new files, one self-contained
directory: the controller, actors, parameter server, relay, replicas, rollout queue, and a
44-function `config_registry` of RL presets.

### Modified — the conflict surface

| File | Δ | What we changed |
|---|---|---|
| `torchtitan/experiments/torchft/manager.py` | +117 / −1 | FT manager wiring for the decentralized strategies. |
| `torchtitan/experiments/torchft/checkpoint.py` | +80 / −1 | Checkpoint handling for fragment-wise sync and HF weight loads. |
| `torchtitan/experiments/torchft/optimizer.py` | +25 / −1 | `default_ft_adamw` and friends. |
| `torchtitan/experiments/torchft/trainer.py` | +3 / −0 | |
| `torchtitan/config/manager.py` | +31 / −8 | `_import_registry` re-raises a real `ImportError` instead of reporting "module not found". |
| `torchtitan/components/metrics.py` | +26 / −2 | `StdoutJsonLogger` fallback so runs with neither wandb nor TensorBoard still emit metrics. |
| `torchtitan/experiments/rl/models/vllm_wrapper.py` | +151 / −47 | vLLM integration for the RL generator. |
| `torchtitan/experiments/rl/controller.py` | +43 / −28 | |
| `torchtitan/experiments/rl/models/attention.py` | +44 / −0 | |
| `torchtitan/experiments/rl/models/vllm_registry.py` | +19 / −0 | |
| `torchtitan/experiments/rl/components/work_buffer.py` | +18 / −1 | |
| `torchtitan/experiments/rl/components/training_sample_builder.py` | +5 / −4 | |
| `torchtitan/experiments/transformers_modeling_backend/parallelize.py` | +26 / −20 | HF-architecture backend fixes. |
| `torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py` | +18 / −0 | Tied-embedding aliasing at load. |
| `torchtitan/distributed/utils.py` | +8 / −1 | |
| `torchtitan/experiments/__init__.py` | +1 / −0 | Registers `decentralized_rl`. |
| `run_train.sh` | +43 / −5 | |
| `tests/unit_tests/test_config_manager.py` | +23 / −0 | Covers the `_import_registry` change. |

## Why `decentralized_rl` cannot leave the fork

It subclasses `torchtitan.experiments.rl`'s `PolicyTrainer` — upstream's *own experimental
directory*, the least stable API surface in the repo. Out of tree we would still eat every
API break (`dae3961af compat(rl): adapt decentralized_rl to upstream's post-rebase APIs` is
that bill already arriving once) **and** lose the ability to patch upstream in place when it
breaks. Strictly worse.

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

**Target: 18 modified files → 2–3.** Then the next upstream sync is minutes, not a branch.

Nothing in this repo blocks on a PR landing. The fork already carries every patch and keeps
carrying it; each merge is simply a free deletion whenever it happens.

## Consumers

This fork is pinned by SHA (never by branch) from
[panofabric-engine](https://github.com/PanocularAI/panofabric-engine)'s `[train]` extra.
A moving ref in a published dist is not reproducible and breaks outright when the branch is
deleted — which is exactly what happened to the old `@async_rl` ref.
