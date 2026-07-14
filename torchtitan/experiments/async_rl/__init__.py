# Copyright (c) Panocular AI.
#
# async_rl: decentralized RL post-training on torchtitan's RL actors, one
# subpackage per coordination strategy:
#
#   - torchtitan.experiments.async_rl.heloco -- N workers sync
#     pseudo-gradients through a standalone CPU parameter server with no
#     barrier (HeLoCoRLReplica + HeLoCoRLClient; server: ``python -m
#     torchtitan.experiments.async_rl.heloco.server``).
#   - torchtitan.experiments.async_rl.diloco -- N workers sync through a
#     torchft Manager/Lighthouse quorum, stock synchronous DiLoCo
#     (DiLoCoRLReplica).
#   - torchtitan.experiments.async_rl.prime -- one trainer replica
#     (PrimeReplica: prime-rl-style decoupled generation, arXiv:2505.07291)
#     broadcasts weights outward through a relay-server tier to independent
#     PrimeWorker processes (SHARDCAST-style); workers push generated
#     rollouts back into the trainer's embedded rollout queue.
#
# (A single-worker/no-coordination baseline lives only in
# ___benchmark/local.py, as a benchmark comparison arm -- it's intentionally
# not part of this package.)
#
# All coordinator classes share RLControllerMixin
# (torchtitan.experiments.async_rl.controller) -- the outer train loop, the
# sync_every-step window runner, and the optional LlamaRL-style
# generation/training overlap -- plugging their coordination into its hooks.
# Configuration goes through torchtitan.experiments.async_rl.config_registry
# (``--module async_rl --config <function>``); ``python -m
# torchtitan.experiments.async_rl.train`` launches a worker for any strategy
# (the --config picks it, mirroring torchtitan.experiments.rl.train).
#
# Deliberately no re-exports: importing a coordinator pulls torchtitan's RL
# stack including vLLM (~10s), which the CPU-only processes (heloco parameter
# server, prime relay server) must not pay just for importing their
# parent package. Import classes from their defining submodule.
