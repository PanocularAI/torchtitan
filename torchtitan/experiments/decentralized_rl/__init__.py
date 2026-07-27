# Copyright (c) Panocular AI.
#
# decentralized_rl: decentralized RL post-training on torchtitan's RL actors. Flat
# layout, one module per role; the four coordination strategies' replica
# classes all live in ``trainers``:
#
#   - DiLoCoRLReplica -- N workers sync through a torchft Manager/Lighthouse
#     quorum, stock synchronous DiLoCo.
#   - HeLoCoRLReplica -- N workers sync pseudo-gradients through a standalone
#     CPU parameter server with no barrier (client: ``heloco_client``;
#     server: ``python -m torchtitan.experiments.decentralized_rl.server``).
#   - AsyncInferenceReplica -- prime-rl-style decoupled generation
#     (arXiv:2505.07291): one pure-learner trainer broadcasts weights outward
#     through a relay-server tier (``relay``) to independent
#     AsyncInferenceWorker processes (``worker``, SHARDCAST-style); workers
#     push rollouts into a standalone queue process (``rollout_queue``).
#   - HeLoCoAsyncInferenceReplica -- both combined: N pure-learner HeLoCo
#     trainers share one rollout queue and coordinate through the parameter
#     server (run with ``--relay_addr`` so it publishes global theta to the
#     relay for the generator pool).
#
# The strategies' Monarch trainer actors (DiLoCoManagerTrainer,
# HeLoCoPolicyTrainer, SnapshotPolicyTrainer) live in ``actors``.
#
# (A single-worker/no-coordination baseline lives only in
# ___benchmark/local.py, as a benchmark comparison arm -- it's intentionally
# not part of this package.)
#
# The windowed coordinator classes share RLControllerMixin
# (torchtitan.experiments.decentralized_rl.controller) -- the outer train loop, the
# sync_every-step window runner, and the optional LlamaRL-style
# generation/training overlap -- plugging their coordination into its hooks.
# Configuration goes through torchtitan.experiments.decentralized_rl.config_registry
# (``--module decentralized_rl --config <function>``); ``python -m
# torchtitan.experiments.decentralized_rl.train`` launches a worker for any strategy
# (the --config picks it, mirroring torchtitan.experiments.rl.train).
#
# Deliberately no re-exports: importing a coordinator pulls torchtitan's RL
# stack including vLLM (~10s), which the CPU-only processes (parameter
# server, relay server, rollout queue) must not pay just for importing their
# parent package. Import classes from their defining submodule.
