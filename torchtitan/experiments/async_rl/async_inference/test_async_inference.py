# Copyright (c) Panocular AI.
#
# async-inference relay swarm (prime-rl): relay sharding round-trip/integrity,
# RelayServer publish/fetch, the rollout-return queue wire protocol,
# AsyncInferenceWorker's free-running poll loop, and the pure-learner
# AsyncInferenceReplica (remote-rollout consumer, fail-fast buffer liveness,
# staleness dropping against the published checkpoint version, publish
# cadence, and an end-to-end train loop that proves zero local generation).
# CPU-only: no GPU, no vLLM, no Monarch actors -- everything above the wire
# is faked.

import asyncio
import itertools
import pickle
from types import SimpleNamespace

import aiohttp
import pytest
import torch
from aiohttp.test_utils import TestServer

import torchtitan.experiments.async_rl.async_inference.worker as worker_mod
from torchtitan.experiments.async_rl.async_inference.relay import (
    build_manifest,
    reassemble_state_dict,
    RelayClient,
    RelayServer,
    shard_state_dict,
    ShardIntegrityError,
    verify_shard,
)
from torchtitan.experiments.async_rl.async_inference.trainer import (
    AsyncInferenceReplica,
    RolloutQueueServer,
)
from torchtitan.experiments.async_rl.async_inference.worker import (
    AsyncInferenceWorker,
    RolloutQueueClient,
)


def _ep(fn):
    """A Monarch-endpoint stand-in: exposes .call(...)."""
    return SimpleNamespace(call=fn)


def _state_dict():
    return {
        "layer.weight": torch.randn(8, 4),
        "layer.bias": torch.randn(8),
        "small": torch.randn(1),
    }


# --------------------------------------------------------------------- #
# relay.py: sharding.
# --------------------------------------------------------------------- #


def test_shard_reassemble_round_trip_and_integrity():
    sd = _state_dict()
    shards = shard_state_dict(sd, num_shards=3)
    assert len(shards) == 3
    # 3 shards for 3 tensors of very different sizes -- balanced bin-packing
    # should not dump everything into one shard.
    sizes = [len(s) for s in shards]
    assert max(sizes) < sum(sizes)  # no single shard holds everything
    assert all(size > 0 for size in sizes)  # torch.save({}) still writes bytes

    manifest = build_manifest(version=3, shard_bytes=shards)
    assert manifest.version == 3 and manifest.num_shards == 3
    assert len(manifest.shard_checksums) == 3 and len(manifest.shard_sizes) == 3

    restored = reassemble_state_dict(shards, manifest)
    assert restored.keys() == sd.keys()
    for name in sd:
        assert torch.equal(restored[name], sd[name])

    # A corrupted shard is rejected by checksum, both when verified alone and
    # at reassembly.
    corrupted = bytearray(shards[0])
    corrupted[0] ^= 0xFF
    with pytest.raises(ShardIntegrityError, match="checksum mismatch"):
        verify_shard(0, bytes(corrupted), manifest)
    with pytest.raises(ShardIntegrityError):
        reassemble_state_dict([bytes(corrupted), shards[1], shards[2]], manifest)


# --------------------------------------------------------------------- #
# relay.RelayServer (aiohttp's own TestServer -- a real loopback socket,
# aiohttp's documented way to test a web.Application without manual binding).
# --------------------------------------------------------------------- #


async def _start_relay(retain_last=5):
    relay = RelayServer(retain_last=retain_last)
    server = TestServer(relay.app())
    await server.start_server()
    base_url = str(server.make_url("")).rstrip("/")
    return relay, server, base_url


def test_relay_server_publish_and_fetch_round_trip():
    async def scenario():
        relay, server, base_url = await _start_relay(retain_last=2)
        try:
            client = RelayClient([base_url])
            assert await client.fetch_latest() is None  # nothing published yet

            sd = _state_dict()
            shards = shard_state_dict(sd, num_shards=2)
            manifest = build_manifest(7, shards)
            await client.publish(7, shards, manifest)

            assert relay.latest_version() == 7
            assert relay.get_manifest(7).to_json() == manifest.to_json()
            assert relay.get_shard(7, 0) == shards[0]
            assert relay.get_shard(7, 1) == shards[1]

            result = await client.fetch_latest(min_version=0)
            assert result is not None
            version, restored = result
            assert version == 7
            for name in sd:
                assert torch.equal(restored[name], sd[name])

            # min_version gating: nothing newer than what's published.
            assert await client.fetch_latest(min_version=7) is None
            assert await client.fetch_latest(min_version=10) is None

            # retain_last=2: two more publishes evict version 7.
            for version in (8, 9):
                one_shard = shard_state_dict(sd, num_shards=1)
                await client.publish(
                    version, one_shard, build_manifest(version, one_shard)
                )
            assert relay.get_manifest(7) is None
            assert relay.latest_version() == 9
        finally:
            await server.close()

    asyncio.run(scenario())


# --------------------------------------------------------------------- #
# The rollout-return queue (workers -> trainer).
# --------------------------------------------------------------------- #


async def _start_rollout_queue(maxsize=64):
    server = RolloutQueueServer(maxsize=maxsize)
    test_server = TestServer(server.app())
    await test_server.start_server()
    base_url = str(test_server.make_url("")).rstrip("/")
    return server, test_server, base_url


def test_rollout_queue_wire_protocol_and_client_robustness():
    """push/consume round trip; a full queue rejects the push (client sees
    False, server counts it); malformed payloads get 400 without crashing
    the server; and send() reports failure as False rather than raising --
    a dead trainer endpoint must not take the worker down."""

    async def scenario():
        # Nothing listening: send() must return False, not raise.
        assert await RolloutQueueClient("http://localhost:1").send(0, 1, ["g"]) is False
        server, test_server, base_url = await _start_rollout_queue(maxsize=1)
        try:
            client = RolloutQueueClient(base_url)
            accepted = await client.send(worker_id=3, version=7, groups=["g1", "g2"])
            assert accepted is True
            # Nothing has consumed yet, so the queue (maxsize=1) is now full.
            assert await client.send(0, 8, ["g3"]) is False
            assert server.num_received == 1
            assert server.num_rejected == 1

            worker_id, version, groups = await server.get()
            assert worker_id == 3 and version == 7 and groups == ["g1", "g2"]

            async with aiohttp.ClientSession() as session:
                for payload in (
                    b"not a pickle",  # UnpicklingError
                    pickle.dumps(12345),  # unpickles fine but isn't a tuple: TypeError
                    pickle.dumps(("only", "two")),  # wrong arity: ValueError
                ):
                    async with session.post(
                        f"{base_url}/rollouts", data=payload
                    ) as resp:
                        assert resp.status == 400
            assert server.num_received == 1  # nothing malformed was enqueued
        finally:
            await test_server.close()

    asyncio.run(scenario())


# --------------------------------------------------------------------- #
# AsyncInferenceReplica's remote-rollout consumer: worker-pushed batches must land
# in the SAME buffer the local per-engine producers feed, tagged with the
# batch's version exactly like a local producer tags its own puts.
# --------------------------------------------------------------------- #


class _FakeRelayClient:
    """Stands in for RelayClient on both ends of the relay tier: records
    publishes (trainer side) and pops canned fetch_latest results (worker
    side)."""

    def __init__(self, results=()):
        self.published = []
        self._results = list(results)

    async def publish(self, version, shard_bytes, manifest):
        self.published.append((version, len(shard_bytes), manifest))

    async def fetch_latest(self, min_version=0):
        return self._results.pop(0) if self._results else None


def make_async_inference_replica(*, publish_every=1):
    """A pure-learner AsyncInferenceReplica with only the window-sync/publish
    state, skipping RLTrainer.__init__ (no actors, no generators)."""
    r = object.__new__(AsyncInferenceReplica)
    r.config = SimpleNamespace(publish_every=publish_every, num_shards=2)
    r._relay_client = _FakeRelayClient()
    r._checkpoint_version = 0
    r._window_count = 0
    sd = _state_dict()

    async def fake_get_full_state_dict_cpu():
        return {0: sd}  # {rank: value}, so inherited _get_rank_0_value (.get(0)) works

    r.trainer = SimpleNamespace(
        get_full_state_dict_cpu=_ep(fake_get_full_state_dict_cpu)
    )
    r._buffer = asyncio.Queue()
    r._num_dropped = 0
    return r


def test_window_sync_publishes_only_on_boundary():
    async def scenario():
        r = make_async_inference_replica(publish_every=2)

        stats1 = await r._window_sync(t0=0.0)
        assert stats1.startswith("buffer: depth=0 dropped=0")  # base stats line
        assert "relay: published" not in stats1  # window 1: not a boundary
        assert r._relay_client.published == []

        stats2 = await r._window_sync(t0=0.0)
        assert "relay: published v1" in stats2
        assert len(r._relay_client.published) == 1
        version, num_shards, manifest = r._relay_client.published[0]
        assert version == 1 and num_shards == 2 and manifest.version == 1

        stats3 = await r._window_sync(t0=0.0)
        assert "relay: published" not in stats3
        assert len(r._relay_client.published) == 1  # still just one publish

    asyncio.run(scenario())


# --------------------------------------------------------------------- #
# AsyncInferenceWorker poll loop.
# --------------------------------------------------------------------- #


class _FakeRollouter:
    def __init__(self):
        self.groups_run = []

    def get_training_sample(self):
        return "sample"

    async def run_group_rollouts(
        self, *, generate_fn, sample, group_id, group_size, sampling, renderer
    ):
        self.groups_run.append(group_id)
        return group_id  # stands in for a real RolloutGroup


class _FakeRolloutQueueClient:
    def __init__(self):
        self.sent = []

    async def send(self, worker_id, version, groups):
        self.sent.append((worker_id, version, list(groups)))
        return True


class _FakeTorchStore:
    def __init__(self):
        self.puts = []

    async def put_state_dict(self, state_dict, key):
        self.puts.append((key, state_dict))


def make_worker(monkeypatch, results, *, num_rounds=0, groups_per_round=1):
    fake_ts = _FakeTorchStore()
    monkeypatch.setattr(worker_mod, "ts", fake_ts)

    pull_calls = []

    async def fake_pull(version):
        pull_calls.append(version)

    w = object.__new__(AsyncInferenceWorker)
    w.config = SimpleNamespace(
        worker_id=0,
        num_rounds=num_rounds,
        poll_interval_s=0,
        groups_per_round=groups_per_round,
        group_size=1,
        round_slowdown_factor=1.0,
    )
    w._version = 0
    w._relay_client = _FakeRelayClient(results)
    w._rollout_queue_client = _FakeRolloutQueueClient()
    w.generator = SimpleNamespace(pull_model_state_dict=_ep(fake_pull))
    w._rollouter = _FakeRollouter()
    w._sampling = None
    w.renderer = None
    return w, fake_ts, pull_calls


def test_worker_free_runs_without_a_newer_checkpoint(monkeypatch):
    """Regression for the heloco_async_inference window-0 deadlock: the
    worker loads the first checkpoint (v1), then keeps generating rounds at
    that SAME version even though the relay never publishes anything newer
    (fetch_latest returns None forever after the first). A checkpoint-gated
    worker would produce exactly ONE round and then idle, starving a trainer
    whose only rollout source is this worker."""
    # Only ONE checkpoint ever exists; every later poll returns None.
    results = [(1, {"w": torch.zeros(1)}), None, None, None]
    w, fake_ts, pull_calls = make_worker(
        monkeypatch, results, num_rounds=3, groups_per_round=1
    )

    asyncio.run(w.run())

    # Three rounds ran despite only one checkpoint fetch, all tagged v1.
    assert pull_calls == [1]  # loaded the checkpoint exactly once
    assert w._version == 1
    assert len(w._rollouter.groups_run) == 3  # 1 group/round x 3 rounds
    sent = w._rollout_queue_client.sent
    assert [(wid, ver, len(g)) for wid, ver, g in sent] == [
        (0, 1, 1),
        (0, 1, 1),
        (0, 1, 1),
    ]


def _group(num_tokens=4, reward=0.5):
    turn = SimpleNamespace(
        prompt_token_ids=[0] * 2, completion_token_ids=[0] * (num_tokens - 1)
    )
    return SimpleNamespace(rollouts=[SimpleNamespace(turns=[turn], reward=reward)])


async def _noop(*a, **k):
    return None


async def _return(value):
    return value


def make_replica(*, max_staleness=4, checkpoint_version=10):
    """A pure-learner AsyncInferenceReplica with only the consumer state,
    skipping RLTrainer.__init__ (no actors, no generators)."""
    r = object.__new__(AsyncInferenceReplica)
    r.config = SimpleNamespace(
        replica_id=0,
        sync_every=2,
        num_outer_steps=1,
        train_seconds=0.0,
        num_groups_per_rollout_batch=2,
        buffer_groups=8,
        max_staleness=max_staleness,
        queue_poll_interval_s=0,
    )
    r._buffer = asyncio.Queue(maxsize=8)
    r._num_dropped = 0
    r._checkpoint_version = checkpoint_version  # the staleness reference
    return r


async def _boom():
    raise RuntimeError("consumer exploded")


@pytest.mark.parametrize(
    "consumer, match",
    [
        (_boom, "remote rollout consumer died"),
        (_noop, "remote rollout consumer exited"),
    ],
)
def test_buffer_get_fails_fast_when_consumer_dies(consumer, match):
    async def scenario():
        r = make_replica()
        r._remote_consumer_task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)
        with pytest.raises(RuntimeError, match=match):
            await r._buffer_get_checked()

    asyncio.run(scenario())


def test_collect_and_build_drops_groups_stale_against_checkpoint_version():
    async def scenario():
        r = make_replica(max_staleness=4, checkpoint_version=10)
        r.trainer = SimpleNamespace(sync_log_step=_ep(lambda step: _noop()))
        r.trainer_dp_degree = 1
        r.batcher = SimpleNamespace(
            num_tokens_target=lambda dp: 4,
            batch=lambda eps, dp_degree: ([["mb"]], 4, None),
        )
        r._build_episodes = lambda groups: ([SimpleNamespace(policy_version=10)], None)
        r._remote_consumer_task = asyncio.create_task(asyncio.sleep(30))
        await r._buffer.put((_group(num_tokens=5), 3))  # 10-3=7 > 4 -> dropped
        await r._buffer.put((_group(num_tokens=5), 8))  # 10-8=2 <= 4 -> kept
        rollout_groups, episodes, mbs, nvt = await asyncio.wait_for(
            r._collect_and_build(1), 1
        )
        assert r._num_dropped == 1
        assert len(rollout_groups) == 1
        r._remote_consumer_task.cancel()

    asyncio.run(scenario())


# --------------------------------------------------------------------- #
# End-to-end controller loop on fakes (mirrors the GPU smoke's shape) for the
# PURE-LEARNER trainer: no generators, consume from a fake embedded queue,
# publish to a fake relay. generator_router is never set -- if any code path
# touched a generator the test would AttributeError, proving zero generation.
# --------------------------------------------------------------------- #


class _InfiniteQueueServer:
    """Embedded rollout queue that always has a batch tagged at ``version``
    (the remote worker pool, never empty)."""

    def __init__(self, *, version=1, groups_per_batch=4):
        self.version = version
        self.groups_per_batch = groups_per_batch

    async def get(self):
        return (
            0,
            self.version,
            [_group(num_tokens=5) for _ in range(self.groups_per_batch)],
        )


def test_train_end_to_end_pure_learner_on_fakes():
    async def scenario():
        r = object.__new__(AsyncInferenceReplica)
        r.config = SimpleNamespace(
            replica_id=0,
            sync_every=2,
            num_outer_steps=2,
            train_seconds=0.0,
            num_groups_per_rollout_batch=2,
            buffer_groups=0,
            max_staleness=4,
            queue_poll_interval_s=0,
            publish_every=999,  # never fires mid-run (initial publish covered elsewhere)
            num_shards=2,
        )
        r._policy_version = 0
        r._checkpoint_version = 1  # bootstrapped by the (mocked) initial publish
        r._window_count = 0
        r._num_dropped = 0
        # Rollouts arrive tagged v1; staleness reference is v1 -> nothing dropped.
        r._rollout_queue_server = _InfiniteQueueServer(version=1)
        r._relay_client = _FakeRelayClient()

        versions = itertools.count(1)

        async def get_full():
            return {0: {"w": 0}}

        r.trainer = SimpleNamespace(
            sync_log_step=_ep(_noop),
            forward_backward=_ep(lambda mb, n: _return({"loss": 0.25})),
            optim_step=_ep(
                lambda: _return(SimpleNamespace(policy_version=next(versions)))
            ),
            get_full_state_dict_cpu=_ep(get_full),
        )
        r._get_rank_0_value = lambda x: x
        r.trainer_dp_degree = 1
        r.batcher = SimpleNamespace(
            num_tokens_target=lambda dp: 4,
            batch=lambda eps, dp_degree: ([["mb"]], 4, None),
        )
        r._build_episodes = lambda groups: (
            [SimpleNamespace(policy_version=r._policy_version)],
            None,
        )
        r._aggregate_validation = lambda metrics: {}

        await asyncio.wait_for(r.train(), 15)

        assert r._policy_version == 4  # 2 windows x sync_every=2 optim steps
        assert r._remote_consumer_task.done()  # cleaned up
        # Never spawned a generator: the attribute was never set, and nothing
        # touched it (else AttributeError would have failed train()).
        assert not hasattr(r, "generator_router")

    asyncio.run(scenario())
