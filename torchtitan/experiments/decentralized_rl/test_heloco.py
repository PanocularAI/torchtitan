# Copyright (c) Panocular AI.
#
# HeLoCoRLClient against a real localhost AsyncDiLoCoServer (the HTTP /sync
# protocol end to end), the applied=False re-baseline path (stubbed wire),
# and build_server's outer-method dispatch.

import pytest
import torch
from torch import nn

import torchtitan.experiments.decentralized_rl.server as server_mod
from torchtitan.experiments.decentralized_rl.heloco_client import HeLoCoRLClient
from torchtitan.experiments.decentralized_rl.server import build_server, param_metadata


@pytest.mark.parametrize("should_quantize", [False, True])
def test_real_server_roundtrip(should_quantize):
    """End-to-end over the real HTTP /sync protocol: pull adopts the server's
    weights; push applies the outer step and returns the updated global.
    With plain SGD(lr=1) the outer step theta - 1.0 * (theta - theta_local)
    lands exactly on theta_local, so the expected result is exact (up to int8
    quantization error on the uploaded pseudo-gradient when quantizing)."""
    from torchft.async_diloco import AsyncDiLoCoServer

    torch.manual_seed(0)
    model = nn.Linear(4, 3)  # fp32 CPU
    server = AsyncDiLoCoServer(
        model,
        torch.optim.SGD(model.parameters(), lr=1.0),
        port=0,
        should_quantize=should_quantize,
    )
    try:
        names, shapes, dtypes = param_metadata(model)
        assert names == [n for n, _ in model.named_parameters()]
        # Reported as the server's storage dtype regardless of model dtype.
        assert all(dt is torch.float32 for dt in dtypes.values())
        client = HeLoCoRLClient(
            server.address(),
            names,
            shapes,
            dtypes,
            should_quantize=should_quantize,
            sync_timeout=10.0,
        )

        pulled = client.pull()
        for name, p in model.named_parameters():
            assert torch.equal(pulled[name], p.detach())

        local = {name: p.detach() + 0.25 for name, p in model.named_parameters()}
        pushed = client.push(local, speed=1.0)
        for name in names:
            # Quantization error bound: scale/2 = max|Delta|/254 per block;
            # Delta is uniformly 0.25 here so the bound is tiny but not zero.
            tol = 0.0 if not should_quantize else 0.25 / 254 + 1e-7
            assert torch.allclose(pushed[name], local[name], atol=tol, rtol=0)
        assert client._baseline_revision == 1
    finally:
        server.shutdown()


def test_push_rejected_by_server_rebaselines():
    """applied=False (stale baseline, e.g. server checkpoint restore): the
    push is dropped but the response is still adopted as a re-baseline."""
    client = HeLoCoRLClient(
        "http://fake/sync",
        ["w", "b"],
        {"w": torch.Size([2]), "b": torch.Size([1])},
        {"w": torch.float32, "b": torch.float32},
    )
    client._global_params["w"].copy_(torch.tensor([1.0, 2.0]))
    # Replace the parent's HTTP roundtrip with a canned rejection response.
    client._session_roundtrip = lambda flag, speed, flat_grads: (
        torch.tensor([9.0, 9.0, 9.0]),
        0,
        42,
        False,
    )

    got = client.push({"w": torch.zeros(2), "b": torch.zeros(1)})

    assert torch.equal(got["w"], torch.tensor([9.0, 9.0]))
    assert client._baseline_revision == 42


def test_build_server_selects_outer_method(monkeypatch):
    class _StubServer:
        def __init__(self, model, outer, **kwargs):
            self.model, self.outer, self.kwargs = model, outer, kwargs

    class _StubServer2(_StubServer):
        pass

    monkeypatch.setattr(server_mod, "HeLoCoServer", _StubServer)
    monkeypatch.setattr(server_mod, "AsyncDiLoCoServer", _StubServer2)

    model = nn.Linear(2, 2)
    heloco = build_server(model, outer_method="heloco", should_quantize=True)
    assert type(heloco) is _StubServer
    assert type(heloco.outer).__name__ == "HeLoCoOptimizer"
    assert heloco.kwargs["should_quantize"] is True

    diloco = build_server(nn.Linear(2, 2), outer_method="diloco")
    assert type(diloco) is _StubServer2
    assert type(diloco.outer).__name__ == "DelayedNesterovOptimizer"

    with pytest.raises(ValueError, match="unknown outer_method"):
        build_server(nn.Linear(2, 2), outer_method="bogus")
