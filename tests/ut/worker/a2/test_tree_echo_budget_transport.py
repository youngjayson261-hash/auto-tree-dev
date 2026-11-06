# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU stream spies check ordering; real NPU transport is a separate test."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

from vllm_ascend.worker.v2.spec_decode.tree.draft_tokens import TreeDraftTokensHandler


@pytest.fixture
def transport(monkeypatch):
    calls = []

    class Event:
        def record(self, *args):
            calls.append("record")

        def synchronize(self):
            calls.append("sync")

    stream = SimpleNamespace(wait_event=lambda event: calls.append("wait"))

    def init(handler, device):
        handler.copy_stream = stream
        handler.copy_event = Event()

    empty = torch.empty
    copy = torch.Tensor.copy_

    def allocate(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    def copy_packet(target, source, **kwargs):
        calls.append("copy")
        return copy(target, source, **kwargs)

    monkeypatch.setattr(DraftTokensHandler, "__init__", init)
    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_packet)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: calls.append("retain"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=Event, stream=lambda stream: nullcontext()), raising=False)
    state = SimpleNamespace(max_num_reqs=3, req_id_to_index={"a": 1, "b": 0}, tree_next_budget_cpu=[None] * 3)
    spec = SimpleNamespace(tree_budget_packet=torch.tensor([[2, 0], [0, 0], [8, 0]], dtype=torch.int32), budget=8)
    handler = TreeDraftTokensHandler(torch.device("cpu"), state, spec, verification_policy="confidence", threshold=0.1)
    batch = SimpleNamespace(req_ids=["a", "b"], idx_mapping_np=np.array([1, 0]), has_structured_output_reqs=False)
    return handler, state, spec, batch, calls


def test_one_copy_after_seal_and_frozen_identity(transport):
    handler, state, spec, batch, calls = transport
    handler.set_draft_tokens(batch, None)
    assert calls == []
    batch.req_ids[:] = ["mutated", "ids"]
    batch.idx_mapping_np[:] = [0, 1]
    with pytest.raises(RuntimeError, match="sealed"):
        handler.get_draft_tokens()
    handler.seal_budget_production()
    assert calls == ["record"]
    result = handler.get_draft_tokens()
    assert result.req_ids == ["a", "b"]
    assert result.draft_token_ids == [[-1, -1], []]
    assert state.tree_next_budget_cpu == [0, 2, None]
    assert calls == ["record", "wait", "copy", "retain", "record", "sync"]
    assert handler.get_draft_tokens() is None


def test_cancelled_request_skipped(transport):
    handler, state, spec, batch, calls = transport
    handler.set_draft_tokens(batch, None)
    del state.req_id_to_index["a"]
    handler.seal_budget_production()
    assert handler.get_draft_tokens().req_ids == ["b"]
    assert state.tree_next_budget_cpu == [0, None, None]


@pytest.mark.parametrize("fault", ["slot", "count", "status"])
def test_transport_rejects_corruption_without_partial_state(transport, fault):
    handler, state, spec, batch, calls = transport
    handler.set_draft_tokens(batch, None)
    if fault == "slot":
        state.req_id_to_index["b"] = 2
    elif fault == "count":
        spec.tree_budget_packet[1, 0] = 9
    else:
        spec.tree_budget_packet[1, 1] = 4
    handler.seal_budget_production()
    with pytest.raises((RuntimeError, ValueError)):
        handler.get_draft_tokens()
    assert state.tree_next_budget_cpu == [None] * 3


def test_reentry_and_grammar_rejected(transport):
    handler, state, spec, batch, calls = transport
    batch.has_structured_output_reqs = True
    with pytest.raises(ValueError, match="structured"):
        handler.set_draft_tokens(batch, None)
    batch.has_structured_output_reqs = False
    handler.set_draft_tokens(batch, None)
    with pytest.raises(RuntimeError, match="pending"):
        handler.set_draft_tokens(batch, None)
