# SPDX-License-Identifier: Apache-2.0
"""Contracts of confidence budget placeholder scheduling (方案三 v010).

The placeholder publication hands the scheduler the full tree width without
reading the device packet; ``resolve_placeholder_budget`` then patches the
true width into the scheduler output before execution. These tests pin that
the executed shapes are bit-identical to the synchronous path, that error
semantics (LAYOUT_ERROR / slot drift / corruption) still raise before launch,
and that the feature is off unless explicitly enabled.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ascend_config import TreeSpecConfig
from vllm_ascend.worker.v2.spec_decode.tree.budget_policy import PolicyStatus
from vllm_ascend.worker.v2.spec_decode.tree.draft_tokens import TreeDraftTokensHandler
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper

BUDGET = 8
MAX_REQS = 4


@pytest.fixture(autouse=True, scope="module")
def _cuda_api_as_npu():
    # The upstream V2 base handler binds torch.cuda.Stream/Event directly;
    # production wraps torch.cuda with torch.npu for the whole runner. Bare
    # pytest has no CUDA build, so apply the same wrapper here.
    with torch_cuda_wrapper():
        yield


class _FakeSpeculator:
    def __init__(self, device):
        self.budget = BUDGET
        self.tree_budget_packet = torch.zeros((MAX_REQS, 2), dtype=torch.int32, device=device)
        self._budget_produced = torch.npu.Event()
        self.tree_builder = None


class _FakeReqStates:
    def __init__(self):
        self.max_num_reqs = MAX_REQS
        self.req_id_to_index = {}
        self.tree_next_budget_cpu = [None] * MAX_REQS


class _FakeInputBatch:
    has_structured_output_reqs = False

    def __init__(self, req_ids, slots):
        self.req_ids = list(req_ids)
        self.idx_mapping_np = list(slots)


def _make_handler(device, **kwargs):
    req_states = _FakeReqStates()
    speculator = _FakeSpeculator(device)
    handler = TreeDraftTokensHandler(
        device,
        req_states,
        speculator,
        verification_policy="confidence",
        threshold=0.2,
        budget_stats=None,
        **kwargs,
    )
    return handler, req_states, speculator


def _produce(handler, speculator, req_states, rows):
    """Seal one step of production: rows maps req_id -> (budget, status)."""
    req_ids = tuple(rows)
    slots = tuple(range(len(req_ids)))
    for slot, req_id in enumerate(req_ids):
        speculator.tree_budget_packet[slot] = torch.tensor(rows[req_id], dtype=torch.int32)
        req_states.req_id_to_index[req_id] = slot
    speculator._budget_produced.record()
    handler.set_draft_tokens(_FakeInputBatch(req_ids, slots), torch.zeros((len(req_ids), BUDGET), dtype=torch.int32))
    handler.seal_budget_production()
    return req_ids


def _placeholder_output(req_ids, full=BUDGET):
    spec = {req_id: [-1] * full for req_id in req_ids}
    num = {req_id: 1 + full for req_id in req_ids}
    return SimpleNamespace(
        scheduled_spec_decode_tokens=spec,
        num_scheduled_tokens=num,
        total_num_scheduled_tokens=sum(num.values()),
    )


def test_placeholder_publication_defers_packet_read():
    device = torch.device("npu:0")
    handler, req_states, speculator = _make_handler(device, budget_placeholder_schedule=True)
    _produce(handler, speculator, req_states, {"r0": (3, PolicyStatus.CONFIDENCE)})
    out = handler.get_draft_tokens()
    assert out is not None
    assert out.req_ids == ["r0"]
    assert len(out.draft_token_ids[0]) == BUDGET
    assert all(t == -1 for t in out.draft_token_ids[0])
    # The packet stays pending for the runner-side patch; nothing was consumed.
    assert handler._placeholder_outstanding is not None
    assert handler._pending_budget is not None
    assert req_states.tree_next_budget_cpu[0] is None


def test_resolve_patches_width_and_accounting():
    device = torch.device("npu:0")
    handler, req_states, speculator = _make_handler(device, budget_placeholder_schedule=True)
    _produce(handler, speculator, req_states, {"r0": (3, PolicyStatus.CONFIDENCE), "r1": (8, PolicyStatus.CONFIDENCE)})
    handler.get_draft_tokens()
    out = _placeholder_output(["r0", "r1"])
    wait = handler.resolve_placeholder_budget(out)
    assert wait >= 0.0
    assert len(out.scheduled_spec_decode_tokens["r0"]) == 3
    assert len(out.scheduled_spec_decode_tokens["r1"]) == 8  # full width: untouched
    assert out.num_scheduled_tokens == {"r0": 4, "r1": 9}
    assert out.total_num_scheduled_tokens == 13
    assert req_states.tree_next_budget_cpu[0] == 3
    assert req_states.tree_next_budget_cpu[1] == 8
    # Pending state fully consumed; a second resolve is a no-op.
    assert handler._pending_budget is None
    assert handler._placeholder_outstanding is None
    assert handler.resolve_placeholder_budget(out) == 0.0


def test_resolve_bootstrap_row_keeps_full_width():
    device = torch.device("npu:0")
    handler, req_states, speculator = _make_handler(device, budget_placeholder_schedule=True)
    _produce(handler, speculator, req_states, {"r0": (BUDGET, PolicyStatus.BOOTSTRAP_FIXED)})
    handler.get_draft_tokens()
    out = _placeholder_output(["r0"])
    handler.resolve_placeholder_budget(out)
    assert len(out.scheduled_spec_decode_tokens["r0"]) == BUDGET
    assert out.total_num_scheduled_tokens == 1 + BUDGET


def test_resolve_layout_error_raises_before_mutation():
    device = torch.device("npu:0")
    handler, req_states, speculator = _make_handler(device, budget_placeholder_schedule=True)
    _produce(handler, speculator, req_states, {"r0": (5, PolicyStatus.LAYOUT_ERROR)})
    handler.get_draft_tokens()
    out = _placeholder_output(["r0"])
    with pytest.raises(ValueError, match="invalid confidence budget or tree layout"):
        handler.resolve_placeholder_budget(out)
    assert len(out.scheduled_spec_decode_tokens["r0"]) == BUDGET
    assert out.total_num_scheduled_tokens == 1 + BUDGET


def test_resolve_slot_mismatch_raises():
    device = torch.device("npu:0")
    handler, req_states, speculator = _make_handler(device, budget_placeholder_schedule=True)
    _produce(handler, speculator, req_states, {"r0": (5, PolicyStatus.CONFIDENCE)})
    handler.get_draft_tokens()
    req_states.req_id_to_index["r0"] = 2  # slot changed after publication
    with pytest.raises(RuntimeError, match="slot changed"):
        handler.resolve_placeholder_budget(_placeholder_output(["r0"]))


def test_resolve_rejects_corrupted_placeholder():
    device = torch.device("npu:0")
    handler, req_states, speculator = _make_handler(device, budget_placeholder_schedule=True)
    _produce(handler, speculator, req_states, {"r0": (5, PolicyStatus.CONFIDENCE)})
    handler.get_draft_tokens()
    out = _placeholder_output(["r0"])
    out.scheduled_spec_decode_tokens["r0"] = [-1] * (BUDGET - 1)  # corrupted width
    with pytest.raises(RuntimeError, match="corrupted"):
        handler.resolve_placeholder_budget(out)


def test_resolve_without_outstanding_is_noop():
    device = torch.device("npu:0")
    handler, _, _ = _make_handler(device, budget_placeholder_schedule=True)
    out = _placeholder_output(["r0"])
    assert handler.resolve_placeholder_budget(out) == 0.0
    assert len(out.scheduled_spec_decode_tokens["r0"]) == BUDGET


def test_placeholder_off_by_default_keeps_sync_path():
    device = torch.device("npu:0")
    handler, req_states, speculator = _make_handler(device)
    _produce(handler, speculator, req_states, {"r0": (3, PolicyStatus.CONFIDENCE)})
    out = handler.get_draft_tokens()
    assert out is not None
    assert len(out.draft_token_ids[0]) == 3  # synchronous path reads the true width
    assert handler._placeholder_outstanding is None
    assert req_states.tree_next_budget_cpu[0] == 3


def test_config_validation():
    base = dict(enabled=True, budget=BUDGET, topk=4)
    cfg = TreeSpecConfig(
        **base, verification_policy="confidence", verification_threshold=0.2,
        budget_placeholder_schedule=True,
    )
    assert cfg.budget_placeholder_schedule is True
    with pytest.raises(ValueError, match="budget_placeholder_schedule requires"):
        TreeSpecConfig(**base, verification_policy="fixed", budget_placeholder_schedule=True)
