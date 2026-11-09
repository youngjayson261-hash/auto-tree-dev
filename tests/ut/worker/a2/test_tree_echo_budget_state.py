# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise host budget ownership; native device allocation is tested separately."""

import numpy as np
import torch
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.states import AscendRequestState


def test_budget_cleared_on_remove_and_slot_reuse(monkeypatch):
    def init(state, max_num_reqs, *args):
        state.max_num_reqs = max_num_reqs
        state.num_computed_tokens_np = np.zeros(max_num_reqs, dtype=np.int32)
        state.req_id_to_index = {}

    def add(state, req_id, *args, **kwargs):
        state.req_id_to_index[req_id] = 0

    def remove(state, req_id):
        return state.req_id_to_index.pop(req_id, None)

    monkeypatch.setattr(RequestState, "__init__", init)
    monkeypatch.setattr(RequestState, "add_request", add)
    monkeypatch.setattr(RequestState, "remove_request", remove)
    state = AscendRequestState(2, 16, 16, 4, 32, torch.device("cpu"))
    assert state.tree_next_budget_cpu == [None, None]
    state.add_request("old")
    state.tree_next_budget_cpu[0] = 5
    assert state.remove_request("old") == 0
    assert state.tree_next_budget_cpu == [None, None]
    state.tree_next_budget_cpu[0] = 7
    state.add_request("new")
    assert state.tree_next_budget_cpu == [None, None]
    state.tree_next_budget_cpu[0] = 2
    assert state.remove_request("missing") is None
    assert state.tree_next_budget_cpu == [2, None]


def test_batch_budget_is_optional():
    field = AscendInputBatch.__dataclass_fields__["tree_verified_counts_cpu"]
    assert field.default is None


def test_budget_clear_is_limited_to_released_nonzero_slot(monkeypatch):
    added = object()

    def init(state, max_num_reqs, *args):
        state.max_num_reqs = max_num_reqs
        state.num_computed_tokens_np = np.zeros(max_num_reqs, dtype=np.int32)
        state.req_id_to_index = {}

    def add(state, req_id, *, slot):
        state.req_id_to_index[req_id] = slot
        return added

    def remove(state, req_id):
        return state.req_id_to_index.pop(req_id, None)

    monkeypatch.setattr(RequestState, "__init__", init)
    monkeypatch.setattr(RequestState, "add_request", add)
    monkeypatch.setattr(RequestState, "remove_request", remove)
    state = AscendRequestState(4, 16, 16, 4, 32, torch.device("cpu"))
    assert state.add_request("retained", slot=0) is added
    assert state.add_request("old", slot=2) is added
    assert state.add_request("other", slot=3) is added
    state.tree_next_budget_cpu[:] = [4, 5, 6, 7]

    assert state.remove_request("old") == 2
    assert state.tree_next_budget_cpu == [4, 5, None, 7]
    assert state.remove_request("missing") is None
    assert state.tree_next_budget_cpu == [4, 5, None, 7]

    state.tree_next_budget_cpu[2] = 8
    assert state.add_request("replacement", slot=2) is added
    assert state.tree_next_budget_cpu == [4, 5, None, 7]
    assert state.req_id_to_index == {"retained": 0, "other": 3, "replacement": 2}
