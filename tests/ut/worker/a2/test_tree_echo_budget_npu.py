# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.tree.draft_tokens import TreeDraftTokensHandler
from vllm_ascend.worker.v2.spec_decode.tree.shape import TreeShape
from vllm_ascend.worker.v2.states import AscendRequestState
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper


@pytest.mark.skipif(os.environ.get("ECHO_TEST_NPU") != "1", reason="requires explicitly reserved NPU")
def test_native_budget_transport_across_streams():
    pytest.importorskip("torch_npu")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    state = SimpleNamespace(max_num_reqs=2, req_id_to_index={"a": 1, "b": 0}, tree_next_budget_cpu=[None, None])
    spec = SimpleNamespace(budget=8, tree_budget_packet=torch.zeros((2, 2), dtype=torch.int32, device=device))
    with torch_cuda_wrapper():
        handler = TreeDraftTokensHandler(device, state, spec, verification_policy="confidence", threshold=0.1)
    batch = SimpleNamespace(req_ids=["a", "b"], idx_mapping_np=np.array([1, 0]), has_structured_output_reqs=False)
    producer = torch.npu.Stream(device=device)
    for counts in ([3, 0], [0, 2], [1, 1]):
        with torch.npu.stream(producer):
            spec.tree_budget_packet.copy_(torch.tensor([[counts[0], 0], [counts[1], 0]], dtype=torch.int32))
            handler.set_draft_tokens(batch, None)
            handler.seal_budget_production()
        result = handler.get_draft_tokens()
        assert [len(tokens) for tokens in result.draft_token_ids] == list(counts)
        assert state.tree_next_budget_cpu == [counts[1], counts[0]]
        assert handler.get_draft_tokens() is None


@pytest.mark.skipif(os.environ.get("ECHO_TEST_NPU") != "1", reason="requires explicitly reserved NPU")
def test_native_request_budget_lifecycle():
    pytest.importorskip("torch_npu")
    # Native workers install this compatibility layer before allocating state.
    from vllm_ascend.patch.worker.patch_v2 import patch_uva  # noqa: F401

    torch.npu.set_device(0)
    with torch_cuda_wrapper():
        state = AscendRequestState(2, 32, 32, 4, 64, torch.device("npu:0"), TreeShape(4, 8, 2))
        assert state.tree_next_budget_cpu == [None, None]
        state.add_request("old", 2, [1, 2], 0, 8)
        slot = state.req_id_to_index["old"]
        state.tree_next_budget_cpu[slot] = 3
        assert state.remove_request("old") == slot
        assert state.tree_next_budget_cpu == [None, None]
        state.add_request("new", 2, [1, 2], 0, 8)
        assert state.req_id_to_index["new"] == slot
        assert state.tree_next_budget_cpu[slot] is None
