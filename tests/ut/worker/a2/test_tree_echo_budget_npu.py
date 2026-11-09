# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
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


@pytest.mark.skipif(os.environ.get("ECHO_TEST_NPU") != "1", reason="requires explicitly reserved NPU")
def test_native_multiround_reorder_cancel_and_recovery():
    pytest.importorskip("torch_npu")
    from vllm_ascend.patch.worker.patch_v2 import patch_uva  # noqa: F401

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    with torch_cuda_wrapper():
        state = AscendRequestState(2, 32, 32, 4, 64, device, TreeShape(4, 8, 2))
        state.add_request("a", 2, [1, 2], 0, 8)
        state.add_request("b", 2, [1, 2], 0, 8)
        spec = SimpleNamespace(budget=8, tree_budget_packet=torch.zeros((2, 2), dtype=torch.int32, device=device))
        handler = TreeDraftTokensHandler(device, state, spec, verification_policy="confidence", threshold=0.1)
    producer = torch.npu.Stream(device=device)
    packet_address = spec.tree_budget_packet.data_ptr()

    def seal(req_ids, counts, statuses=None):
        batch = SimpleNamespace(
            req_ids=list(req_ids),
            idx_mapping_np=np.array([state.req_id_to_index[req_id] for req_id in req_ids]),
            has_structured_output_reqs=False,
        )
        rows = list(zip(counts, statuses if statuses is not None else [0] * len(counts)))
        with torch.npu.stream(producer):
            spec.tree_budget_packet[: len(rows)].copy_(torch.tensor(rows, dtype=torch.int32))
            handler.set_draft_tokens(batch, None)
            handler.seal_budget_production()
        assert spec.tree_budget_packet.data_ptr() == packet_address

    seal(["a", "b"], [4, 2])
    first = handler.get_draft_tokens()
    assert first.req_ids == ["a", "b"]
    assert first.draft_token_ids == [[-1] * 4, [-1] * 2]
    seal(["a"], [0])
    assert handler.get_draft_tokens().draft_token_ids == [[]]
    assert state.tree_next_budget_cpu[state.req_id_to_index["b"]] == 2
    seal(["b", "a"], [3, 1])
    reordered = handler.get_draft_tokens()
    assert reordered.req_ids == ["b", "a"]
    assert reordered.draft_token_ids == [[-1] * 3, [-1]]

    seal(["a", "b"], [4, 0])
    old_slot = state.req_id_to_index["a"]
    assert state.remove_request("a") == old_slot
    with torch_cuda_wrapper():
        state.add_request("replacement", 2, [1, 2], 0, 8)
    assert state.req_id_to_index["replacement"] == old_slot
    cancelled = handler.get_draft_tokens()
    assert cancelled.req_ids == ["b"]
    assert cancelled.draft_token_ids == [[]]
    assert state.tree_next_budget_cpu[old_slot] is None

    seal(["replacement", "b"], [2, 1])
    assert handler.get_draft_tokens().draft_token_ids == [[-1] * 2, [-1]]
    previous = list(state.tree_next_budget_cpu)
    seal(["replacement", "b"], [4, 8], [0, 4])
    with pytest.raises(ValueError, match="budget|layout"):
        handler.get_draft_tokens()
    assert state.tree_next_budget_cpu == previous
    assert handler.get_draft_tokens() is None
    seal(["b", "replacement"], [0, 3])
    assert handler.get_draft_tokens().draft_token_ids == [[], [-1] * 3]
    assert state.tree_next_budget_cpu[state.req_id_to_index["b"]] == 0
    assert state.tree_next_budget_cpu[old_slot] == 3
    assert first.draft_token_ids == [[-1] * 4, [-1] * 2]


@pytest.mark.skipif(os.environ.get("ECHO_TEST_NPU") != "1", reason="requires explicitly reserved NPU")
@pytest.mark.parametrize("delayed_stage", ["producer", "copy"])
def test_native_budget_waits_for_delayed_stream(delayed_stage):
    pytest.importorskip("torch_npu")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    # Sleep is unavailable on some installed 910B operator packages. Queue
    # bounded, preallocated work instead; query below must prove it is pending.
    delay_matrix_size = 4096
    delay_iterations = 128
    delay_left = torch.full((delay_matrix_size, delay_matrix_size), 1 / 64, device=device)
    delay_right = torch.full_like(delay_left, 1 / 64)
    delay_output = torch.empty_like(delay_left)

    def enqueue_delay():
        for _ in range(delay_iterations):
            torch.mm(delay_left, delay_right, out=delay_output)

    state = SimpleNamespace(max_num_reqs=2, req_id_to_index={"a": 1, "b": 0}, tree_next_budget_cpu=[None, None])
    spec = SimpleNamespace(budget=8, tree_budget_packet=torch.zeros((2, 2), dtype=torch.int32, device=device))
    with torch_cuda_wrapper():
        handler = TreeDraftTokensHandler(device, state, spec, verification_policy="confidence", threshold=0.1)
    batch = SimpleNamespace(req_ids=["a", "b"], idx_mapping_np=np.array([1, 0]), has_structured_output_reqs=False)
    producer = torch.npu.Stream(device=device)
    before_write = torch.npu.Event()
    before_copy = torch.npu.Event()
    old_counts = [1, 2]
    rounds = [[4, 0], [0, 3]]
    sources = [
        torch.tensor([[count, 0] for count in counts], dtype=torch.int32, device=device)
        for counts in [old_counts, *rounds]
    ]
    packet_address = spec.tree_budget_packet.data_ptr()
    # Seed both buffers so synchronization-fault controls reach the delayed round.
    spec.tree_budget_packet.copy_(sources[0])
    handler._budget_cpu.copy_(torch.tensor([[1, 0], [2, 0]], dtype=torch.int32))
    torch.mm(delay_left, delay_right, out=delay_output)
    torch.npu.synchronize()
    try:
        # Warm up exactly the copy/event/get path before opening a delay window.
        with torch.npu.stream(producer):
            if delayed_stage == "producer":
                torch.mm(delay_left, delay_right, out=delay_output)
            before_write.record()
            spec.tree_budget_packet.copy_(sources[0], non_blocking=True)
            handler.set_draft_tokens(batch, None)
            handler.seal_budget_production()
        with torch.npu.stream(handler.copy_stream):
            if delayed_stage == "copy":
                torch.mm(delay_left, delay_right, out=delay_output)
            before_copy.record()
        warmup = handler.get_draft_tokens()
        assert [len(tokens) for tokens in warmup.draft_token_ids] == old_counts
        torch.npu.synchronize()

        for index, expected in enumerate(rounds):
            with torch.npu.stream(producer):
                if delayed_stage == "producer":
                    enqueue_delay()
                before_write.record()
                spec.tree_budget_packet.copy_(sources[index + 1], non_blocking=True)
                handler.set_draft_tokens(batch, None)
                handler.seal_budget_production()
            if delayed_stage == "copy":
                with torch.npu.stream(handler.copy_stream):
                    enqueue_delay()
                    before_copy.record()
            delayed_event = before_write if delayed_stage == "producer" else before_copy
            pending = not delayed_event.query()
            print(
                "ECHO_STREAM_PRE "
                + json.dumps(
                    {
                        "stage": delayed_stage,
                        "round": index,
                        "pending_before_write_or_copy": pending,
                        "delay_operator": "mm",
                        "matrix_size": delay_matrix_size,
                        "iterations": delay_iterations,
                    }
                ),
                flush=True,
            )
            assert pending, "delay window did not open; this run cannot prove stream waiting"
            delivered = handler.get_draft_tokens()
            actual = [len(tokens) for tokens in delivered.draft_token_ids]
            print(
                "ECHO_STREAM_RESULT "
                + json.dumps(
                    {
                        "stage": delayed_stage,
                        "round": index,
                        "previous": old_counts,
                        "expected": expected,
                        "actual": actual,
                    }
                ),
                flush=True,
            )
            assert actual == expected, "ECHO_BUDGET_AFTER_DELAY must contain this round's completed packet"
            assert delivered.req_ids == ["a", "b"]
            assert state.tree_next_budget_cpu == [expected[1], expected[0]]
            assert spec.tree_budget_packet.data_ptr() == packet_address
            assert handler.get_draft_tokens() is None
            old_counts = expected
        assert warmup.draft_token_ids == [[-1], [-1, -1]]
    finally:
        # Drain after capturing Host results, including in a failing private control.
        producer.synchronize()
        handler.copy_stream.synchronize()
