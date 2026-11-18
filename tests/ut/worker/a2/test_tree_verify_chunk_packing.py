# SPDX-License-Identifier: Apache-2.0
"""Check the real recurrent kernel's tight-row contract for wide trees."""

import math
import os

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import compute_tree_verify_plan


@pytest.mark.parametrize("widths", [[17, 17], [17, 1], [1, 17], [33, 17, 1], [0, 17, 0], [17, 0, 1], [0, 0]])
def test_chunk_live_prefix_matches_kernel_accumulated_offsets(widths):
    rows, slots = len(widths), max(17, max(widths))
    state = torch.arange(rows * slots, dtype=torch.int32).reshape(rows, slots) + 100
    initial = state + 1000
    plan = compute_tree_verify_plan(torch.tensor(widths), state, initial, sum(widths))
    for chunk_index, chunk in enumerate(plan.chunks):
        expected, counts, start = [], [], 0
        for length in widths:
            stop = min((chunk_index + 1) * 16, length)
            first = min(chunk_index * 16, length)
            expected.extend(range(start + first, start + stop))
            counts.append(stop - first)
            start += length
        # The native op starts a row at sum(previous row counts), not row*16.
        live = sum(counts)
        assert chunk.cu_seqlens.tolist() == [0, *counts]
        assert chunk.token_indices[:live].tolist() == expected
        assert chunk.out_indices[:live].tolist() == expected
        assert chunk.state_indices[:live].tolist() == state.flatten()[expected].tolist()
        assert chunk.init_state_indices[:live].tolist() == initial.flatten()[expected].tolist()
        assert (chunk.state_indices[live:] == -1).all()
        assert (chunk.init_state_indices[live:] == -1).all()
        assert (chunk.out_indices[live:] >= sum(widths)).all()
        assert chunk.token_indices.numel() == rows * 16


@pytest.mark.skipif(os.environ.get("ECHO_TEST_NPU") != "1", reason="explicit native validation only")
@pytest.mark.parametrize("widths", [[17, 17], [17, 1]])
@pytest.mark.parametrize("graph", [False, True])
def test_chunked_native_gdn_matches_independent_parent_state_recurrence(widths, graph):
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.gdn import _chunked_tree_verify
    from vllm_ascend.utils import enable_custom_op

    assert torch.npu.device_count() == 1
    torch.npu.set_device(0)
    assert enable_custom_op()
    rows, slots, dim = len(widths), 17, 128
    own, parent, expected_values = [], [], []
    initial_state = torch.zeros((1 + rows * (slots + 1), 1, dim, dim))
    for row, length in enumerate(widths):
        prefix_page = 1 + row * (slots + 1) + slots
        initial_state[prefix_page].fill_((row + 1) / dim)
        values = []
        for node in range(length):
            own.append(1 + row * (slots + 1) + node)
            parent_node = (node - 1) // 2
            parent.append(prefix_page if node == 0 else 1 + row * (slots + 1) + parent_node)
            value = 0.5 * ((row + 1) if node == 0 else values[parent_node])
            values.append(value)
            expected_values.append(value)
    tokens = len(own)
    state_indices = torch.full((rows * slots,), -1, dtype=torch.int32)
    init_indices = state_indices.clone()
    state_indices[:tokens] = torch.tensor(own)
    init_indices[:tokens] = torch.tensor(parent)
    state_indices = state_indices.reshape(rows, slots).to("npu")
    init_indices = init_indices.reshape(rows, slots).to("npu")
    plan = compute_tree_verify_plan(torch.tensor(widths, device="npu"), state_indices, init_indices, tokens)
    state = initial_state.to("npu")
    query = torch.ones((1, tokens, 1, dim), dtype=torch.bfloat16, device="npu")
    key, value = torch.zeros_like(query), torch.zeros_like(query)
    g = torch.full((1, tokens, 1), math.log(0.5), dtype=torch.float32, device="npu")
    beta = torch.zeros_like(g, dtype=torch.bfloat16)
    accepted = torch.ones(rows, dtype=torch.int32, device="npu")

    def run():
        return _chunked_tree_verify(query, key, value, g, beta, state, state_indices, accepted, 1.0, plan, tokens)

    output = run()
    if graph:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            output = run()
    for factor in (1.0, 2.0):
        state.copy_(initial_state.to("npu"))
        query.fill_(factor)
        if graph:
            captured.replay()
        else:
            output = run()
        expected = torch.tensor(expected_values)[:, None, None].expand(tokens, 1, dim) * factor
        actual = output.cpu().float()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
        expected_state = initial_state.clone()
        for page, expected_value in zip(own, expected_values):
            expected_state[page].fill_(expected_value / dim)
        torch.testing.assert_close(state.cpu(), expected_state, rtol=2e-4, atol=1e-7)
