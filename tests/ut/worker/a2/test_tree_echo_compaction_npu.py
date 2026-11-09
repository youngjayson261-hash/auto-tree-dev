# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest


@pytest.mark.skipif(os.environ.get("ECHO_TEST_NPU") != "1", reason="requires explicitly reserved NPU")
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_native_accepted_path_compaction_preserves_request_boundaries(dtype_name):
    pytest.importorskip("torch_npu")
    import torch

    from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
    from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_tree_query_along_path

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    dtype = getattr(torch, dtype_name)
    hidden = torch.arange(44, dtype=dtype).reshape(11, 4).to(device)
    auxiliary = (64 + torch.arange(33, dtype=dtype)).reshape(11, 3).to(device)
    query_start_loc_cpu = torch.tensor([0, 4, 5, 8, 11], dtype=torch.int32)
    query_start_loc = query_start_loc_cpu.to(device)
    path_cpu = torch.tensor(
        [
            [3, 1, -1],  # Gather overlapping sources before writing destinations.
            [-1, -1, -1],  # Active root-only query has exactly one scheduled row.
            [2, 3, -1],  # Node 3 is the next request's root, outside this query.
            [2, -1, -1],  # Inactive request must preserve every row and position.
        ],
        dtype=torch.int32,
    )
    num_sampled_cpu = torch.tensor([3, 1, 3, 0], dtype=torch.int32)
    path = path_cpu.to(device)
    num_sampled = num_sampled_cpu.to(device)
    acceptance = TreeAcceptance.from_path(path, num_sampled)
    acceptance_expected = {
        "path_node_ids": path_cpu,
        "num_sampled": num_sampled_cpu,
        "num_accepted": torch.tensor([2, 0, 2, 0], dtype=torch.int32),
        "endpoint_node": torch.tensor([1, 0, 3, 0], dtype=torch.int32),
        "active": torch.tensor([True, True, True, False]),
    }
    for field, expected in acceptance_expected.items():
        assert torch.equal(getattr(acceptance, field).cpu(), expected), field

    positions = torch.tensor([100, 101, 101, 102, 200, 300, 301, 301, 400, 401, 401], device=device)
    # Fixed CPU expectations are independent of the production gather/scatter.
    # The synthetic out-of-interval node tests a boundary guard, not actual
    # scheduler stop/context truncation or end-to-end acceptance.
    expected_hidden = torch.tensor(
        [
            [0, 1, 2, 3],
            [12, 13, 14, 15],
            [4, 5, 6, 7],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
            [28, 29, 30, 31],
            [28, 29, 30, 31],
            [32, 33, 34, 35],
            [36, 37, 38, 39],
            [40, 41, 42, 43],
        ],
        dtype=dtype,
    )
    expected_auxiliary = torch.tensor(
        [
            [64, 65, 66],
            [73, 74, 75],
            [67, 68, 69],
            [73, 74, 75],
            [76, 77, 78],
            [79, 80, 81],
            [85, 86, 87],
            [85, 86, 87],
            [88, 89, 90],
            [91, 92, 93],
            [94, 95, 96],
        ],
        dtype=dtype,
    )
    expected_positions = torch.tensor([100, 101, 102, 102, 200, 300, 301, 301, 400, 401, 401])

    compact_tree_query_along_path([hidden, auxiliary], query_start_loc, acceptance, positions)

    assert torch.equal(hidden.cpu(), expected_hidden)
    assert torch.equal(auxiliary.cpu(), expected_auxiliary)
    assert torch.equal(positions.cpu(), expected_positions)
    assert torch.equal(query_start_loc.cpu(), query_start_loc_cpu)
    assert torch.equal(path.cpu(), path_cpu)
    assert torch.equal(num_sampled.cpu(), num_sampled_cpu)
    for field, expected in acceptance_expected.items():
        assert torch.equal(getattr(acceptance, field).cpu(), expected), field
