# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_ascend.worker.v2.spec_decode.tree.budget_policy import PolicyStatus
from vllm_ascend.worker.v2.spec_decode.tree.speculator import AscendTreeSpeculator


def test_fixed_packet_uses_live_stored_counts_without_host_sync(monkeypatch):
    spec = AscendTreeSpeculator.__new__(AscendTreeSpeculator)
    spec.input_batch = SimpleNamespace(num_reqs=5)
    spec.tree = SimpleNamespace(num_nodes=torch.tensor([0, 1, 48, -1, 49]), depths=torch.empty(6, 48))
    spec.tree_budget_packet = torch.full((6, 2), -9, dtype=torch.int32)

    def forbidden(*args, **kwargs):
        raise AssertionError("host extraction in packet write")

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "item", forbidden)
        guard.setattr(torch.Tensor, "tolist", forbidden)
        spec._write_fixed_tree_budget()
    assert spec.tree_budget_packet.dtype == torch.int32
    assert spec.tree_budget_packet[:, 0].stride() == (2,)
    assert spec.tree_budget_packet.tolist() == [
        [0, PolicyStatus.FIXED],
        [1, PolicyStatus.FIXED],
        [48, PolicyStatus.FIXED],
        [-1, PolicyStatus.LAYOUT_ERROR],
        [49, PolicyStatus.LAYOUT_ERROR],
        [-9, -9],
    ]
