# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select -k npu on physical NPU; -k fake validates the same case data on CPU."""

import json
import math
from ctypes import c_float

import numpy as np
import pytest
import torch
from test_tree_budget_selector_graph import (
    FakeBackend,
    TreeBudgetSelectorGraphRunner,
    graph_module,
    make_binding,
    policy,
    reference,
)

SCENARIOS = (
    "normal",
    "reject_deep",
    "empty_tree",
    "missing_depth",
    "depth_hole",
    "bad_parent",
    "negative_count",
    "deep_layout",
    "nan",
    "posinf",
    "neginf",
    "invalid_score",
    "bootstrap_invalid_score",
    "layout_and_score",
    "mixed_bootstrap",
    "row_swap",
    "threshold_equal",
    "threshold_below",
)


def fill_scenario(binding, name):
    rows, device = binding.num_reqs, binding.packet.device
    binding.layout.depths.copy_(torch.tensor([[1, 1, 2, 2]] * rows, dtype=torch.int32, device=device))
    binding.layout.parents.copy_(torch.tensor([[0, 0, 1, 2]] * rows, dtype=torch.int32, device=device))
    binding.layout.num_nodes.fill_(4)
    binding.summary.max_path_logp.fill_(-1)
    binding.summary.valid_depths.fill_(True)
    binding.summary.invalid_scores.fill_(False)
    binding.bootstrap_mask.fill_(False)
    if name == "reject_deep":
        binding.summary.max_path_logp[:, 1].fill_(-1000)
    elif name == "empty_tree":
        binding.layout.num_nodes.zero_()
        binding.summary.valid_depths.zero_()
    elif name == "missing_depth":
        binding.summary.valid_depths.zero_()
    elif name == "depth_hole":
        binding.summary.valid_depths[:, 0].fill_(False)
    elif name == "bad_parent":
        binding.layout.parents[:, 1].fill_(1)
    elif name == "negative_count":
        binding.layout.num_nodes.fill_(-1)
    elif name == "deep_layout":
        binding.layout.depths[:, 2:].fill_(4)
    elif name in ("nan", "posinf", "neginf"):
        binding.summary.max_path_logp[0, 1] = {"nan": float("nan"), "posinf": float("inf"), "neginf": float("-inf")}[
            name
        ]
    elif name == "invalid_score":
        binding.summary.invalid_scores[0] = True
    elif name == "bootstrap_invalid_score":
        binding.summary.invalid_scores[0] = True
        binding.bootstrap_mask[0] = True
    elif name == "layout_and_score":
        binding.layout.parents[0, 1] = 1
        binding.summary.max_path_logp[0, 1] = float("nan")
    elif name == "mixed_bootstrap":
        binding.bootstrap_mask[::2].fill_(True)
        binding.summary.max_path_logp[:, 1].fill_(-1000)
    elif name == "row_swap":
        binding.layout.num_nodes[0] = 0
        binding.summary.valid_depths[0].fill_(False)
        binding.bootstrap_mask[1::2].fill_(True)
        for tensor in binding.tensors()[:-1]:
            tensor.copy_(tensor.flip(0))
    elif name == "threshold_equal":
        binding.summary.max_path_logp.fill_(c_float(math.log(binding.tau)).value)
    elif name == "threshold_below":
        threshold = np.float32(c_float(math.log(binding.tau)).value)
        binding.summary.max_path_logp.fill_(float(np.nextafter(threshold, np.float32(-np.inf))))


@pytest.mark.parametrize("mode", ["fake", "npu"])
@pytest.mark.parametrize("rows", [1, 2, 4])
@pytest.mark.parametrize("tau", [0.1, 1e-300])
def test_dynamic_semantics(mode, rows, tau, monkeypatch):
    if mode == "npu":
        # An explicit NPU run must fail rather than silently skip unavailable hardware.
        import torch_npu  # registers the NPU device

        assert torch_npu is not None and torch.npu.is_available()
        device, backend = "npu:0", None
    else:
        device, backend = "cpu", FakeBackend()

        def record(*args, **kwargs):
            if backend.current is not None:
                backend.current.operation = lambda: policy.select_tree_prefix_device(*args, **kwargs)
            return policy.select_tree_prefix_device(*args, **kwargs)

        monkeypatch.setattr(graph_module, "select_tree_prefix_device", record)
    runner = TreeBudgetSelectorGraphRunner(backend=backend)
    with torch.inference_mode():
        binding = make_binding(rows, tau=tau, device=device)
        original = binding.packet.clone()
        assert runner.prepare(binding, pool=None)
        assert torch.equal(original, binding.packet)
        signature = binding.signature()
        for name in SCENARIOS:
            fill_scenario(binding, name)
            inputs = [t.clone() for t in binding.tensors()[:-1]]
            expected = reference(binding)
            assert runner.try_replay(binding)
            assert torch.equal(binding.packet, expected), (mode, rows, tau, name)
            assert binding.signature() == signature
            for actual, before in zip(binding.tensors()[:-1], inputs):
                torch.testing.assert_close(actual, before, rtol=0, atol=0, equal_nan=True)
            if name == "threshold_equal":
                assert binding.packet[:, 0].tolist() == [4] * rows
            elif name == "threshold_below":
                assert binding.packet[:, 0].tolist() == [0] * rows
        diag = runner.diagnostics()
        assert diag["prepared_rows"] == [rows]
        assert diag["replays"] == len(SCENARIOS) and not diag["eager_misses"]
        print(json.dumps(dict(mode=mode, rows=rows, tau=tau, scenarios=len(SCENARIOS), diagnostics=diag)))
