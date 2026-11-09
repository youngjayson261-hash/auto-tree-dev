# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise complete production Builder/selector classes with CPU tensors."""

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.tree.budget_policy import (
    PolicyStatus,
    TreeScoreSummary,
    select_tree_prefix_device,
    select_tree_prefix_reference,
)
from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout
from vllm_ascend.worker.v2.spec_decode.tree.speculator import _TreeBudgetRuntime

BUDGET = 6
DEPTH = 3
VOCAB = 4
CPU = torch.device("cpu")


def layout(rows):
    return TreeLayout(
        tokens=torch.empty((rows, BUDGET), dtype=torch.int64, device=CPU),
        depths=torch.empty((rows, BUDGET), dtype=torch.int64, device=CPU),
        parents=torch.empty((rows, BUDGET), dtype=torch.int64, device=CPU),
        num_nodes=torch.empty(rows, dtype=torch.int32, device=CPU),
        visibility=torch.empty((rows, BUDGET, BUDGET), dtype=torch.bool, device=CPU),
        first_child=torch.empty((rows, BUDGET + 1), dtype=torch.int32, device=CPU),
        next_sibling=torch.empty((rows, BUDGET + 1), dtype=torch.int32, device=CPU),
    )


def summary(rows):
    return TreeScoreSummary(
        torch.empty((rows, DEPTH), dtype=torch.float32, device=CPU),
        torch.empty((rows, DEPTH), dtype=torch.bool, device=CPU),
        torch.empty(rows, dtype=torch.bool, device=CPU),
    )


def build(builder, rows, scores=None):
    logits = torch.tensor([[4.0, 0.0, -2.0, -4.0], [0.0] * VOCAB, [0.0] * VOCAB], device=CPU)
    logits = logits.unsqueeze(0).repeat(rows, 1, 1)
    kwargs = {} if scores is None else {"score_summary": scores, "full_vocab_size": VOCAB}
    return builder.build(logits, layout(rows), root_token_ids=torch.arange(rows, device=CPU), **kwargs)


@pytest.mark.parametrize("tau", [0.3, 0.5, 0.9, 1.0])
def test_full_builder_summary_and_device_selector_match_reference(tau):
    builder = TreeBuilder(BUDGET, 2)
    scores = summary(2)
    tree = build(builder, 2, scores)
    bootstrap = torch.tensor([False, True], device=CPU)
    packet = torch.empty((2, 2), dtype=torch.int32, device=CPU)
    pointer = packet.data_ptr()
    select_tree_prefix_device(tree, scores, packet, tau=tau, bootstrap_mask=bootstrap)
    expected = []
    for row in range(2):
        nodes = int(tree.num_nodes[row])
        decision = select_tree_prefix_reference(
            depths=tree.depths[row, :nodes].tolist(),
            parents=tree.parents[row, :nodes].tolist(),
            max_path_logp=scores.max_path_logp[row].tolist(),
            valid_depths=scores.valid_depths[row].tolist(),
            invalid_scores=bool(scores.invalid_scores[row]),
            policy="confidence",
            tau=tau,
            bootstrap=bool(bootstrap[row]),
        )
        expected.append([decision[0], int(decision[1])])
    assert packet.tolist() == expected
    assert packet.data_ptr() == pointer
    assert packet[1, 0] == tree.num_nodes[1]
    assert packet[1, 1] == PolicyStatus.BOOTSTRAP_FIXED
    if tau == 0.5:
        assert packet[0, 0] < tree.num_nodes[0]


def test_fixed_builder_does_not_allocate_base_confidence_workspace():
    builder = TreeBuilder(BUDGET, 2)
    build(builder, 2)
    assert builder._base_path_scores_buf is None


def test_confidence_workspace_must_be_reserved_before_freezing():
    builder = TreeBuilder(BUDGET, 2)
    builder.prepare_for_capture(2, VOCAB, DEPTH, CPU)
    with pytest.raises(ValueError, match="not reserved before capture"):
        build(builder, 2, summary(2))


def test_reserved_workspace_is_reused_when_batch_shrinks():
    builder = TreeBuilder(BUDGET, 2)
    builder.prepare_for_capture(2, VOCAB, DEPTH, CPU, reserve_base_confidence=True)
    pointers = tuple(value.data_ptr() for value in builder.workspace_tensors())
    build(builder, 2, summary(2))
    build(builder, 1, summary(1))
    assert tuple(value.data_ptr() for value in builder.workspace_tensors()) == pointers
    with pytest.raises(ValueError, match="captured workspace capacity"):
        build(builder, 3, summary(3))


def test_runtime_bootstrap_follows_current_batch_after_reorder_and_shrink():
    runtime = _TreeBudgetRuntime(2, DEPTH, CPU, tau=0.5)
    builder = TreeBuilder(BUDGET, 2)
    runtime.update_batch(np.array([True, False], dtype=bool), 2)
    scores = runtime.score_summary(2)
    tree = build(builder, 2, scores)
    packet = torch.empty((2, 2), dtype=torch.int32, device=CPU)
    runtime.publish(tree, packet)
    assert packet[0, 1] == PolicyStatus.BOOTSTRAP_FIXED
    assert packet[1, 0] < tree.num_nodes[1]
    runtime.update_batch(np.array([False, True], dtype=bool), 2)
    runtime.publish(tree, packet)
    assert packet[0, 1] == PolicyStatus.CONFIDENCE
    assert packet[0, 0] < tree.num_nodes[0]
    assert packet[1, 1] == PolicyStatus.BOOTSTRAP_FIXED
    runtime.update_batch(np.array([False], dtype=bool), 1)
    smaller_scores = runtime.score_summary(1)
    assert smaller_scores.max_path_logp.data_ptr() == scores.max_path_logp.data_ptr()
    tree = build(builder, 1, smaller_scores)
    runtime.publish(tree, packet[:1])
    assert packet[0, 1] == PolicyStatus.CONFIDENCE
    assert packet[0, 0] < tree.num_nodes[0]


def test_zero_threshold_runtime_keeps_full_tree_without_score_workspace():
    runtime = _TreeBudgetRuntime(2, DEPTH, CPU, tau=0.0)
    assert runtime.score_summary(2) is None
    tree = build(TreeBuilder(BUDGET, 2), 2)
    runtime.update_batch(np.array([False, True], dtype=bool), 2)
    packet = torch.empty((2, 2), dtype=torch.int32, device=CPU)
    runtime.publish(tree, packet)
    assert packet[:, 0].tolist() == tree.num_nodes.tolist()
    assert packet[:, 1].tolist() == [int(PolicyStatus.FIXED)] * 2
