# SPDX-License-Identifier: Apache-2.0
"""Batch-shrink regression for the tree metadata scatter (model_runner:699 crash).

The speculator tree buffers are statically shaped (max_num_reqs, budget).
When a request finishes before its batchmates (tau policies), the running
batch shrinks and the scatter must copy only the leading num_reqs rows.
"""

from types import SimpleNamespace

import torch

from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _runner(num_rows, budget=32, max_reqs=2):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    tree = SimpleNamespace(
        depths=torch.arange(max_reqs * budget, dtype=torch.int32).reshape(max_reqs, budget),
        parents=torch.full((max_reqs, budget), -7, dtype=torch.int32),
        visibility=torch.zeros((max_reqs, budget, budget), dtype=torch.bool),
        num_nodes=torch.arange(max_reqs, dtype=torch.int32) + 3,
        first_child=torch.full((max_reqs, budget + 1), -5, dtype=torch.int32),
        next_sibling=torch.full((max_reqs, budget + 1), -3, dtype=torch.int32),
    )
    tree.parents[1] = -70
    tree.visibility[1] = True
    idx = torch.arange(num_rows, dtype=torch.int64)
    runner.speculator = SimpleNamespace(tree=tree, input_batch=SimpleNamespace(idx_mapping=idx))
    runner.req_states = SimpleNamespace(
        tree_depths=torch.zeros((4, budget), dtype=torch.int32),
        tree_parents=torch.zeros((4, budget), dtype=torch.int32),
        tree_visibility=torch.zeros((4, budget, budget), dtype=torch.bool),
        tree_num_nodes=torch.zeros((4,), dtype=torch.int32),
        tree_first_child=torch.zeros((4, budget + 1), dtype=torch.int32),
        tree_next_sibling=torch.zeros((4, budget + 1), dtype=torch.int32),
    )
    return runner, tree


def test_full_batch_scatter_unchanged():
    runner, tree = _runner(num_rows=2)
    runner._commit_tree_metadata_to_req_states()
    assert torch.equal(runner.req_states.tree_depths[:2], tree.depths)
    assert torch.equal(runner.req_states.tree_parents[:2], tree.parents)
    assert torch.equal(runner.req_states.tree_visibility[:2], tree.visibility)
    assert torch.equal(runner.req_states.tree_num_nodes[:2], tree.num_nodes)
    assert torch.equal(runner.req_states.tree_first_child[:2], tree.first_child)
    assert torch.equal(runner.req_states.tree_next_sibling[:2], tree.next_sibling)


def test_shrunk_batch_scatter_slices_leading_rows():
    runner, tree = _runner(num_rows=1)
    # Pre-shrink crash signature: tree buffers keep [2, 32] while idx selects 1 row.
    runner._commit_tree_metadata_to_req_states()
    assert torch.equal(runner.req_states.tree_depths[0], tree.depths[0])
    assert torch.equal(runner.req_states.tree_parents[0], tree.parents[0])
    assert torch.equal(runner.req_states.tree_visibility[0], tree.visibility[0])
    assert runner.req_states.tree_num_nodes[0].item() == tree.num_nodes[0].item()
    assert torch.equal(runner.req_states.tree_first_child[0], tree.first_child[0])
    assert torch.equal(runner.req_states.tree_next_sibling[0], tree.next_sibling[0])
    # Stale rows of the wider buffer must not leak into request state.
    assert runner.req_states.tree_parents[1].sum().item() == 0
    assert not runner.req_states.tree_visibility[1].any()


def test_shrunk_batch_with_permuted_idx_mapping():
    runner, tree = _runner(num_rows=1)
    # The surviving request may occupy a different persistent-state row.
    runner.speculator.input_batch.idx_mapping = torch.tensor([2], dtype=torch.int64)
    runner._commit_tree_metadata_to_req_states()
    assert torch.equal(runner.req_states.tree_depths[2], tree.depths[0])
    assert runner.req_states.tree_num_nodes[2].item() == tree.num_nodes[0].item()
    assert runner.req_states.tree_parents[0].sum().item() == 0
