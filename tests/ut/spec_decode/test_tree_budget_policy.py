# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import math
import struct
import sys
import unittest
from ctypes import c_float
from pathlib import Path

# Import the standard-library reference without importing the NPU worker package.
source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/tree/budget_policy.py"
spec = importlib.util.spec_from_file_location("_tree_budget_policy_test", source)
assert spec is not None and spec.loader is not None
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


def example_row(**changes):
    row = dict(
        depths=[1, 1, 2, 2, 3, 3],
        parents=[0, 0, 1, 2, 3, 4],
        max_path_logp=[math.log(0.8), math.log(0.2), math.log(0.04)],
        valid_depths=[True, True, True],
        invalid_scores=False,
        bootstrap=False,
    )
    row.update(changes)
    return row


def runtime_facts(**changes):
    facts = dict(
        async_scheduling=False,
        tp=1,
        pp=1,
        max_concurrent_batches=1,
        bonus_count=1,
        enforce_eager=True,
        tree_graph_present=False,
        adaptive_manager_present=False,
        greedy=True,
        grammar=False,
        processors_supported=True,
        scorer_admitted=True,
    )
    facts.update(changes)
    return facts


class TestBudgetPolicy(unittest.TestCase):
    def test_batch_fallback_only_changes_confidence_rows(self):
        rows = [example_row(), example_row(invalid_scores=True), example_row(bootstrap=True)]
        self.assertEqual(policy.select_tree_prefix_batch_reference(rows, policy="confidence", tau=0.1), [
            (6, policy.PolicyStatus.SCORE_FALLBACK), (6, policy.PolicyStatus.SCORE_FALLBACK),
            (6, policy.PolicyStatus.BOOTSTRAP_FIXED),
        ])
        self.assertEqual(policy.select_tree_prefix_batch_reference(rows, policy="fixed", tau=0.1), [(6, policy.PolicyStatus.FIXED)] * 3)
        rows = [example_row(parents=[0, 0, 3, 2, 3, 4]), example_row(invalid_scores=True)]
        self.assertEqual(policy.select_tree_prefix_batch_reference(rows, policy="confidence", tau=0.1), [
            (0, policy.PolicyStatus.LAYOUT_ERROR), (6, policy.PolicyStatus.SCORE_FALLBACK),
        ])
        self.assertEqual(policy.select_tree_prefix_batch_reference([], policy="confidence", tau=0.1), [])

    def test_row_policy_fixed_bootstrap_and_errors(self):
        def choose(row, mode="confidence", tau=0.1):
            return policy.select_tree_prefix_reference(**row, policy=mode, tau=tau)

        self.assertEqual(choose(example_row()), (4, policy.PolicyStatus.CONFIDENCE))
        self.assertEqual(choose(example_row(), "fixed"), (6, policy.PolicyStatus.FIXED))
        self.assertEqual(choose(example_row(), tau=0), (6, policy.PolicyStatus.FIXED))
        self.assertEqual(choose(example_row(bootstrap=True, max_path_logp=None)), (6, policy.PolicyStatus.BOOTSTRAP_FIXED))
        for value in [math.nan, math.inf, 0.01]:
            with self.subTest(value=value):
                self.assertEqual(choose(example_row(max_path_logp=[value, -0.2, -0.3])), (6, policy.PolicyStatus.SCORE_FALLBACK))
        self.assertEqual(choose(example_row(max_path_logp=[-math.inf] * 3)), (0, policy.PolicyStatus.CONFIDENCE))
        self.assertEqual(choose(example_row(valid_depths=[True, False, True])), (0, policy.PolicyStatus.LAYOUT_ERROR))
        self.assertEqual(choose(example_row(parents=[0, 0, 3, 2, 3, 4])), (0, policy.PolicyStatus.LAYOUT_ERROR))
        self.assertEqual(choose(example_row(max_path_logp=None)), (6, policy.PolicyStatus.SCORE_FALLBACK))
        self.assertEqual(choose(example_row(invalid_scores=True)), (6, policy.PolicyStatus.SCORE_FALLBACK))
        self.assertEqual(choose(example_row(valid_depths=[False] * 3)), (6, policy.PolicyStatus.SCORE_FALLBACK))
        self.assertEqual(choose(example_row(valid_depths=[True, True, False])), (0, policy.PolicyStatus.LAYOUT_ERROR))
        self.assertEqual(choose(example_row(depths=[], parents=[], max_path_logp=[], valid_depths=[])), (0, policy.PolicyStatus.CONFIDENCE))
        self.assertEqual(choose(example_row(invalid_scores=True), "fixed"), (6, policy.PolicyStatus.FIXED))

    def test_real_node_count_and_prefix_topology(self):
        depths = [1, 1, 1, 2, 3]
        parents = [0, 0, 0, 1, 4]
        self.assertEqual(policy.count_selected_prefix(depths, parents, 2), 4)
        self.assertEqual(depths, [1, 1, 1, 2, 3])
        self.assertEqual(parents, [0, 0, 0, 1, 4])
        self.assertEqual(policy.count_selected_prefix([1, 1, 2], [0, 0, 1], 0), 0)
        self.assertEqual(policy.count_selected_prefix([], [], 0), 0)
        malformed = [([1, 2, 1], [0, 1, 0]), ([1, 2], [0, 2]),
                     ([1, 3], [0, 1]), ([1], []), ([True], [0]),
                     ([1], [-1]), ([1], [False])]
        for depths, parents in malformed:
            with self.subTest(depths=depths, parents=parents), self.assertRaises(policy.TreeLayoutError):
                policy.count_selected_prefix(depths, parents, 2)
        for kept_depth in (-1, True, 1.0):
            with self.subTest(kept_depth=kept_depth), self.assertRaises(policy.TreeLayoutError):
                policy.count_selected_prefix([1], [0], kept_depth)

    def test_continuous_depth_and_equality(self):
        scores = [math.log(0.8), math.log(0.2), math.log(0.04)]
        self.assertEqual(policy.consecutive_kept_depth(scores, [True] * 3, 0.1), 2)
        self.assertEqual(policy.consecutive_kept_depth([math.log(0.1)], [True], 0.1), 1)
        self.assertEqual(policy.consecutive_kept_depth([math.log(0.05), math.log(0.8)], [True] * 2, 0.1), 0)
        self.assertEqual(policy.consecutive_kept_depth([-math.inf], [True], 0.1), 0)
        self.assertEqual(policy.consecutive_kept_depth([0.0], [True], 1.0), 1)
        with self.assertRaises(policy.TreeLayoutError):
            policy.consecutive_kept_depth([-0.1, -0.2, -0.3], [True, False, True], 0.1)
        with self.assertRaises(policy.TreeLayoutError):
            policy.consecutive_kept_depth([-0.1], [], 0.1)
        boundary = c_float(math.log(0.8)).value
        below_bits = struct.unpack("!I", struct.pack("!f", boundary))[0] + 1
        below = struct.unpack("!f", struct.pack("!I", below_bits))[0]
        self.assertEqual(policy.consecutive_kept_depth([boundary], [True], 0.8), 1)
        self.assertEqual(policy.consecutive_kept_depth([below], [True], 0.8), 0)
        self.assertEqual(policy.consecutive_kept_depth([math.log(1e-50)], [True], 1e-50), 1)
        self.assertEqual(scores, [math.log(0.8), math.log(0.2), math.log(0.04)])

    def test_config_and_count_contracts(self):
        for mode in ("fixed", "confidence"):
            for threshold in (0, 0.1, 1):
                policy.validate_policy_config(mode, threshold)
        policy.validate_tree_counts(6, 4, 3, 8)
        policy.validate_tree_counts(6, 0, 0, 8)
        policy.validate_tree_counts(0, 0, 0, 0)
        for mode, tau in [
            ("other", 0.1),
            ("fixed", -0.1),
            ("fixed", 1.1),
            ("fixed", math.nan),
            ("fixed", math.inf),
            ("fixed", -math.inf),
            ("fixed", True),
            ("confidence", "0.1"),
        ]:
            with self.subTest(mode=mode, tau=tau), self.assertRaises(ValueError):
                policy.validate_policy_config(mode, tau)
        for counts in [
            (6, 4, 5, 8),
            (6, 7, 3, 8),
            (6, 4, 3, 5),
            (6, 0, -1, 8),
            (True, 0, 0, 8),
            (6, 4.0, 3, 8),
        ]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                policy.validate_tree_counts(*counts)
        self.assertEqual([int(value) for value in policy.PolicyStatus], list(range(5)))
        self.assertEqual(policy.TreeScoreSummary._fields, ("max_path_logp", "valid_depths", "invalid_scores"))
