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
