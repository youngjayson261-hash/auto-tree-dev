# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Pure non-root count contracts; no device/runtime imports."""

import importlib.util
import sys
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/tree/budget_policy.py"
spec = importlib.util.spec_from_file_location("_tree_budget_counts_actual", SOURCE)
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


class BudgetCountTests(unittest.TestCase):
    def test_counts_only_non_root_and_separates_denominators(self):
        self.assertEqual(
            policy.summarize_budget_counts([(6, 4, 3, 2), (6, 0, 0, 0), (4, 4, 4, 4)]),
            {"draft_non_root": 16, "published_non_root": 8, "verified_non_root": 7, "accepted_non_root": 6},
        )

    def test_empty_and_root_only_are_valid(self):
        expected = {"draft_non_root": 0, "published_non_root": 0, "verified_non_root": 0, "accepted_non_root": 0}
        self.assertEqual(policy.summarize_budget_counts([]), expected)
        self.assertEqual(policy.summarize_budget_counts([(0, 0, 0, 0)]), expected)
        self.assertEqual(policy.summarize_budget_counts([(6, 0, 0, 0)])["accepted_non_root"], 0)

    def test_negative_and_inverted_counts_are_rejected(self):
        for row in (
            (4, 3, 4, 1),
            (4, 3, 2, 3),
            (4, -1, 0, 0),
            (-1, 0, 0, 0),
            (3, 4, 0, 0),
            (4, 3, -1, 0),
            (4, 3, 2, -1),
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                policy.summarize_budget_counts([row])

    def test_boolean_float_and_string_are_not_host_counts(self):
        for index in range(4):
            for bad in (True, False, 1.0, "1", None):
                row = [4, 3, 2, 1]
                row[index] = bad
                with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                    policy.summarize_budget_counts([row])

    def test_large_host_counts_do_not_overflow(self):
        value = 2**40
        self.assertEqual(
            policy.summarize_budget_counts([(value, value, value, value)] * 3)["accepted_non_root"], 3 * value
        )

    def test_generator_and_inputs_are_not_mutated(self):
        rows = [[6, 4, 3, 2], [4, 4, 2, 0]]
        before = [row[:] for row in rows]
        result = policy.summarize_budget_counts(row for row in rows)
        self.assertEqual(rows, before)
        self.assertEqual(result["verified_non_root"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
