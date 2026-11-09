# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Numerical contracts for confidence from normalized scorer edges."""

import importlib
import sys
import unittest
from pathlib import Path
from types import ModuleType

import torch

TREE_DIR = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/tree"
PACKAGE = "_echo_scorer_confidence_test"
package = ModuleType(PACKAGE)
package.__path__ = [str(TREE_DIR)]
sys.modules[PACKAGE] = package
policy = importlib.import_module(f"{PACKAGE}.budget_policy")
confidence = importlib.import_module(f"{PACKAGE}.base_confidence")


class TestScorerConfidence(unittest.TestCase):
    def buffers(self, rows=1, depths=2, capacity=7):
        return (
            torch.ones(rows, capacity, dtype=torch.float32),
            policy.TreeScoreSummary(
                torch.ones(rows, depths, dtype=torch.float32),
                torch.ones(rows, depths, dtype=torch.bool),
                torch.ones(rows, dtype=torch.bool),
            ),
        )

    def test_neutral_parent_and_original_base_constructor(self):
        paths, summary = self.buffers(depths=1)
        with self.assertRaises(TypeError):
            confidence.ConfidenceContext()
        scorer = confidence.prepare_scorer_confidence(paths, summary)
        self.assertIsInstance(scorer, confidence.ConfidenceContext)
        self.assertEqual(set(vars(scorer)), {"path_scores", "summary"})
        logits = torch.tensor([[[2.0, 1.0, -1.0]]])
        normalizers = logits.logsumexp(-1)
        base = confidence.BaseConfidenceContext(logits, normalizers, paths, summary)
        self.assertIsInstance(base, confidence.ConfidenceContext)
        self.assertIs(base.logits, logits)
        flat = base.candidates(torch.tensor([[[0, 2]]]), torch.tensor([[0]]), torch.ones(1, 2, dtype=torch.bool), 0)
        torch.testing.assert_close(flat, logits.log_softmax(-1)[:, 0, [0, 2]])

    def test_parent_specific_edges_and_final_selected_paths(self):
        paths, summary = self.buffers()
        ctx = confidence.prepare_scorer_confidence(paths, summary)
        first_scores = torch.tensor([[[0.6, 0.3]]]).log()
        first = ctx.candidates(
            torch.tensor([[[1, 2]]]),
            torch.tensor([[0]]),
            torch.ones(1, 2, dtype=torch.bool),
            0,
            edge_scores=first_scores,
        )
        # Forced selection may reverse score order; the next frontier follows it.
        ctx.record_selected(first, torch.tensor([[1, 0]]), 1)
        second_scores = torch.tensor([[[0.8, 0.1], [0.2, 0.4]]]).log()
        second = ctx.candidates(
            torch.tensor([[[3, 4], [3, 4]]]),
            torch.tensor([[1, 2]]),
            torch.ones(2, 2, dtype=torch.bool),
            1,
            edge_scores=second_scores,
        )
        # Token 3 has different normalized scores for the two parent states.
        torch.testing.assert_close(second.exp(), torch.tensor([[0.24, 0.03, 0.12, 0.24]]))
        ctx.record_selected(second, torch.tensor([[2, 0]]), 3)
        torch.testing.assert_close(paths[:, 3:5].exp(), torch.tensor([[0.12, 0.24]]))
        torch.testing.assert_close(summary.max_path_logp.exp(), torch.tensor([[0.6, 0.24]]))
        self.assertTrue(summary.valid_depths.all())
        self.assertFalse(summary.invalid_scores.any())

    def test_topk_is_not_renormalized_and_summary_precedes_selection(self):
        paths, summary = self.buffers(depths=1)
        ctx = confidence.prepare_scorer_confidence(paths, summary)
        # These are the top two values of a larger normalized candidate set.
        scores = torch.tensor([[[0.4, 0.2]]]).log()
        flat = ctx.candidates(
            torch.tensor([[[1, 2]]]),
            torch.tensor([[0]]),
            torch.ones(1, 2, dtype=torch.bool),
            0,
            edge_scores=scores,
        )
        torch.testing.assert_close(flat.exp(), torch.tensor([[0.4, 0.2]]))
        ctx.record_selected(flat, torch.tensor([[1]]), 1)
        torch.testing.assert_close(summary.max_path_logp.exp(), torch.tensor([[0.4]]))
        torch.testing.assert_close(paths[:, 1].exp(), torch.tensor([0.2]))

    def test_invalid_mask_excludes_bad_edges_and_negative_infinity_is_legal(self):
        paths, summary = self.buffers(rows=2, depths=1)
        ctx = confidence.prepare_scorer_confidence(paths, summary)
        scores = torch.tensor([[[-torch.inf, torch.nan], [-2.0, torch.inf]], [[-1.0, 2.0], [-3.0, torch.nan]]])
        flat = ctx.candidates(
            torch.zeros(2, 2, 2, dtype=torch.long),
            torch.zeros(2, 2, dtype=torch.long),
            torch.tensor([[True, False], [True, False]]),
            0,
            edge_scores=scores,
        )
        torch.testing.assert_close(
            flat, torch.tensor([[-torch.inf, -torch.inf, -2.0, -torch.inf], [-1.0, -torch.inf, -3.0, -torch.inf]])
        )
        self.assertFalse(summary.invalid_scores.any())

    def test_invalid_edges_are_detected_even_below_a_negative_parent(self):
        paths, summary = self.buffers(rows=4, depths=1)
        ctx = confidence.prepare_scorer_confidence(paths, summary)
        paths[:, 1].fill_(-10)
        ctx.candidates(
            torch.zeros(4, 1, 1, dtype=torch.long),
            torch.ones(4, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            0,
            edge_scores=torch.tensor([[[0.1]], [[torch.nan]], [[torch.inf]], [[-torch.inf]]]),
        )
        self.assertEqual(summary.invalid_scores.tolist(), [True, True, True, False])

    def test_fp32_math_and_active_buffer_reuse(self):
        paths, summary = self.buffers(rows=3, depths=1)
        original_ptr = paths.data_ptr()
        for indices in (torch.tensor([2, 0, 1]), torch.tensor([1, 2])):
            rows = len(indices)
            active_paths = paths[:rows]
            active_summary = policy.TreeScoreSummary(*(value[:rows] for value in summary))
            ctx = confidence.prepare_scorer_confidence(active_paths, active_summary)
            self.assertEqual(ctx.path_scores.data_ptr(), original_ptr)
            self.assertFalse(active_summary.valid_depths.any())
            self.assertFalse(active_summary.invalid_scores.any())
            self.assertTrue(torch.isneginf(active_summary.max_path_logp).all())
            self.assertEqual(torch.count_nonzero(active_paths), 0)
            scores = torch.tensor([-0.25, -1.0, -2.0], dtype=torch.float16)[indices].reshape(rows, 1, 1)
            flat = ctx.candidates(
                torch.zeros(rows, 1, 1, dtype=torch.long),
                torch.zeros(rows, 1, dtype=torch.long),
                torch.ones(1, 1, dtype=torch.bool),
                0,
                edge_scores=scores,
            )
            self.assertEqual(flat.dtype, torch.float32)
            torch.testing.assert_close(flat, scores.float().flatten(1))
            ctx.record_selected(flat, torch.zeros(rows, 1, dtype=torch.long), 1)

    def test_edge_metadata_errors_are_fatal(self):
        paths, summary = self.buffers(depths=1)
        ctx = confidence.prepare_scorer_confidence(paths, summary)
        tokens = torch.zeros(1, 1, 2, dtype=torch.long)
        frontier = torch.zeros(1, 1, dtype=torch.long)
        valid = torch.ones(1, 2, dtype=torch.bool)
        for scores in (
            None,
            torch.zeros(1, 2),
            torch.zeros(1, 1, 2, dtype=torch.long),
            torch.zeros(1, 1, 2, device="meta"),
        ):
            with self.subTest(scores=scores), self.assertRaises(policy.TreeLayoutError):
                ctx.candidates(tokens, frontier, valid, 0, edge_scores=scores)
        for args in (
            (tokens.float(), frontier, valid, 0),
            (tokens, frontier.float(), valid, 0),
            (tokens, frontier, valid.float(), 0),
            (tokens, frontier, valid, True),
            (tokens, frontier, valid, 1),
        ):
            with self.subTest(args=args), self.assertRaises(policy.TreeLayoutError):
                ctx.candidates(*args, edge_scores=torch.zeros(1, 1, 2))

    def test_summary_and_path_metadata_errors_are_fatal(self):
        paths, summary = self.buffers()
        for invalid_paths in (paths.double(), paths[:, :0], torch.zeros(2, 7), paths.to("meta")):
            with self.subTest(paths=invalid_paths), self.assertRaises(policy.TreeLayoutError):
                confidence.prepare_scorer_confidence(invalid_paths, summary)
        for invalid_summary in (
            summary._replace(max_path_logp=summary.max_path_logp.double()),
            summary._replace(valid_depths=torch.ones(1, 1, dtype=torch.bool)),
            summary._replace(invalid_scores=torch.ones(2, dtype=torch.bool)),
            summary._replace(valid_depths=summary.valid_depths.to("meta")),
        ):
            with self.subTest(summary=invalid_summary), self.assertRaises(policy.TreeLayoutError):
                confidence.prepare_scorer_confidence(paths, invalid_summary)

    def test_empty_rows_and_depths(self):
        paths, summary = self.buffers(rows=0)
        ctx = confidence.prepare_scorer_confidence(paths, summary)
        flat = ctx.candidates(
            torch.zeros(0, 1, 2, dtype=torch.long),
            torch.zeros(0, 1, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
            0,
            edge_scores=torch.zeros(0, 1, 2),
        )
        self.assertEqual(flat.shape, (0, 2))
        paths, summary = self.buffers(depths=0)
        confidence.prepare_scorer_confidence(paths, summary)
        self.assertEqual(summary.max_path_logp.shape, (1, 0))


if __name__ == "__main__":
    unittest.main()
