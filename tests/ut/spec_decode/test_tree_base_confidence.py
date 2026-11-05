# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
import importlib
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType

import torch

if os.environ.get("ECHO_TEST_NPU") == "1":
    import torch_npu  # noqa: F401

TREE_DIR = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/tree"
PACKAGE = "_echo_base_confidence_test"
package = ModuleType(PACKAGE)
package.__path__ = [str(TREE_DIR)]
sys.modules[PACKAGE] = package
policy = importlib.import_module(f"{PACKAGE}.budget_policy")
confidence = importlib.import_module(f"{PACKAGE}.base_confidence")


class TestBaseConfidence(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("ECHO_TEST_NPU") == "1", "explicit NPU validation only")
    def test_npu_matches_cpu_for_base_confidence(self):
        results = []
        logits_cpu = torch.tensor(
            [[[2.0, 1.0, -torch.inf], [0.5, 1.5, 0.0]], [[0.0, 1.0, 2.0], [torch.nan, 0.0, 1.0]]],
            dtype=torch.float16,
        )
        torch.npu.set_device(0)
        for device in ("cpu", "npu:0"):
            logits = logits_cpu.to(device)
            paths, summary = self.buffers(rows=2, device=device)
            ctx = confidence.prepare_base_confidence(logits, 3, paths, summary)
            first = ctx.candidates(
                torch.tensor([[[0, 2]], [[1, 2]]], device=device),
                torch.zeros(2, 1, dtype=torch.long, device=device),
                torch.ones(1, 2, dtype=torch.bool, device=device),
                0,
            )
            ctx.record_selected(first, torch.tensor([[1, 0], [1, 0]], device=device), 1)
            second = ctx.candidates(
                torch.tensor([[[0, 1], [1, 2]], [[0, 1], [1, 2]]], device=device),
                torch.tensor([[1, 2], [1, 2]], device=device),
                torch.ones(2, 2, dtype=torch.bool, device=device),
                1,
            )
            results.append([tensor.cpu() for tensor in (first, second, paths, *summary)])
        for cpu, npu in zip(*results):
            torch.testing.assert_close(cpu, npu, rtol=1e-4, atol=1e-5, equal_nan=True)

    def buffers(self, rows=1, depths=2, capacity=7, device="cpu"):
        return (
            torch.ones(rows, capacity, dtype=torch.float32, device=device),
            policy.TreeScoreSummary(
                torch.ones(rows, depths, dtype=torch.float32, device=device),
                torch.ones(rows, depths, dtype=torch.bool, device=device),
                torch.ones(rows, dtype=torch.bool, device=device),
            ),
        )

    def test_full_vocab_denominator_and_original_selected_parents(self):
        logits = torch.tensor([[[0.5, 0.3, 0.15, 0.05], [0.1, 0.2, 0.3, 0.4]]]).log()
        paths, summary = self.buffers()
        ctx = confidence.prepare_base_confidence(logits, 4, paths, summary)
        self.assertTrue(torch.isneginf(summary.max_path_logp).all())
        self.assertFalse(summary.valid_depths.any())
        self.assertFalse(summary.invalid_scores.any())
        self.assertEqual(torch.count_nonzero(paths), 0)
        flat = ctx.candidates(
            torch.tensor([[[1, 2], [0, 3]]]),
            torch.tensor([[0, 0]]),
            torch.tensor([[True, True], [False, False]]),
            0,
        )
        torch.testing.assert_close(flat[:, :2].exp(), torch.tensor([[0.3, 0.15]]))
        self.assertTrue(torch.isneginf(flat[:, 2:]).all())
        torch.testing.assert_close(summary.max_path_logp[0, 0].exp(), torch.tensor(0.3))
        ctx.record_selected(flat, torch.tensor([[1, 0]]), 1)
        next_flat = ctx.candidates(
            torch.tensor([[[3, 1], [0, 2]]]),
            torch.tensor([[1, 2]]),
            torch.ones(2, 2, dtype=torch.bool),
            1,
        )
        torch.testing.assert_close(next_flat.exp(), torch.tensor([[0.06, 0.03, 0.03, 0.09]]))
        ctx.record_selected(next_flat, torch.tensor([[3, 0]]), 3)
        torch.testing.assert_close(paths[:, 3:5].exp(), torch.tensor([[0.09, 0.06]]))
        self.assertTrue(summary.valid_depths.all())

    def test_summary_keeps_discarded_candidate(self):
        logits = torch.tensor([[[0.6, 0.3, 0.1]]]).log()
        paths, summary = self.buffers(depths=1)
        ctx = confidence.prepare_base_confidence(logits, 3, paths, summary)
        flat = ctx.candidates(torch.tensor([[[1, 2]]]), torch.tensor([[0]]), torch.ones(1, 2, dtype=torch.bool), 0)
        ctx.record_selected(flat, torch.tensor([[1]]), 1)
        torch.testing.assert_close(summary.max_path_logp.exp(), torch.tensor([[0.3]]))
        torch.testing.assert_close(paths[:, 1].exp(), torch.tensor([0.1]))

    def test_fp32_math_and_summary_reuse(self):
        logits = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float16)
        paths, summary = self.buffers(depths=1)
        for _ in range(2):
            ctx = confidence.prepare_base_confidence(logits, 3, paths, summary)
            self.assertFalse(summary.valid_depths.any())
            flat = ctx.candidates(torch.tensor([[[0, 2]]]), torch.tensor([[0]]), torch.ones(1, 2, dtype=torch.bool), 0)
            self.assertEqual(flat.dtype, torch.float32)
            expected = logits.float().log_softmax(-1)[:, 0, [0, 2]]
            torch.testing.assert_close(flat, expected)

    def test_invalid_normalizers_and_legal_negative_infinity(self):
        logits = torch.tensor([[[0.0, -torch.inf]], [[torch.nan, 0.0]], [[torch.inf, 0.0]], [[-torch.inf, -torch.inf]]])
        paths, summary = self.buffers(rows=4, depths=1)
        ctx = confidence.prepare_base_confidence(logits, 2, paths, summary)
        self.assertEqual(summary.invalid_scores.tolist(), [False, True, True, True])
        flat = ctx.candidates(
            torch.tensor([[[1]], [[0]], [[0]], [[0]]]),
            torch.zeros(4, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            0,
        )
        self.assertTrue(torch.isneginf(flat[0, 0]))
        self.assertFalse(summary.invalid_scores[0])

    def test_metadata_errors_are_fatal(self):
        logits = torch.zeros(1, 2, 3)
        paths, summary = self.buffers()
        for full_vocab in (True, 2, 3.0, None):
            with self.subTest(full_vocab=full_vocab), self.assertRaises(policy.TreeLayoutError):
                confidence.prepare_base_confidence(logits, full_vocab, paths, summary)
        with self.assertRaises(policy.TreeLayoutError):
            confidence.prepare_base_confidence(logits.long(), 3, paths, summary)
        with self.assertRaises(policy.TreeLayoutError):
            confidence.prepare_base_confidence(logits, 3, paths.double(), summary)
        with self.assertRaises(policy.TreeLayoutError):
            confidence.prepare_base_confidence(
                logits, 3, paths, summary._replace(valid_depths=torch.ones(1, 1, dtype=torch.bool))
            )
        ctx = confidence.prepare_base_confidence(logits, 3, paths, summary)
        with self.assertRaises(policy.TreeLayoutError):
            ctx.candidates(
                torch.zeros(1, 1, 1), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool), 0
            )
        with self.assertRaises(policy.TreeLayoutError):
            ctx.candidates(
                torch.zeros(1, 1, 1, dtype=torch.long),
                torch.zeros(1, 1, dtype=torch.long),
                torch.ones(1, 1, dtype=torch.bool),
                2,
            )

    def test_empty_rows_and_depths(self):
        paths, summary = self.buffers(rows=0, depths=2)
        ctx = confidence.prepare_base_confidence(torch.zeros(0, 2, 3), 3, paths, summary)
        flat = ctx.candidates(
            torch.zeros(0, 1, 2, dtype=torch.long),
            torch.zeros(0, 1, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
            0,
        )
        self.assertEqual(flat.shape, (0, 2))
        paths, summary = self.buffers(depths=0)
        confidence.prepare_base_confidence(torch.zeros(1, 0, 3), 3, paths, summary)
        self.assertFalse(summary.invalid_scores.any())


if __name__ == "__main__":
    unittest.main()
