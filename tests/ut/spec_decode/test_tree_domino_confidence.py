# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
import dataclasses
import importlib
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[3]
TREE_PACKAGE = "vllm_ascend.worker.v2.spec_decode.tree"


def load_tree_modules():
    # Skip plugin initialization and the unused distributed embedding boundary.
    # Builder, layout and Domino candidate/GRU update logic execute unmodified.
    modules = {}
    for suffix in ("", ".worker", ".worker.v2", ".worker.v2.spec_decode", ".worker.v2.spec_decode.tree", ".ops"):
        name = "vllm_ascend" + suffix
        module = ModuleType(name)
        module.__path__ = [str(REPO / name.replace(".", "/"))]
        modules[name] = module
    embedding = ModuleType("vllm_ascend.ops.vocab_parallel_embedding")
    embedding.AscendVocabParallelEmbedding = torch.nn.Embedding
    embedding.VocabParallelMode = ModuleType("unused_vocab_parallel_mode")
    modules[embedding.__name__] = embedding
    with patch.dict(sys.modules, modules):
        return tuple(
            importlib.import_module(f"{TREE_PACKAGE}.{name}")
            for name in ("builder", "layout", "scoring", "budget_policy")
        )


builder_module, layout_module, scoring, policy = load_tree_modules()


class SmallCorrection:
    """Deterministic model boundary; real DominoScorer performs the scoring."""

    def __init__(self):
        self.gru_hidden_dim = 1
        self.gru_w_hh = torch.zeros(1)
        self.w_s = torch.ones(1, 1)
        self.middle = torch.nn.Identity()
        self.fc2_weight = torch.tensor([[-8.0], [5.0], [-1.7], [4.0], [2.5], [-0.5]])
        self.fc2_bias = torch.zeros(6)

    def project_z(self, hidden):
        return hidden[..., :1]

    def update_hidden(self, tokens, states):
        return states * 0.5 + (tokens.float()[:, None] + 1) * 0.1


class RecordingBuilder(builder_module.TreeBuilder):
    """Compute an independent scalar oracle over the received candidate pool."""

    def set_reference(self, logits):
        self.reference_logits = logits.double().tolist()
        self.reference_paths = [{0: 0.0} for _ in self.reference_logits]
        self.reference_maxima = []

    def _expand_layer(self, state, layer, child_depth, origin_candidate, **kwargs):
        ids = layer.token_ids.tolist()
        parents = state.frontier.tolist()
        live = 1 if child_depth == 1 else state.width
        candidate_count = layer.token_ids.shape[-1]
        candidate_rows = []
        for row, logits in enumerate(self.reference_logits):
            values = logits[child_depth - 1]
            peak = max(values)
            log_z = peak + math.log(sum(math.exp(value - peak) for value in values))
            paths = []
            for parent in range(state.width):
                tokens = ids[row][0 if len(ids[row]) == 1 else parent]
                for token in tokens:
                    paths.append(
                        self.reference_paths[row][parents[row][parent]] + values[token] - log_z
                        if parent < live
                        else -math.inf
                    )
            self.assert_candidate_count = candidate_count
            candidate_rows.append(paths)
        self.reference_maxima.append([max(paths) for paths in candidate_rows])
        frontier, selected = super()._expand_layer(state, layer, child_depth, origin_candidate, **kwargs)
        start = state.num_nodes + 1
        for row, indices in enumerate(selected.tolist()):
            for offset, index in enumerate(indices):
                self.reference_paths[row][start + offset] = candidate_rows[row][index]
        return frontier, selected


class TestDominoConfidence(unittest.TestCase):
    def inputs(self):
        logits = torch.tensor(
            [
                [[3.0, 2.5, 2.0, 1.5, 0.0, -1.0], [2.0, 0.1, 1.1, 2.3, -1.0, -2.0], [0.2, 1.0, 2.0, 0.7, 0.0, -1.0]],
                [[0.5, 1.5, 2.0, 1.2, -1.0, 0.0], [1.0, 2.0, 0.1, 2.5, -2.0, 0.0], [1.7, 0.2, 2.2, 1.3, 0.3, -0.5]],
            ]
        )
        return logits, torch.tensor([5, 2]), torch.full((2, 3, 1), 0.8)

    def output(self, rows, budget):
        return layout_module.TreeLayout(
            tokens=torch.empty(rows, budget, dtype=torch.long),
            depths=torch.empty(rows, budget, dtype=torch.long),
            parents=torch.empty(rows, budget, dtype=torch.long),
            num_nodes=torch.empty(rows, dtype=torch.int32),
            visibility=torch.empty(rows, budget, budget, dtype=torch.bool),
            first_child=torch.empty(rows, budget + 1, dtype=torch.int32),
            next_sibling=torch.empty(rows, budget + 1, dtype=torch.int32),
        )

    def summary(self, rows=2, depths=3):
        return policy.TreeScoreSummary(
            torch.empty(rows, depths, dtype=torch.float32),
            torch.empty(rows, depths, dtype=torch.bool),
            torch.empty(rows, dtype=torch.bool),
        )

    def build_pair(self, prefix, origin, penalty, budget):
        logits, roots, hidden = self.inputs()
        builders, layouts = [], []
        summary = self.summary()
        for enabled in (False, True):
            scorer = (
                scoring.IndependentScorer()
                if prefix is None
                else scoring.DominoScorer(SmallCorrection(), prefix_len=prefix, candidate_size=4)
            )
            builder = RecordingBuilder(budget, 2, scorer, keep_origin_chain=origin, depth_penalty=penalty)
            builder.set_reference(logits)
            output = self.output(2, budget)
            kwargs = {"score_summary": summary, "full_vocab_size": 6} if enabled else {}
            builder.build(logits, output, root_token_ids=roots, draft_hidden=hidden, **kwargs)
            builders.append(builder)
            layouts.append(output)
        for field in dataclasses.fields(layout_module.TreeLayout):
            self.assertTrue(torch.equal(getattr(layouts[0], field.name), getattr(layouts[1], field.name)), field.name)
        torch.testing.assert_close(builders[0]._path_scores_buf, builders[1]._path_scores_buf, rtol=0, atol=0)
        if prefix is not None:
            torch.testing.assert_close(builders[0].scorer._hidden, builders[1].scorer._hidden, rtol=0, atol=0)
        expected = torch.tensor(builders[1].reference_maxima).T
        torch.testing.assert_close(summary.max_path_logp, expected)
        self.assertTrue(summary.valid_depths.all())
        self.assertFalse(summary.invalid_scores.any())
        for row, paths in enumerate(builders[1].reference_paths):
            for node, value in paths.items():
                torch.testing.assert_close(builders[1]._base_path_scores_buf[row, node], torch.tensor(value))
        self.assertNotEqual(builders[1]._base_path_scores_buf.data_ptr(), builders[1]._path_scores_buf.data_ptr())
        return builders[1], layouts[1], summary

    def test_same_tree_scores_and_hidden_with_separate_confidence(self):
        for prefix in (None, 0, 1, 3):
            for origin, penalty, budget in ((False, 0.0, 3), (False, 0.4, 5), (True, 0.4, 3), (True, 0.0, 6)):
                with self.subTest(prefix=prefix, origin=origin, penalty=penalty, budget=budget):
                    self.build_pair(prefix, origin, penalty, budget)

    def test_confidence_summary_closes_reference_budget_policy(self):
        _, layout, summary = self.build_pair(0, False, 0.0, 5)
        rows = [
            dict(
                depths=layout.depths[row].tolist(),
                parents=layout.parents[row].tolist(),
                max_path_logp=summary.max_path_logp[row].tolist(),
                valid_depths=summary.valid_depths[row].tolist(),
                invalid_scores=bool(summary.invalid_scores[row]),
                bootstrap=False,
            )
            for row in range(2)
        ]
        for tau in (0.0, 0.01, 0.2, 1.0):
            decisions = policy.select_tree_prefix_batch_reference(rows, policy="confidence", tau=tau)
            for row, (count, status) in zip(rows, decisions):
                self.assertIn(status, (policy.PolicyStatus.FIXED, policy.PolicyStatus.CONFIDENCE))
                self.assertTrue(0 <= count <= len(row["depths"]))
                if tau == 0:
                    self.assertEqual(count, 5)
                else:
                    log_tau = float(torch.tensor(math.log(tau), dtype=torch.float32))
                    depth = 0
                    for maximum in row["max_path_logp"]:
                        if maximum < log_tau:
                            break
                        depth += 1
                    self.assertEqual(count, sum(value <= depth for value in row["depths"]))
        rows[0]["invalid_scores"] = True
        self.assertEqual(
            policy.select_tree_prefix_batch_reference(rows, policy="confidence", tau=0.2),
            [(5, policy.PolicyStatus.SCORE_FALLBACK)] * 2,
        )

    def test_frozen_workspace_reuse_and_admission_errors(self):
        logits, roots, hidden = self.inputs()
        scorer = scoring.DominoScorer(SmallCorrection(), candidate_size=4)
        builder = builder_module.TreeBuilder(5, 2, scorer)
        builder.prepare_for_capture(2, 6, 3, torch.device("cpu"), reserve_base_confidence=True)
        pointers = [tensor.data_ptr() for tensor in builder.workspace_tensors()]
        for _ in range(2):
            builder.build(
                logits,
                self.output(2, 5),
                root_token_ids=roots,
                draft_hidden=hidden,
                score_summary=self.summary(),
                full_vocab_size=6,
            )
            self.assertEqual(pointers, [tensor.data_ptr() for tensor in builder.workspace_tensors()])
        self.assertIn(builder._base_path_scores_buf.data_ptr(), pointers)
        for kwargs in (
            {"full_vocab_size": 5},
            {"draft_candidate_ids": torch.arange(6).expand_as(logits).long()},
            {"score_summary": self.summary(depths=2)},
        ):
            options = dict(score_summary=self.summary(), full_vocab_size=6)
            options.update(kwargs)
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(policy.TreeLayoutError):
                builder.build(logits, self.output(2, 5), root_token_ids=roots, draft_hidden=hidden, **options)

        class Child(scoring.DominoScorer):
            pass

        unsupported = builder_module.TreeBuilder(5, 2, Child(SmallCorrection()))
        with self.assertRaises(policy.TreeLayoutError):
            unsupported.build(
                logits,
                self.output(2, 5),
                root_token_ids=roots,
                draft_hidden=hidden,
                score_summary=self.summary(),
                full_vocab_size=6,
            )


if __name__ == "__main__":
    unittest.main()
