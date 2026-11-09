# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Source contracts using production policy and AST-isolated runtime methods.

No model weights, accelerator or vLLM import is needed for these host checks.
The real TreeSpecConfig/vLLM integration is covered separately on the runtime.
"""

import ast
import importlib.util
import math
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[4]
POLICY_NAME = "echo_r003_runtime_policy"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load_file(POLICY_NAME, REPO / "vllm_ascend/worker/v2/spec_decode/tree/budget_policy.py")
compat = load_file("echo_r003_compat_fixtures", Path(__file__).with_name("test_tree_confidence_runtime_compat.py"))


def load_config_class():
    # Run the production validator body; the native vLLM decorator is not used.
    path = REPO / "vllm_ascend/ascend_config.py"
    cls = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "TreeSpecConfig"
    )
    cls.decorator_list = []
    cls.body = [
        node
        for node in cls.body
        if not isinstance(node, ast.FunctionDef)
        or node.name in {"_validate", "confidence_verification_enabled", "confidence_verification_requested"}
    ]
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_validate":
            node.decorator_list = []
    namespace = {"math": math}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return dataclass(namespace[cls.name])


class TestConfidenceSourceAdmission(unittest.TestCase):
    def test_exact_known_types_and_mapping_are_required(self):
        class Independent:
            pass

        class Domino(Independent):
            pass

        class Custom(Domino):
            pass

        for source in ("base_full_vocab", "scorer_logprob"):
            for scorer in (Independent, Domino):
                self.assertTrue(policy.admit_verification_confidence_scorer(scorer, Independent, Domino, True, source))
                self.assertFalse(
                    policy.admit_verification_confidence_scorer(scorer, Independent, Domino, False, source)
                )
            self.assertFalse(policy.admit_verification_confidence_scorer(Custom, Independent, Domino, True, source))
        self.assertFalse(
            policy.admit_verification_confidence_scorer(Independent, Independent, Domino, True, "priority")
        )

    def test_config_default_and_valid_sources(self):
        config = load_config_class()
        self.assertEqual(config().verification_confidence_source, "base_full_vocab")
        for source in ("base_full_vocab", "scorer_logprob"):
            value = config(verification_confidence_source=source)
            self.assertIs(value._validate(), value)

    def test_config_rejects_unknown_source_even_when_disabled(self):
        config = load_config_class()
        for value in ("priority", "", None, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "verification_confidence_source"):
                config(verification_confidence_source=value)._validate()

    def test_domino_metadata_records_effective_candidates_and_segmentation(self):
        metadata = policy.confidence_source_metadata(
            source="scorer_logprob",
            scorer_name="domino",
            candidate_size=64,
            prefix_len=2,
            width=8,
            vocab_size=100,
            threshold=0.1,
        )
        self.assertEqual(metadata["candidate_size_configured"], 64)
        self.assertEqual(metadata["candidate_size_effective"], 64)
        self.assertEqual(metadata["prefix_len"], 2)
        self.assertEqual(metadata["prefix_score_source"], "base_full_vocab")
        self.assertEqual(metadata["suffix_score_source"], "domino_top_c")
        self.assertEqual(metadata["threshold"], 0.1)
        self.assertEqual(metadata["frontier_width"], 8)
        self.assertIn("width", metadata["candidate_size_rule"])
        self.assertIn("topk", metadata["frontier_width_rule"])
        for configured, width, vocab, expected in (
            (None, 8, 100, 8),
            (None, 1, 100, 1),
            (2, 8, 100, 8),
            (64, 8, 12, 12),
        ):
            item = policy.confidence_source_metadata(
                source="scorer_logprob",
                scorer_name="domino",
                candidate_size=configured,
                prefix_len=0,
                width=width,
                vocab_size=vocab,
                threshold=0.1,
            )
            self.assertEqual(item["candidate_size_effective"], expected)

    def test_base_metadata_does_not_claim_candidate_normalized_confidence(self):
        metadata = policy.confidence_source_metadata(
            source="base_full_vocab",
            scorer_name="domino",
            candidate_size=64,
            prefix_len=2,
            width=8,
            vocab_size=100,
            threshold=0.1,
        )
        self.assertEqual(metadata["suffix_score_source"], "base_full_vocab")
        self.assertEqual(metadata["candidate_size_effective"], 64)


class TestConfidenceSourceRuntime(unittest.TestCase):
    def setUp(self):
        self.fixture = compat.TestTreeConfidenceRuntimeCompatibility()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.runner = self.fixture.runner
        self.runner.speculator._tree_confidence_source = "scorer_logprob"
        sys.modules[
            "vllm_ascend.worker.v2.spec_decode.tree.budget_policy"
        ].admit_verification_confidence_scorer = policy.admit_verification_confidence_scorer

    def test_domino_scorer_logprob_is_admitted(self):
        self.runner.speculator.tree_builder.scorer = compat.DominoScorer()
        self.fixture.validate(self.runner)
        self.assertTrue(self.fixture.facts["scorer_admitted"])

    def test_mapped_vocabulary_is_rejected(self):
        self.fixture.model.draft_id_to_target_id = object()
        with self.assertRaisesRegex(ValueError, "scorer rejected"):
            self.fixture.validate(self.runner)

    def test_subclass_is_rejected(self):
        class Custom(compat.DominoScorer):
            pass

        self.runner.speculator.tree_builder.scorer = Custom()
        with self.assertRaisesRegex(ValueError, "scorer rejected"):
            self.fixture.validate(self.runner)

    def test_unknown_source_is_rejected(self):
        self.runner.speculator._tree_confidence_source = "priority"
        with self.assertRaisesRegex(ValueError, "scorer rejected"):
            self.fixture.validate(self.runner)


class TestConfidenceSourceFinalizeWiring(unittest.TestCase):
    def setUp(self):
        path = REPO / "vllm_ascend/worker/v2/spec_decode/tree/speculator.py"
        method = next(
            node
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name == "_run_tree_finalize"
        )
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        self.finalize = namespace[method.name]
        tensor = MagicMock()
        tensor.shape = (1, 2, 8)
        tensor.view.return_value = tensor
        tensor.__getitem__.return_value = tensor
        self.builder = SimpleNamespace(build=MagicMock(return_value="built-tree"))
        self.speculator = SimpleNamespace(
            _draft_hidden_buf=tensor,
            sample_indices=tensor,
            num_speculative_steps=2,
            draft_backend="dflash",
            model=SimpleNamespace(compute_logits=lambda hidden: tensor),
            _tree_confidence_requested=True,
            _tree_budget_enabled=True,
            _tree_full_vocab_size=8,
            _load_layout_from_buffers=lambda rows: "layout",
            num_query_per_req=1,
            input_buffers=SimpleNamespace(input_ids=tensor),
            tree_builder=self.builder,
            _tree_budget_runtime=SimpleNamespace(score_summary=lambda rows: "summary"),
        )

    def test_scorer_source_reaches_builder(self):
        self.speculator._tree_confidence_source = "scorer_logprob"
        self.finalize(self.speculator, 1)
        self.assertEqual(self.builder.build.call_args.kwargs["verification_confidence_source"], "scorer_logprob")
        self.assertEqual(self.speculator.tree, "built-tree")

    def test_legacy_speculator_keeps_default_source(self):
        self.finalize(self.speculator, 1)
        self.assertEqual(self.builder.build.call_args.kwargs["verification_confidence_source"], "base_full_vocab")

    def test_fixed_budget_does_not_add_confidence_arguments(self):
        self.speculator._tree_budget_runtime = None
        self.speculator._tree_confidence_requested = False
        self.speculator._tree_budget_enabled = False
        self.speculator._tree_confidence_source = "scorer_logprob"
        self.finalize(self.speculator, 1)
        self.assertNotIn("verification_confidence_source", self.builder.build.call_args.kwargs)
        self.assertNotIn("score_summary", self.builder.build.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
