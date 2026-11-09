# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise runtime admission with the native DFlash configuration layout."""

import ast
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class DominoModel:
    pass


class IndependentScorer:
    pass


class DominoScorer:
    pass


class LogitBiasState:
    pass


class PenaltiesState:
    pass


class BadWordsState:
    pass


def load_validation_method():
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_tree_confidence_runtime"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


class TestTreeConfidenceRuntimeCompatibility(unittest.TestCase):
    def setUp(self):
        self.validate = load_validation_method()
        self.facts = None
        self.model = DominoModel()
        # Native vLLM 0.31 DFlash stores this value on config, not the model.
        self.model.config = SimpleNamespace(vocab_size=8, draft_vocab_size=8)
        self.model.draft_id_to_target_id = None
        self.model.logits_processor = SimpleNamespace(vocab_size=8, org_vocab_size=8)
        self.runner = SimpleNamespace(
            _tree_budget_enabled=True,
            _tree_confidence_requested=True,
            speculator=SimpleNamespace(
                model=self.model,
                tree_builder=SimpleNamespace(scorer=IndependentScorer()),
                _tree_graphs={},
                enable_adaptive_verification=False,
            ),
            sampler=SimpleNamespace(
                logits_processors=[LogitBiasState(), PenaltiesState(), BadWordsState()],
                thinking_budget_state=SimpleNamespace(enabled=False),
                trace_replay_state=None,
            ),
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(async_scheduling=False),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                    prefill_context_parallel_size=1,
                    decode_context_parallel_size=1,
                ),
                max_concurrent_batches=1,
                model_config=SimpleNamespace(enforce_eager=True),
            ),
            model_state=SimpleNamespace(num_new_sampled_tokens_per_step=1),
            ascend_config=SimpleNamespace(
                finegrained_tp_config=SimpleNamespace(
                    oproj_tensor_parallel_size=0,
                    lmhead_tensor_parallel_size=0,
                    embedding_tensor_parallel_size=0,
                    mlp_tensor_parallel_size=0,
                )
            ),
            adaptive_verification=None,
        )
        bindings = {
            "vllm.sampling_params": {"SamplingType": SimpleNamespace(GREEDY=0)},
            "vllm.v1.worker.gpu.sample.sampler": {
                "LogitBiasState": LogitBiasState,
                "PenaltiesState": PenaltiesState,
                "BadWordsState": BadWordsState,
            },
            "vllm_ascend.models.qwen3_domino": {"AscendDominoQwen3ForCausalLM": DominoModel},
            "vllm_ascend.worker.v2.spec_decode.tree.scoring": {
                "IndependentScorer": IndependentScorer,
                "DominoScorer": DominoScorer,
            },
            "vllm_ascend.worker.v2.spec_decode.tree.budget_policy": {
                "admit_base_confidence_scorer": lambda scorer, independent, domino, full_vocab: (
                    full_vocab and scorer in (independent, domino)
                ),
                "validate_runtime_facts": self.record_facts,
            },
        }
        modules = {}
        for name, attributes in bindings.items():
            parts = name.split(".")
            for length in range(1, len(parts) + 1):
                module_name = ".".join(parts[:length])
                if module_name not in modules:
                    modules[module_name] = ModuleType(module_name)
                    modules[module_name].__path__ = []
            modules[name].__dict__.update(attributes)
        self.module_patch = patch.dict("sys.modules", modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def record_facts(self, **facts):
        self.facts = facts
        if not facts["scorer_admitted"]:
            raise ValueError("confidence scorer rejected")

    def test_native_config_full_vocab_is_admitted(self):
        self.validate(self.runner)
        self.assertTrue(self.facts["scorer_admitted"])
        self.assertEqual(self.runner.speculator._tree_full_vocab_size, 8)

    def test_reduced_draft_vocab_is_rejected(self):
        self.model.config.draft_vocab_size = 4
        with self.assertRaisesRegex(ValueError, "scorer rejected"):
            self.validate(self.runner)

    def test_mapped_draft_vocab_is_rejected(self):
        self.model.draft_id_to_target_id = object()
        with self.assertRaisesRegex(ValueError, "scorer rejected"):
            self.validate(self.runner)

    def test_logits_vocab_mismatch_is_rejected(self):
        self.model.logits_processor.vocab_size = 4
        with self.assertRaisesRegex(ValueError, "scorer rejected"):
            self.validate(self.runner)

    def test_original_vocab_mismatch_is_rejected(self):
        self.model.logits_processor.org_vocab_size = 4
        with self.assertRaisesRegex(ValueError, "scorer rejected"):
            self.validate(self.runner)

    def test_other_model_type_is_rejected_without_reading_vocab(self):
        self.runner.speculator.model = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "scorer rejected"):
            self.validate(self.runner)

    def test_disabled_confidence_skips_admission(self):
        self.runner._tree_budget_enabled = False
        self.runner._tree_confidence_requested = False
        self.runner.speculator.model = SimpleNamespace()
        self.validate(self.runner)
        self.assertIsNone(self.facts)

    def test_tau_zero_confidence_skips_admission(self):
        self.runner._tree_budget_enabled = False
        self.runner.speculator.model = SimpleNamespace()
        self.assertTrue(self.runner._tree_confidence_requested)
        self.validate(self.runner)
        self.assertIsNone(self.facts)


if __name__ == "__main__":
    unittest.main()
