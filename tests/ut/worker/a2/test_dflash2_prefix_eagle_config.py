# SPDX-License-Identifier: Apache-2.0
"""DFlash2 FULL admission after vLLM's EAGLEConfig normalization."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dflash2_confidence_admission as base
import test_dflash2_prefix_runtime_admission as runtime
import test_tree_prefix_graph as legacy


def wrap(config):
    original = config.speculative_config.draft_model_config.hf_config
    # EAGLEConfig(method="dflash", model_type="eagle") retains the original
    # config in .model and copies its draft metadata to the wrapper.
    normalized = NS(**copy.deepcopy(vars(original)))
    normalized.model = original
    normalized.model_type = "eagle"
    config.speculative_config.draft_model_config.hf_config = normalized
    return config


class TestNormalizedDFlash2Prefix(unittest.TestCase):
    def test_raw_and_normalized_tp1_tp2_candidates(self):
        for size in (1, 2):
            for normalized in (False, True):
                config = base.hybrid_candidate_config()
                config.parallel_config.tensor_parallel_size = size
                if normalized:
                    wrap(config)
                with self.subTest(tp=size, normalized=normalized):
                    self.assertTrue(base.prefix.qwen35_dflash2_prefix_configured(config))
                    self.assertTrue(base.prefix.prefix_graph_configured(config))

    def test_wrapper_must_retain_the_qwen3_dflash2_model(self):
        mutations = (
            lambda hf: setattr(hf, "model", None),
            lambda hf: setattr(hf.model, "model_type", "llama"),
            lambda hf: setattr(hf.model, "architectures", ["OtherDraftModel"]),
            lambda hf: setattr(hf, "architectures", ["OtherDraftModel"]),
            lambda hf: setattr(hf, "model_type", "other"),
        )
        for mutate in mutations:
            config = wrap(base.hybrid_candidate_config())
            mutate(config.speculative_config.draft_model_config.hf_config)
            with self.subTest(mutation=mutate):
                self.assertFalse(base.prefix.prefix_graph_configured(config))

    def test_outer_runtime_metadata_still_requires_full_validation(self):
        mutations = (
            lambda hf: setattr(hf, "is_causal", True),
            lambda hf: setattr(hf, "hidden_size", 1024),
            lambda hf: setattr(hf, "num_target_layers", 63),
            lambda hf: setattr(hf, "layer_types", ["full_attention"] * 5),
        )
        for mutate in mutations:
            config = wrap(base.hybrid_candidate_config())
            mutate(config.speculative_config.draft_model_config.hf_config)
            with self.subTest(mutation=mutate):
                self.assertFalse(base.prefix.prefix_graph_configured(config))

    def test_parallel_and_batch_limits_are_unchanged(self):
        for size in (0, 3, 4, True, None):
            config = wrap(base.hybrid_candidate_config())
            config.parallel_config.tensor_parallel_size = size
            with self.subTest(tp=size):
                self.assertFalse(base.prefix.prefix_graph_configured(config))
        config = wrap(base.hybrid_candidate_config())
        config.scheduler_config.max_num_seqs = 2
        self.assertFalse(base.prefix.prefix_graph_configured(config))

    def test_normalized_config_does_not_replace_loaded_runtime_proof(self):
        config, proof = runtime.runtime_case(2)
        wrap(config)
        with patch.dict(sys.modules, {
            "vllm_ascend.worker.v2.spec_decode.tree.confidence_admission": base.admission,
        }):
            self.assertFalse(base.prefix.prefix_graph_runtime_proven(config, NS(), proof.tp_group))
            self.assertTrue(base.prefix.prefix_graph_runtime_proven(
                config, NS(_tree_dflash2_prefix_proof=proof), proof.tp_group))

    def test_actual_legacy_capture_receives_the_callback_after_normalization(self):
        def normalized_config():
            config = wrap(base.hybrid_candidate_config())
            config.parallel_config.tensor_parallel_size = 2
            return config
        with patch.object(legacy, "config", normalized_config):
            case = legacy.LegacyDraftIntegration()
            case.test_v031_capture_initializes_padding_before_real_context_callback()


if __name__ == "__main__":
    unittest.main()
