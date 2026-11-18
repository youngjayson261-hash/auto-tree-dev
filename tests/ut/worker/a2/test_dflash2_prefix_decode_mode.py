# SPDX-License-Identifier: Apache-2.0
"""Admit hybrid uniform decode graphs without admitting unsupported modes."""

from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dflash2_confidence_admission as base
import test_dflash2_prefix_eagle_config as normalized
import test_dflash2_prefix_runtime_admission as runtime
import test_tree_prefix_graph as legacy


class TestHybridDecodeOnlyAdmission(unittest.TestCase):
    def test_raw_and_eagle_tp1_tp2_decode_only_candidates(self):
        for tp in (1, 2):
            for wrapped in (False, True):
                for mode in ("FULL", "FULL_DECODE_ONLY"):
                    config = base.hybrid_candidate_config()
                    config.parallel_config.tensor_parallel_size = tp
                    if wrapped:
                        normalized.wrap(config)
                    config.compilation_config.cudagraph_mode.name = mode
                    with self.subTest(tp=tp, wrapped=wrapped, mode=mode):
                        self.assertTrue(base.prefix.prefix_graph_configured(config))

    def test_non_full_decode_modes_remain_rejected(self):
        for mode in ("NONE", "PIECEWISE", "FULL_AND_PIECEWISE", "UNKNOWN", None):
            config = base.hybrid_candidate_config()
            config.compilation_config.cudagraph_mode.name = mode
            with self.subTest(mode=mode):
                self.assertFalse(base.prefix.prefix_graph_configured(config))

    def test_domino_original_full_only_contract_is_preserved(self):
        config = legacy.config()
        self.assertTrue(base.prefix.prefix_graph_configured(config))
        config.compilation_config.cudagraph_mode.name = "FULL_DECODE_ONLY"
        self.assertFalse(base.prefix.prefix_graph_configured(config))

    def test_decode_only_cannot_bypass_hybrid_metadata(self):
        config = base.hybrid_candidate_config()
        config.compilation_config.cudagraph_mode.name = "FULL_DECODE_ONLY"
        config.speculative_config.draft_model_config.hf_config.hidden_size += 1
        self.assertFalse(base.prefix.prefix_graph_configured(config))

    def test_decode_only_keeps_eager_and_batch_restrictions(self):
        for field in ("target_eager", "draft_eager", "batch"):
            config = base.hybrid_candidate_config()
            config.compilation_config.cudagraph_mode.name = "FULL_DECODE_ONLY"
            if field == "target_eager":
                config.model_config.enforce_eager = True
            elif field == "draft_eager":
                config.speculative_config.enforce_eager = True
            else:
                config.scheduler_config.max_num_seqs = 2
            with self.subTest(field=field):
                self.assertFalse(base.prefix.prefix_graph_configured(config))

    def test_decode_only_still_requires_live_group_bound_runtime_proof(self):
        with patch.dict(sys.modules, {
            "vllm_ascend.worker.v2.spec_decode.tree.confidence_admission": base.admission,
        }):
            for tp in (1, 2):
                config, proof = runtime.runtime_case(tp)
                config.compilation_config.cudagraph_mode.name = "FULL_DECODE_ONLY"
                speculator = NS(_tree_dflash2_prefix_proof=proof)
                self.assertTrue(base.prefix.prefix_graph_runtime_proven(config, speculator, proof.tp_group))
                for invalid in (None, True, NS(tp_size=tp)):
                    speculator._tree_dflash2_prefix_proof = invalid
                    self.assertFalse(base.prefix.prefix_graph_runtime_proven(config, speculator, proof.tp_group))
                speculator._tree_dflash2_prefix_proof = proof
                other_group = NS(world_size=tp, rank_in_group=0)
                self.assertFalse(base.prefix.prefix_graph_runtime_proven(config, speculator, other_group))


if __name__ == "__main__":
    unittest.main()
