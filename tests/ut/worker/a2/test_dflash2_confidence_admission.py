# SPDX-License-Identifier: Apache-2.0
"""Loaded-object admission contracts; standard library only, no model run."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


admission = load("_dflash2_admission_test", ROOT / "vllm_ascend/worker/v2/spec_decode/tree/confidence_admission.py")
prefix = load("_dflash2_prefix_test", ROOT / "vllm_ascend/worker/v2/spec_decode/tree/prefix_graph.py")
compat = load("_dflash2_compat_test", Path(__file__).with_name("test_tree_confidence_runtime_compat.py"))


class Model:
    def compute_candidates(self, hidden):
        raise AssertionError("Admission must never execute tensor work")

    def compute_candidates_with_base_logits(self, hidden):
        raise AssertionError("Admission must never execute tensor work")


class Scorer:
    def confidence_metadata(self):
        return dict(normalization_domain="selector_lattice", token_id_domain="target_vocab",
                    normalization_size=self.top_k, target_vocab_size=self.predecessor_codebook.shape[0])


class Builder:
    pass


class Unquantized:
    pass


def fixture():
    model = Model()
    model.config = NS(architectures=["DFlash2DraftModel"], vocab_size=8)
    model.logits_processor = NS(vocab_size=8, org_vocab_size=8)
    model.draft_id_to_target_id = None
    model.lm_head = NS(tp_size=1, tp_rank=0, parallel_group=None,
                       shard_indices=NS(org_vocab_start_index=0, org_vocab_end_index=8))
    selector = NS(top_k=2, predecessor_codebook=NS(shape=(8, 3)), successor_codebook=NS(shape=(8, 3)),
                  hidden_projection=lambda _: None)
    model.model = NS(candidate_selector=selector)
    scorer = Scorer()
    for name in ("top_k", "predecessor_codebook", "successor_codebook", "hidden_projection"):
        setattr(scorer, name, getattr(selector, name))
    builder = Builder()
    builder.scorer = scorer
    return model, scorer, builder


def prove(model, scorer, builder, source="scorer_logprob", **changes):
    facts = dict(model_type=Model, scorer_type=Scorer, builder_type=Builder,
                 target_vocab_size=8, confidence_source=source, tensor_parallel_size=1,
                 tp_group=NS(world_size=1, rank_in_group=0))
    facts.update(changes)
    return admission.prove_dflash2_confidence(model, scorer, builder, **facts)


class TestDFlash2ConfidenceAdmission(unittest.TestCase):
    def test_tp2_and_unknown_parallel_size_remain_rejected(self):
        for size in (2, True, None, 0):
            with self.subTest(size=size):
                self.assertIsNone(prove(*fixture(), tensor_parallel_size=size))

    def test_topk_domain_and_full_base_domain_are_distinct(self):
        model, scorer, builder = fixture()
        topk = prove(model, scorer, builder)
        base = prove(model, scorer, builder, "base_full_vocab")
        self.assertEqual((topk.vocab_size, topk.score_domain, topk.normalization_size), (8, "selector_lattice", 2))
        self.assertEqual((base.vocab_size, base.score_domain, base.normalization_size), (8, "base_full_vocab", 8))

    def test_exact_loaded_types_and_builder_ownership(self):
        model, scorer, builder = fixture()
        for which, value in ((0, type("CustomModel", (Model,), {})()),
                             (1, type("CustomScorer", (Scorer,), {})()),
                             (2, type("CustomBuilder", (Builder,), {})())):
            objects = [model, scorer, builder]
            objects[which] = value
            self.assertIsNone(prove(*objects))
        builder.scorer = Scorer()
        self.assertIsNone(prove(model, scorer, builder))

    def test_vocab_mapping_and_whole_lm_head_are_required(self):
        mutations = (
            lambda m, s, b: setattr(m.config, "vocab_size", 7),
            lambda m, s, b: setattr(m.config, "draft_vocab_size", 7),
            lambda m, s, b: setattr(m, "draft_id_to_target_id", [0]),
            lambda m, s, b: setattr(m.logits_processor, "vocab_size", 7),
            lambda m, s, b: setattr(m.logits_processor, "org_vocab_size", 7),
            lambda m, s, b: setattr(m.lm_head.shard_indices, "org_vocab_start_index", 1),
            lambda m, s, b: setattr(m.lm_head.shard_indices, "org_vocab_end_index", 4),
            lambda m, s, b: setattr(m.config, "architectures", ["OtherDraftModel"]),
        )
        for change in mutations:
            with self.subTest(change=change):
                objects = fixture()
                change(*objects)
                self.assertIsNone(prove(*objects))
        self.assertIsNone(prove(*fixture(), target_vocab_size=True))

    def test_selector_shape_reference_and_metadata_are_required(self):
        changes = (
            lambda m, s, b: setattr(m.model.candidate_selector, "top_k", 0),
            lambda m, s, b: setattr(s, "top_k", 3),
            lambda m, s, b: setattr(s, "predecessor_codebook", NS(shape=(8, 3))),
            lambda m, s, b: setattr(m.model.candidate_selector.predecessor_codebook, "shape", (7, 3)),
            lambda m, s, b: setattr(m.model.candidate_selector.successor_codebook, "shape", (8, 2)),
            lambda m, s, b: setattr(s, "hidden_projection", lambda _: None),
            lambda m, s, b: setattr(s, "confidence_metadata", lambda: dict(
                normalization_domain="full_vocab", token_id_domain="target_vocab", normalization_size=8,
                target_vocab_size=8)),
            lambda m, s, b: setattr(s, "confidence_metadata", lambda: dict(
                domain="selector_lattice", token_id_domain="target_vocab", normalization_size=2,
                target_vocab_size=8)),
            lambda m, s, b: setattr(s, "confidence_metadata", lambda: dict(
                normalization_domain="selector_lattice", token_id_domain="target_vocab", normalization_size=2,
                target_vocab_size=7)),
        )
        for change in changes:
            with self.subTest(change=change):
                objects = fixture()
                change(*objects)
                self.assertIsNone(prove(*objects))

    def test_base_interface_must_be_on_exact_model_class(self):
        objects = fixture()
        with patch.object(Model, "compute_candidates_with_base_logits", None):
            objects[0].compute_candidates_with_base_logits = lambda _: None
            self.assertIsNone(prove(*objects, source="base_full_vocab"))
            self.assertIsNotNone(prove(*objects, source="scorer_logprob"))
        self.assertIsNone(prove(*fixture(), source="unknown"))


class TestDFlash2ModelRunnerAdmission(unittest.TestCase):
    def setUp(self):
        self.legacy = compat.TestTreeConfidenceRuntimeCompatibility()
        self.legacy.setUp()
        self.addCleanup(self.legacy.doCleanups)
        self.model, self.scorer, self.builder = fixture()
        self.runner = self.legacy.runner
        self.runner.vocab_size = 8
        self.runner.speculator.model = self.model
        self.runner.speculator.tree_builder = self.builder
        self.runner.speculator._tree_confidence_source = "scorer_logprob"
        modules = {}
        for name, attrs in {
            "vllm_ascend.models.qwen3_dflash2": {"DFlash2Qwen3ForCausalLM": Model},
            "vllm_ascend.worker.v2.spec_decode.tree.dflash2_builder": {"DFlash2BeamTreeBuilder": Builder},
            "vllm_ascend.worker.v2.spec_decode.tree.confidence_admission": {
                "prove_dflash2_confidence": admission.prove_dflash2_confidence,
                "valid_tp2_confidence_proof": admission.valid_tp2_confidence_proof},
            "vllm.distributed": {"get_tp_group": lambda: NS(world_size=1, rank_in_group=0)},
            "vllm.model_executor.layers.vocab_parallel_embedding": {"UnquantizedEmbeddingMethod": Unquantized},
        }.items():
            module = ModuleType(name)
            module.__dict__.update(attrs)
            modules[name] = module
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        scoring = sys.modules["vllm_ascend.worker.v2.spec_decode.tree.scoring"]
        scoring.DFlash2Scorer = Scorer

    def test_success_sets_only_proven_flag_and_target_vocab(self):
        self.legacy.validate(self.runner)
        self.assertTrue(self.runner.speculator._tree_dflash2_confidence_proven)
        self.assertEqual(self.runner.speculator._tree_full_vocab_size, 8)
        self.assertEqual(self.runner.speculator._tree_dflash2_confidence_proof.score_domain, "selector_lattice")

    def test_static_admission_is_not_revalidated_after_caching(self):
        # The static admission runs once per runner: after a successful first
        # validation the per-step calls take the cached fast path, so a (in
        # production impossible) later mutation of the model is not
        # re-inspected and the bound proof persists.
        self.legacy.validate(self.runner)
        proof = self.runner.speculator._tree_dflash2_confidence_proof
        self.model.draft_id_to_target_id = [0]
        self.legacy.validate(self.runner)
        self.assertTrue(self.runner.speculator._tree_dflash2_confidence_proven)
        self.assertIs(self.runner.speculator._tree_dflash2_confidence_proof, proof)

    def test_rejected_first_validation_caches_nothing(self):
        self.model.draft_id_to_target_id = [0]
        with self.assertRaisesRegex(ValueError, "scorer"):
            self.legacy.validate(self.runner)
        self.assertFalse(hasattr(self.runner, "_tree_confidence_static"))
        self.assertFalse(self.runner.speculator._tree_dflash2_confidence_proven)
        self.assertIsNone(self.runner.speculator._tree_dflash2_confidence_proof)


class TestHybridPrefixFullRemainsClosed(unittest.TestCase):
    def test_new_architecture_and_forged_capabilities_do_not_admit(self):
        old = load("_prefix_fixture_dflash2", Path(__file__).with_name("test_tree_prefix_graph.py"))
        # Reuse the exact known-good dense legacy configuration.
        factory = old.config
        for model_type, hybrid, architectures in (
            ("qwen3_5", True, ["DFlash2DraftModel"]),
            ("qwen3", True, []),
            ("qwen3", False, ["DFlash2DraftModel"]),
        ):
            cfg = factory()
            cfg.model_config.hf_config.model_type = model_type
            cfg.model_config.is_hybrid = hybrid
            cfg.speculative_config.draft_model_config.hf_config.architectures = architectures
            cfg.additional_config["hybrid_prefix_full_supported"] = True
            self.assertFalse(prefix.prefix_graph_configured(cfg))
        self.assertTrue(prefix.prefix_graph_configured(factory()))


def hybrid_candidate_config():
    old = load("_prefix_hybrid_candidate_fixture", Path(__file__).with_name("test_tree_prefix_graph.py"))
    cfg = old.config()
    text = NS(model_type="qwen3_5_text", num_hidden_layers=64, hidden_size=5120, vocab_size=248320,
              layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16)
    cfg.model_config.hf_config = NS(model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"],
                                   text_config=text)
    cfg.model_config.hf_text_config = text
    cfg.model_config.is_hybrid = True
    cfg.speculative_config.draft_model_config.hf_config = NS(
        model_type="qwen3", architectures=["DFlash2DraftModel"], is_causal=False,
        vocab_size=248320, hidden_size=5120, num_hidden_layers=5, num_target_layers=64,
        layer_types=["sliding_attention"] * 5,
        dflash_config=dict(block_size=8, selector_top_k=16, selector_rank=256,
                           target_layer_ids=[5, 19, 33, 47, 61]))
    return cfg


class TestHybridPrefixFullStaticCandidate(unittest.TestCase):
    def test_exact_candidate_is_configured_but_not_runtime_proven(self):
        cfg = hybrid_candidate_config()
        self.assertTrue(prefix.prefix_graph_configured(cfg))
        # No graph_descriptor, captured graph, or recurrent-state proof is
        # created by the static predicate. Runtime dispatch must check those.
        self.assertFalse(hasattr(cfg, "graph_descriptor"))

    def test_model_metadata_mismatch_rejects_candidate(self):
        mutations = (
            lambda c: setattr(c.model_config, "is_hybrid", False),
            lambda c: setattr(c.model_config.hf_config, "architectures", ["OtherForConditionalGeneration"]),
            lambda c: setattr(c.model_config.hf_text_config, "model_type", "qwen3"),
            lambda c: setattr(c.model_config.hf_text_config, "layer_types", ["full_attention"] * 64),
            lambda c: setattr(c.model_config.hf_text_config, "layer_types", [[]] * 64),
            lambda c: setattr(c.speculative_config.draft_model_config.hf_config, "num_target_layers", 63),
            lambda c: setattr(c.speculative_config.draft_model_config.hf_config, "vocab_size", 248321),
            lambda c: setattr(c.speculative_config.draft_model_config.hf_config, "hidden_size", 1024),
            lambda c: setattr(c.speculative_config.draft_model_config.hf_config, "is_causal", True),
            lambda c: c.speculative_config.draft_model_config.hf_config.dflash_config.update(target_layer_ids=[64] * 5),
            lambda c: c.speculative_config.draft_model_config.hf_config.dflash_config.update(block_size=4),
            lambda c: c.speculative_config.draft_model_config.hf_config.dflash_config.update(selector_top_k=True),
            lambda c: c.speculative_config.draft_model_config.hf_config.dflash_config.update(selector_rank=0),
        )
        for mutation in mutations:
            cfg = hybrid_candidate_config()
            mutation(cfg)
            with self.subTest(mutation=mutation):
                self.assertFalse(prefix.prefix_graph_configured(cfg))

    def test_exact_tp2_candidate_is_configured_without_capability_switch(self):
        cfg = hybrid_candidate_config()
        cfg.parallel_config.tensor_parallel_size = 2
        self.assertTrue(prefix.prefix_graph_configured(cfg))


if __name__ == "__main__":
    unittest.main(verbosity=2)
