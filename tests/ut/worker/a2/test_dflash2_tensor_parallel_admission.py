# SPDX-License-Identifier: Apache-2.0
"""Loaded TP2 shard/group contracts; no tensor or collective is executed."""

from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dflash2_confidence_admission as base


class Unquantized:
    pass


def fixture(rank=0):
    model, scorer, builder = base.fixture()
    group = NS(world_size=2, rank_in_group=rank)
    head = model.lm_head
    head.tp_size, head.tp_rank = 2, rank
    head.parallel_group = None
    head.disable_tp = False
    head.num_added_embeddings = 0
    head.num_embeddings = head.org_vocab_size = 8
    head.org_vocab_size_padded = head.num_embeddings_padded = 8
    head.num_embeddings_per_partition = 4
    head.weight = NS(shape=(4, 3))
    head.quant_method = Unquantized()
    head.shard_indices = NS(
        padded_org_vocab_start_index=rank * 4,
        padded_org_vocab_end_index=(rank + 1) * 4,
        org_vocab_start_index=rank * 4,
        org_vocab_end_index=(rank + 1) * 4,
        num_org_vocab_padding=0,
    )
    model.logits_processor.use_all_gather = True
    model.logits_processor.logits_as_input = False
    model.logits_processor._get_logits = lambda *args: (_ for _ in ()).throw(
        AssertionError("host admission must not execute logits"))
    return model, scorer, builder, group


def prove(objects, source="scorer_logprob", **changes):
    model, scorer, builder, group = objects
    facts = dict(
        model_type=base.Model, scorer_type=base.Scorer, builder_type=base.Builder,
        target_vocab_size=8, confidence_source=source, tensor_parallel_size=2,
        tp_group=group, unquantized_head_type=Unquantized,
    )
    facts.update(changes)
    return base.admission.prove_dflash2_confidence(model, scorer, builder, **facts)


def runtime_facts(**changes):
    facts = dict(
        async_scheduling=False, tp=2, pp=1, dp=1, pcp=1, dcp=1,
        finegrained_tp_sizes=(0, 0, 0, 0), max_concurrent_batches=1,
        bonus_count=1, enforce_eager=True, tree_graph_present=False,
        adaptive_manager_present=False, greedy=True, grammar=False,
        processors_supported=True, scorer_admitted=True,
    )
    facts.update(changes)
    return facts


class TestDFlash2TP2Proof(unittest.TestCase):
    def test_both_actual_rank_shards_produce_group_bound_proof(self):
        for rank in (0, 1):
            objects = fixture(rank)
            proof = prove(objects)
            self.assertIsNotNone(proof)
            self.assertEqual((proof.tp_size, proof.tp_rank), (2, rank))
            self.assertIs(proof.tp_group, objects[3])
            self.assertEqual((proof.score_domain, proof.normalization_size), ("selector_lattice", 2))

    def test_base_tp2_requires_versioned_exact_model_interface(self):
        objects = fixture()
        self.assertIsNone(prove(objects, "base_full_vocab"))
        objects[0].base_logits_parallel_contract = "raw_get_logits_rankcat_v1"
        self.assertIsNone(prove(objects, "base_full_vocab"))
        with patch.object(base.Model, "base_logits_parallel_contract", "raw_get_logits_rankcat_v1", create=True):
            proof = prove(objects, "base_full_vocab")
            self.assertEqual((proof.score_domain, proof.normalization_size), ("base_full_vocab", 8))

    def test_group_head_rank_size_and_explicit_group_mismatches_reject(self):
        mutations = (
            lambda m, s, b, g: setattr(g, "world_size", 1),
            lambda m, s, b, g: setattr(g, "world_size", True),
            lambda m, s, b, g: setattr(g, "rank_in_group", 1),
            lambda m, s, b, g: setattr(g, "rank_in_group", True),
            lambda m, s, b, g: setattr(m.lm_head, "tp_size", 1),
            lambda m, s, b, g: setattr(m.lm_head, "tp_rank", 1),
            lambda m, s, b, g: setattr(m.lm_head, "parallel_group", g),
        )
        for mutation in mutations:
            objects = fixture()
            mutation(*objects)
            with self.subTest(mutation=mutation):
                self.assertIsNone(prove(objects))

    def test_native_partition_processor_and_quantization_contracts_reject(self):
        mutations = (
            lambda m, s, b, g: setattr(m.lm_head, "num_added_embeddings", 1),
            lambda m, s, b, g: setattr(m.lm_head, "org_vocab_size", 7),
            lambda m, s, b, g: setattr(m.lm_head, "num_embeddings_padded", 10),
            lambda m, s, b, g: setattr(m.lm_head, "num_embeddings_per_partition", True),
            lambda m, s, b, g: setattr(m.lm_head.weight, "shape", (3, 3)),
            lambda m, s, b, g: setattr(m.lm_head.shard_indices, "org_vocab_start_index", 1),
            lambda m, s, b, g: setattr(m.lm_head.shard_indices, "org_vocab_end_index", 5),
            lambda m, s, b, g: setattr(m.lm_head.shard_indices, "padded_org_vocab_end_index", 5),
            lambda m, s, b, g: setattr(m.lm_head.shard_indices, "num_org_vocab_padding", 1),
            lambda m, s, b, g: setattr(m.logits_processor, "use_all_gather", False),
            lambda m, s, b, g: setattr(m.logits_processor, "logits_as_input", True),
            lambda m, s, b, g: setattr(m.logits_processor, "_get_logits", None),
            lambda m, s, b, g: setattr(m.lm_head, "quant_method", object()),
            lambda m, s, b, g: setattr(m.lm_head, "disable_tp", True),
        )
        for mutation in mutations:
            objects = fixture()
            mutation(*objects)
            with self.subTest(mutation=mutation):
                self.assertIsNone(prove(objects))

    def test_every_shard_must_have_at_least_k_real_candidates(self):
        objects = fixture()
        model, scorer, _, _ = objects
        model.model.candidate_selector.top_k = scorer.top_k = 5
        self.assertIsNone(prove(objects))


class TestTP2RuntimeFacts(unittest.TestCase):
    def setUp(self):
        self.policy = base.load("_tp2_policy", base.ROOT / "vllm_ascend/worker/v2/spec_decode/tree/budget_policy.py")
        self.binding = patch.dict(sys.modules, {
            "vllm_ascend.worker.v2.spec_decode.tree.confidence_admission": base.admission,
        })
        self.binding.start()
        self.addCleanup(self.binding.stop)

    def test_actual_tp2_proof_is_required_and_other_parallel_dimensions_stay_single(self):
        proof = prove(fixture())
        self.assertIsNotNone(proof)
        self.policy.validate_runtime_facts(**runtime_facts(tensor_parallel_proof=proof))
        for name in ("pp", "dp", "pcp", "dcp"):
            with self.subTest(dimension=name):
                with self.assertRaisesRegex(ValueError, "parallelism"):
                    self.policy.validate_runtime_facts(**runtime_facts(tensor_parallel_proof=proof, **{name: 2}))

    def test_boolean_namespace_and_tp1_proof_cannot_enable_tp2(self):
        for value in (None, True, NS(tp_size=2), base.prove(*base.fixture())):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "parallelism"):
                    self.policy.validate_runtime_facts(**runtime_facts(tensor_parallel_proof=value))

    def test_proof_from_changed_group_is_rejected(self):
        objects = fixture()
        proof = prove(objects)
        self.assertIsNotNone(proof)
        objects[3].world_size = 1
        with self.assertRaisesRegex(ValueError, "parallelism"):
            self.policy.validate_runtime_facts(**runtime_facts(tensor_parallel_proof=proof))


class TestTP2RunnerBinding(unittest.TestCase):
    def setUp(self):
        self.fixture = base.TestDFlash2ModelRunnerAdmission()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.objects = fixture()
        model, scorer, builder, self.group = self.objects
        self.runner = self.fixture.runner
        self.runner.speculator.model, self.runner.speculator.tree_builder = model, builder
        self.runner.vllm_config.parallel_config.tensor_parallel_size = 2
        policy = base.load("_tp2_runner_policy", base.ROOT / "vllm_ascend/worker/v2/spec_decode/tree/budget_policy.py")
        self.binding = patch.dict(sys.modules, {
            "vllm_ascend.worker.v2.spec_decode.tree.confidence_admission": base.admission,
        })
        self.binding.start()
        self.addCleanup(self.binding.stop)
        sys.modules["vllm.distributed"].get_tp_group = lambda: self.group
        sys.modules["vllm.model_executor.layers.vocab_parallel_embedding"].UnquantizedEmbeddingMethod = Unquantized
        sys.modules["vllm_ascend.worker.v2.spec_decode.tree.budget_policy"].validate_runtime_facts = policy.validate_runtime_facts

    def test_runner_binds_only_live_group_proof_after_real_policy_accepts(self):
        self.fixture.legacy.validate(self.runner)
        proof = self.runner.speculator._tree_dflash2_confidence_proof
        self.assertEqual(proof.tp_size, 2)
        self.assertIs(proof.tp_group, self.group)
        self.assertTrue(self.runner.speculator._tree_dflash2_confidence_proven)

    def test_revalidation_failure_clears_old_tp2_proof(self):
        self.fixture.legacy.validate(self.runner)
        self.group.world_size = 1
        with self.assertRaisesRegex(ValueError, "parallelism|scorer"):
            self.fixture.legacy.validate(self.runner)
        self.assertIsNone(self.runner.speculator._tree_dflash2_confidence_proof)
        self.assertFalse(self.runner.speculator._tree_dflash2_confidence_proven)


if __name__ == "__main__":
    unittest.main(verbosity=2)
