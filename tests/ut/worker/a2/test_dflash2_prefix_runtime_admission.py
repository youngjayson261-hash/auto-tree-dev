# SPDX-License-Identifier: Apache-2.0
"""Hybrid prefix candidate and live loaded-object proof contracts."""

import ast
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dflash2_confidence_admission as base
import test_dflash2_tensor_parallel_admission as tp2


def runtime_case(tp_size):
    cfg = base.hybrid_candidate_config()
    cfg.parallel_config.tensor_parallel_size = tp_size
    cfg.model_config.hf_text_config.vocab_size = 8
    cfg.speculative_config.draft_model_config.hf_config.vocab_size = 8
    cfg.speculative_config.draft_model_config.hf_config.dflash_config["selector_top_k"] = 2
    if tp_size == 2:
        objects = tp2.fixture()
        proof = tp2.prove(objects, purpose="prefix_full")
    else:
        objects = base.fixture()
        proof = base.prove(*objects, purpose="prefix_full")
    return cfg, proof


def method(name, *, in_class=False):
    source = base.ROOT / "vllm_ascend/worker/v2/model_runner.py"
    node = next(
        node for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"SchedulerOutput": object, "AscendTreeSpeculator": type("OtherSpeculator", (), {})}
    if in_class:
        class Parent:
            def gather_batch_req_state(self, scheduler_output, dummy_run):
                return self.mock_state, 1
        namespace["Parent"] = Parent
        node = ast.ClassDef(name="NPUModelRunner", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                            keywords=[], body=[node], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["NPUModelRunner"] if in_class else namespace[name]


class TestHybridPrefixRuntimeProof(unittest.TestCase):
    def setUp(self):
        self.binding = patch.dict(sys.modules, {
            "vllm_ascend.worker.v2.spec_decode.tree.confidence_admission": base.admission,
        })
        self.binding.start()
        self.addCleanup(self.binding.stop)

    def test_exact_tp1_and_tp2_candidates_need_group_bound_prefix_proof(self):
        for size in (1, 2):
            cfg, proof = runtime_case(size)
            self.assertTrue(base.prefix.prefix_graph_configured(cfg))
            speculator = NS(_tree_dflash2_prefix_proof=proof)
            self.assertTrue(base.prefix.prefix_graph_runtime_proven(cfg, speculator, proof.tp_group))
            for bad in (None, True, NS(tp_size=size)):
                speculator._tree_dflash2_prefix_proof = bad
                self.assertFalse(base.prefix.prefix_graph_runtime_proven(cfg, speculator, proof.tp_group))

    def test_equal_metadata_in_different_group_cannot_reuse_proof(self):
        cfg, proof = runtime_case(2)
        speculator = NS(_tree_dflash2_prefix_proof=proof)
        other = NS(world_size=2, rank_in_group=0)
        self.assertFalse(base.prefix.prefix_graph_runtime_proven(cfg, speculator, other))
        proof.tp_group.world_size = 1
        self.assertFalse(base.prefix.prefix_graph_runtime_proven(cfg, speculator, proof.tp_group))

    def test_prefix_and_confidence_proofs_cannot_be_borrowed_in_either_direction(self):
        cfg, prefix_proof = runtime_case(2)
        self.assertFalse(base.admission.valid_tp2_confidence_proof(prefix_proof))
        objects = tp2.fixture()
        confidence_proof = tp2.prove(objects)
        self.assertFalse(base.prefix.prefix_graph_runtime_proven(
            cfg, NS(_tree_dflash2_prefix_proof=confidence_proof), objects[3]))

    def test_legacy_domino_is_still_tp1_and_needs_no_dflash2_proof(self):
        old = base.load("_prefix_legacy_runtime", Path(__file__).with_name("test_tree_prefix_graph.py"))
        cfg = old.config()
        self.assertTrue(base.prefix.prefix_graph_runtime_proven(cfg, NS()))
        cfg.parallel_config.tensor_parallel_size = 2
        self.assertFalse(base.prefix.prefix_graph_configured(cfg))
        self.assertFalse(base.prefix.prefix_graph_runtime_proven(cfg, NS()))


class TestPrefixRunnerLoadedProof(unittest.TestCase):
    def setUp(self):
        self.fixture = base.TestDFlash2ModelRunnerAdmission()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.runner = self.fixture.runner
        model, scorer, builder, self.group = tp2.fixture()
        self.runner.speculator.model, self.runner.speculator.tree_builder = model, builder
        self.runner.vllm_config = base.hybrid_candidate_config()
        self.runner.vllm_config.model_config.hf_text_config.vocab_size = 8
        self.runner.vllm_config.speculative_config.draft_model_config.hf_config.vocab_size = 8
        self.runner.vllm_config.speculative_config.draft_model_config.hf_config.dflash_config["selector_top_k"] = 2
        self.runner.vllm_config.parallel_config.tensor_parallel_size = 2
        self.runner._tree_budget_enabled = False
        self.binding = patch.dict(sys.modules, {
            "vllm_ascend.worker.v2.spec_decode.tree.prefix_graph": base.prefix,
            "vllm_ascend.worker.v2.spec_decode.tree.confidence_admission": base.admission,
        })
        self.binding.start()
        self.addCleanup(self.binding.stop)
        sys.modules["vllm.distributed"].get_tp_group = lambda: self.group
        sys.modules["vllm.model_executor.layers.vocab_parallel_embedding"].UnquantizedEmbeddingMethod = tp2.Unquantized

    def test_fixed_gets_host_prefix_proof_without_confidence_admission(self):
        init = method("_init_tree_prefix_runtime_proof")
        init(self.runner)
        proof = self.runner.speculator._tree_dflash2_prefix_proof
        self.assertEqual(proof.tp_size, 2)
        self.assertEqual(proof.purpose, "prefix_full")
        self.assertIs(proof.tp_group, self.group)
        self.assertFalse(self.runner._tree_budget_enabled)
        self.assertFalse(getattr(self.runner.speculator, "_tree_dflash2_confidence_proven", False))

    def test_confidence_revalidation_creates_separate_prefix_role(self):
        self.runner._tree_budget_enabled = True
        self.runner._init_tree_prefix_runtime_proof = MethodType(method("_init_tree_prefix_runtime_proof"), self.runner)
        self.fixture.legacy.validate(self.runner)
        confidence = self.runner.speculator._tree_dflash2_confidence_proof
        prefix = self.runner.speculator._tree_dflash2_prefix_proof
        self.assertIsNot(confidence, prefix)
        self.assertEqual((confidence.purpose, prefix.purpose), ("confidence", "prefix_full"))

    def test_requested_hybrid_full_cannot_keep_stale_or_silently_accept_bad_proof(self):
        init = method("_init_tree_prefix_runtime_proof")
        init(self.runner)
        self.group.world_size = 1
        with self.assertRaisesRegex(ValueError, "prefix FULL.*verified DFlash2"):
            init(self.runner)
        self.assertIsNone(self.runner.speculator._tree_dflash2_prefix_proof)

    def test_gather_requires_live_proof_and_preserves_prefill_dummy_guards(self):
        init = method("_init_tree_prefix_runtime_proof")
        init(self.runner)
        runner_type = method("gather_batch_req_state", in_class=True)
        state = runner_type.__new__(runner_type)
        state.__dict__.update(vars(self.runner))
        state._uses_flashmla = False
        state.mock_state = NS(has_prefill=False)
        state.cudagraph_manager = NS(tree_prefix_decode=False)
        state.gather_batch_req_state(NS(), False)
        self.assertTrue(state.cudagraph_manager.tree_prefix_decode)
        state.speculator._tree_dflash2_prefix_proof = None
        state.gather_batch_req_state(NS(), False)
        self.assertFalse(state.cudagraph_manager.tree_prefix_decode)
        init(state)
        state.mock_state.has_prefill = True
        state.gather_batch_req_state(NS(), False)
        self.assertFalse(state.cudagraph_manager.tree_prefix_decode)
        state.mock_state.has_prefill = False
        state.gather_batch_req_state(NS(), True)
        self.assertFalse(state.cudagraph_manager.tree_prefix_decode)

    def test_load_initializes_prefix_proof_after_confidence_validation(self):
        source = base.ROOT / "vllm_ascend/worker/v2/model_runner.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        load = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "load_model")
        calls = [node.func.attr for node in ast.walk(load)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"]
        self.assertLess(calls.index("_validate_tree_confidence_runtime"), calls.index("_init_tree_prefix_runtime_proof"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
