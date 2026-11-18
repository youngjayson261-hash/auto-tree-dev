# SPDX-License-Identifier: Apache-2.0
"""Real prefix descriptor and capture entry contracts without NPU capture."""

import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dflash2_confidence_admission as base
import test_dflash2_prefix_runtime_admission as runtime
import test_qwen38_tree_query_graph as query


def fixture():
    runner, batch = query.fixture()
    cfg, proof = runtime.runtime_case(2)
    runner.vllm_config = cfg
    runner.speculator._tree_dflash2_prefix_proof = proof
    runner.speculator.tree_graph_stats = NS(record_capture=lambda *args: None)
    graphs = {query.Descriptor(n, max_query_len=n): object() for n in (1, 17, 33, 49, 65)}
    runner.cudagraph_manager.graphs = graphs
    runner.tree_verification_plan["graph_descriptor"] = next(desc for desc in graphs if desc.num_tokens == 17)
    return runner, batch, proof.tp_group


def capture_class(calls):
    path = base.ROOT / "vllm_ascend/worker/v2/aclgraph_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelAclGraphManager")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "capture")
    method.returns = None
    for arg in method.args.args + method.args.kwonlyargs:
        arg.annotation = None

    class Parent:
        def capture(self, *args, **kwargs):
            calls.append("capture")
            return "captured"

    def wrap(model, **kwargs):
        calls.append("wrap")
        return model

    namespace = dict(Parent=Parent, ModelWithContext=wrap, communicator_switch=nullcontext,
                     CUDAGraphMode=query.Mode, logger=NS(info=lambda *args: None))
    subject = ast.ClassDef(name="Manager", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[],
                           body=[method], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[subject], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["Manager"]


class TestPrefixDescriptorAndCapture(unittest.TestCase):
    def setUp(self):
        self.runner, self.batch, self.group = fixture()
        distributed = ModuleType("vllm.distributed")
        distributed.get_tp_group = lambda: self.group
        speculator = ModuleType("vllm_ascend.worker.v2.spec_decode.tree.speculator")
        speculator.AscendTreeSpeculator = type("OtherSpeculator", (), {})
        self.binding = patch.dict(sys.modules, {
            "vllm.distributed": distributed,
            "vllm_ascend.worker.v2.spec_decode.tree.prefix_graph": base.prefix,
            "vllm_ascend.worker.v2.spec_decode.tree.confidence_admission": base.admission,
            "vllm_ascend.worker.v2.spec_decode.tree.speculator": speculator,
        })
        self.binding.start()
        self.addCleanup(self.binding.stop)
        self.query = query.load_subject()

    def capacity(self):
        return self.query.qwen_tree_query_graph_capacity(self.runner, self.batch, 65)

    def test_real_none_uniform_prefix_descriptor_is_accepted_with_live_proof(self):
        desc = self.runner.tree_verification_plan["graph_descriptor"]
        self.assertIsNone(desc.uniform_token_count)
        self.assertEqual(desc.max_query_len, 17)
        self.assertEqual(self.capacity(), 17)

    def test_none_uniform_requires_exact_maxquery_captured_gear(self):
        for desc in (
            query.Descriptor(17),
            query.Descriptor(17, max_query_len=13),
            query.Descriptor(17, uniform_token_count=13, max_query_len=17),
        ):
            self.runner.tree_verification_plan["graph_descriptor"] = desc
            self.runner.cudagraph_manager.graphs = {desc: object()}
            self.assertIsNone(self.capacity())

    def test_hybrid_query_cannot_use_confidence_or_missing_prefix_proof(self):
        self.assertEqual(self.capacity(), 17)
        for proof in (None, True, NS(tp_size=2)):
            self.runner.speculator._tree_dflash2_prefix_proof = proof
            self.assertIsNone(self.capacity())

    def manager(self, calls):
        cls = capture_class(calls)
        manager = cls()
        manager.model_runner = self.runner
        manager.vllm_config = self.runner.vllm_config
        manager.compilation_config = self.runner.vllm_config.compilation_config
        manager.graphs = self.runner.cudagraph_manager.graphs.copy()
        return manager

    def capture(self, manager):
        return manager.capture(object(), NS(device_metadata=None), None, None, None, [], None)

    def test_capture_rejects_unproven_hybrid_before_wrapper_or_collectives(self):
        calls = []
        manager = self.manager(calls)
        self.runner.speculator._tree_dflash2_prefix_proof = None
        with self.assertRaisesRegex(ValueError, "Hybrid prefix FULL capture.*proof"):
            self.capture(manager)
        self.assertEqual(calls, [])

    def test_capture_with_proof_keeps_context_and_all_gear_completion(self):
        calls = []
        manager = self.manager(calls)
        self.assertEqual(self.capture(manager), "captured")
        self.assertEqual(calls, ["wrap", "capture"])
        manager.graphs.pop(next(iter(manager.graphs)))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.capture(manager)


if __name__ == "__main__":
    unittest.main(verbosity=2)
