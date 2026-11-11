# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the bounded single-request tree FULL graph path.

Load the dependency-free admission helper directly. Method extraction lets
these contracts run without importing an NPU worker or a vLLM installation.
Tensor mask checks run when PyTorch is available.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "tree_prefix_graph_under_test",
    ROOT / "vllm_ascend/worker/v2/spec_decode/tree/prefix_graph.py",
)
prefix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prefix)


def config():
    return NS(
        additional_config={
            "tree_spec_config": dict(enabled=True, budget=64, verification_confidence_source="scorer_logprob")
        },
        scheduler_config=NS(max_num_seqs=1, async_scheduling=False),
        parallel_config=NS(
            **{
                k: 1
                for k in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                )
            }
        ),
        compilation_config=NS(cudagraph_mode=NS(name="FULL"), cudagraph_capture_sizes=[1, 17, 33, 49, 65]),
        model_config=NS(enforce_eager=False, hf_config=NS(model_type="qwen3")),
        speculative_config=NS(
            enforce_eager=False,
            use_dflash=lambda: True,
            enable_adaptive_verification=False,
            draft_model_config=NS(hf_config=NS(dflash_config={"projector_type": "domino", "shift_label": True})),
        ),
        max_concurrent_batches=1,
        lora_config=None,
        kv_transfer_config=None,
    )


class Admission(unittest.TestCase):
    def test_runner_graph_admission_uses_real_gate_and_runtime_policy(self):
        compat_spec = importlib.util.spec_from_file_location(
            "_prefix_runtime_compat", Path(__file__).with_name("test_tree_confidence_runtime_compat.py")
        )
        compat = importlib.util.module_from_spec(compat_spec)
        compat_spec.loader.exec_module(compat)
        fixture = compat.TestTreeConfidenceRuntimeCompatibility("test_native_config_full_vocab_is_admitted")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        policy_spec = importlib.util.spec_from_file_location(
            "_prefix_runtime_policy", ROOT / "vllm_ascend/worker/v2/spec_decode/tree/budget_policy.py"
        )
        policy = importlib.util.module_from_spec(policy_spec)
        policy_spec.loader.exec_module(policy)
        binding = sys.modules["vllm_ascend.worker.v2.spec_decode.tree.budget_policy"]
        binding.admit_verification_confidence_scorer = policy.admit_verification_confidence_scorer
        fixture.runner.speculator._tree_confidence_source = "scorer_logprob"
        fixture.runner.vllm_config = config()
        fixture.validate(fixture.runner)
        self.assertTrue(fixture.facts["graph_admitted"])
        policy.validate_runtime_facts(**fixture.facts)
        fixture.runner.vllm_config.compilation_config.cudagraph_capture_sizes = [17, 65]
        fixture.validate(fixture.runner)
        self.assertFalse(fixture.facts["graph_admitted"])
        with self.assertRaisesRegex(ValueError, "eager"):
            policy.validate_runtime_facts(**fixture.facts)

    def test_supported_five_and_nine_gears(self):
        cfg = config()
        self.assertTrue(prefix.prefix_graph_configured(cfg))
        cfg.compilation_config.cudagraph_capture_sizes = [1, 9, 17, 25, 33, 41, 49, 57, 65]
        self.assertTrue(prefix.prefix_graph_configured(cfg))

    def test_unsupported_runtime_rejected(self):
        changes = [
            ("scheduler_config.max_num_seqs", 2),
            ("scheduler_config.async_scheduling", True),
            ("model_config.enforce_eager", True),
            ("speculative_config.enforce_eager", True),
            ("compilation_config.cudagraph_mode.name", "FULL_DECODE_ONLY"),
            ("max_concurrent_batches", 2),
            ("lora_config", object()),
            ("kv_transfer_config", object()),
            ("speculative_config.enable_adaptive_verification", True),
            ("model_config.hf_config.model_type", "other"),
        ]
        changes += [("parallel_config." + key, 2) for key in vars(config().parallel_config)]
        for path, value in changes:
            with self.subTest(path=path):
                cfg = config()
                obj = cfg
                *parts, last = path.split(".")
                for part in parts:
                    obj = getattr(obj, part)
                setattr(obj, last, value)
                self.assertFalse(prefix.prefix_graph_configured(cfg))

    def test_invalid_gears_rejected(self):
        for gears in (
            [],
            [17, 65],
            [1, 17],
            [1, 17, 65, 65],
            [1, 17, 66],
            [True, 65],
            [1, 17.0, 65],
            list(range(1, 66)),
        ):
            with self.subTest(gears=gears):
                cfg = config()
                cfg.compilation_config.cudagraph_capture_sizes = gears
                self.assertFalse(prefix.prefix_graph_configured(cfg))

    def test_scorer_and_domino_required(self):
        for key, value in [
            ("enabled", False),
            ("budget", True),
            ("budget", 0),
            ("verification_confidence_source", "base_full_vocab"),
        ]:
            cfg = config()
            cfg.additional_config["tree_spec_config"][key] = value
            self.assertFalse(prefix.prefix_graph_configured(cfg))
        for data in (
            {},
            {"projector_type": "other", "shift_label": True},
            {"projector_type": "domino", "shift_label": False},
        ):
            cfg = config()
            cfg.speculative_config.draft_model_config.hf_config.dflash_config = data
            self.assertFalse(prefix.prefix_graph_configured(cfg))

    def test_shape_boundaries_and_rejections(self):
        for capacity in [1, 17, 33, 49, 65]:
            for q in range(1, capacity + 1):
                self.assertEqual(prefix.prefix_shape([q], 1, capacity), (1, q))
        for args in [([], 1, 65), ([0], 1, 65), ([66], 1, 65), ([1, 2], 2, 65), ([True], 1, 65), ([1.0], 1, 65)]:
            with self.assertRaises(ValueError):
                prefix.prefix_shape(*args)

    def test_capture_role_restores_across_draft_scope_and_exception(self):
        self.assertFalse(prefix.prefix_capture_active())
        with prefix.target_prefix_capture(True):
            self.assertTrue(prefix.prefix_capture_active())
            with prefix.target_prefix_capture(False):
                self.assertFalse(prefix.prefix_capture_active())
            self.assertTrue(prefix.prefix_capture_active())
            with self.assertRaises(RuntimeError), prefix.target_prefix_capture(True):
                raise RuntimeError("capture failed")
            self.assertTrue(prefix.prefix_capture_active())
        self.assertFalse(prefix.prefix_capture_active())


class LegacyDraftIntegration(unittest.TestCase):
    @staticmethod
    def method(relative, owner, name, namespace):
        node = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        cls = next(n for n in node.body if isinstance(n, ast.ClassDef) and n.name == owner)
        method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))

        class RemoveImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None

            def visit_Import(self, node):
                return None

        method = RemoveImports().visit(method)
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), "<actual candidate method>", "exec"), namespace)
        return namespace[name]

    def test_v031_capture_initializes_padding_before_real_context_callback(self):
        for legacy, enabled in [(True, True), (True, False), (False, True)]:
            with self.subTest(legacy=legacy, prefix=enabled):
                cfg = config()
                if not enabled:
                    cfg.scheduler_config.max_num_seqs = 2
                calls = []
                padded = []
                tensor = NS(zero_=lambda: None, fill_=lambda value: None)
                manager = NS(graphs={})

                def capture(*args, legacy=legacy, enabled=enabled, padded=padded, manager=manager, **kwargs):
                    callback = kwargs.get("precompute_context_kv")
                    if legacy and enabled:
                        self.assertEqual(padded, [-1])
                        self.assertTrue(callable(callback))
                        callback(1)
                    else:
                        self.assertIsNone(callback)
                    manager.graphs[1] = object()

                manager.capture = capture
                obj = NS(
                    vllm_config=cfg,
                    _speculator_name="DFlashTree",
                    sample_indices=tensor,
                    sample_pos=tensor,
                    sample_idx_mapping=tensor,
                    _context_slot_mappings=NS(fill_=padded.append),
                    query_cudagraph_manager=manager,
                    _prepare_dflash2_graph_attention=lambda: None,
                    _precompute_context_kv=lambda start, end, calls=calls: calls.append((start, end)),
                    _num_graph_context_tokens=lambda n: n * 17,
                    _run_draft_forward=lambda: None,
                    input_buffers=None,
                    block_tables=None,
                    attn_groups=None,
                    kv_cache_config=None,
                    max_model_len=1024,
                    _group_causal=False,
                    tree_graph_stats=NS(
                        record_capture=lambda phase: None,
                        set_workspace_bytes=lambda *a: None,
                        tensor_bytes=lambda values: 0,
                    ),
                    _draft_hidden_buf=None,
                    _capture_tree_graphs=lambda: None,
                    tree_kv_compact=None,
                )
                ns = dict(
                    vllm_version_is=lambda value, legacy=legacy: legacy,
                    prefix_graph_configured=prefix.prefix_graph_configured,
                    PAD_SLOT_ID=-1,
                    logger=NS(info=lambda *a: None),
                )
                fn = self.method(
                    "vllm_ascend/worker/v2/spec_decode/tree/speculator.py", "AscendTreeSpeculator", "capture", ns
                )
                fn(obj)
                self.assertEqual(calls, [(0, 17)] if legacy and enabled else [])

    def test_v031_draft_replay_marks_actual_graph_call(self):
        for legacy, enabled in [(True, True), (True, False), (False, True)]:
            cfg = config()
            if not enabled:
                cfg.scheduler_config.max_num_seqs = 2
            replayed = []
            speculator = NS(
                _draft_graph_used=False,
                attn_backends={"layer": object()},
                input_batch=NS(seq_lens_cpu_upper_bound=[100]),
                build_draft_attn_metadatas=lambda n, bound: {"metadata": True},
            )
            obj = NS(
                vllm_config=cfg,
                speculator=speculator,
                _updatable_graph_replay=lambda desc, metadata, replayed=replayed: replayed.append(
                    (desc.num_tokens, metadata)
                )
                or "output",
            )
            ns = dict(
                vllm_version_is=lambda version, legacy=legacy: legacy,
                prefix_graph_configured=prefix.prefix_graph_configured,
                use_updatable_graph=lambda backend: True,
            )
            fn = self.method(
                "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py", "DFlashAclGraphManager", "run_fullgraph", ns
            )
            self.assertEqual(fn(obj, NS(num_tokens=16, num_reqs=1)), "output")
            self.assertEqual(speculator._draft_graph_used, legacy and enabled)
            self.assertEqual(replayed, [(16, {"metadata": True})])


class ReplayContracts(unittest.TestCase):
    @staticmethod
    def manager():
        path = ROOT / "vllm_ascend/worker/v2/aclgraph_utils.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        owner = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ModelAclGraphManager")
        method = copy.deepcopy(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "dispatch"))

        class RemoveImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None

        method = RemoveImports().visit(method)
        modes = NS(FULL="full", NONE="none")

        class Parent:
            def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, **kwargs):
                self.parent_calls += 1
                if self.override is not None:
                    return self.override
                capacity = min(n for n in self.compilation_config.cudagraph_capture_sizes if n >= num_tokens)
                return NS(cg_mode=modes.FULL, num_tokens=capacity, num_reqs=num_reqs)

        wrapper = ast.ClassDef(
            name="UnderTest",
            bases=[ast.Name(id="Parent", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        code = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), wrapper],
            type_ignores=[],
        )
        namespace = dict(
            Parent=Parent,
            prefix_graph_configured=prefix.prefix_graph_configured,
            CUDAGraphMode=modes,
            BatchExecutionDescriptor=NS,
        )
        exec(compile(ast.fix_missing_locations(code), "<actual graph dispatch>", "exec"), namespace)
        manager = namespace["UnderTest"]()
        manager.vllm_config = config()
        manager.compilation_config = manager.vllm_config.compilation_config
        manager._graphs_captured = True
        manager.tree_prefix_decode = True
        manager.parent_calls = 0
        manager.override = None
        return manager

    def test_dispatch_selects_smallest_graph_for_every_query(self):
        manager = self.manager()
        for q in range(1, 66):
            actual = manager.dispatch(1, q, None, 0)
            self.assertEqual(actual.num_tokens, next(n for n in [1, 17, 33, 49, 65] if n >= q))
            self.assertEqual(actual.num_reqs, 1)
            self.assertEqual(actual.cg_mode, "full")

    def test_prefill_and_incomplete_capture_do_not_replay(self):
        manager = self.manager()
        for captured, decode in [(False, True), (True, False), (False, False)]:
            manager._graphs_captured = captured
            manager.tree_prefix_decode = decode
            actual = manager.dispatch(1, 73, None, 0)
            self.assertEqual(actual.cg_mode, "none")
            self.assertEqual(actual.num_tokens, 73)
        self.assertEqual(manager.parent_calls, 0)

    def test_dispatch_rejects_wrong_gears_or_request_layout(self):
        manager = self.manager()
        for mode, capacity, requests in [("none", 17, 1), ("full", 65, 1), ("full", 17, 2)]:
            manager.override = NS(cg_mode=mode, num_tokens=capacity, num_reqs=requests)
            with self.subTest(mode=mode, capacity=capacity, requests=requests), self.assertRaises(RuntimeError):
                manager.dispatch(1, 13, None, 0)
        manager.override = None
        for requests, loras, microbatches in [(2, 0, 1), (1, 1, 1), (1, 0, 2)]:
            with self.assertRaises(RuntimeError):
                manager.dispatch(requests, 13, None, loras, num_ubatches=microbatches)

    def test_fia_provider_rebinds_real_query_and_kv_lengths(self):
        mask = NS(ndim=4)
        backend = NS(_bsnd_tree_attn_mask=lambda value, requests, capacity: (value, requests, capacity))
        ns = dict(prefix_shape=prefix.prefix_shape, AscendAttentionBackendImpl=backend)
        resolve = LegacyDraftIntegration.method(
            "vllm_ascend/attention/attention_v1.py", "TreeFIAParamProvider", "resolve", ns
        )
        provider = NS(layer_name="layer", num_decodes=1, query_len=65, prefix_gear=True)
        for q in [65, 13, 1, 29, 65]:
            metadata = NS(
                actual_seq_lengths_q=[q],
                num_decodes=1,
                num_prefills=0,
                attn_mask=mask,
                seq_lens_list=[19 + q],
                block_tables=None,
            )
            params = resolve(provider, {"layer": metadata})
            self.assertEqual(params["actual_seq_lengths"], [q])
            self.assertEqual(params["actual_seq_lengths_kv"], [19 + q])
            self.assertEqual(params["atten_mask"], (mask, 1, 65))
        for q, decodes, prefills, bad_mask in [
            (0, 1, 0, mask),
            (66, 1, 0, mask),
            (1, 2, 0, mask),
            (1, 1, 1, mask),
            (1, 1, 0, None),
            (1, 1, 0, NS(ndim=2)),
        ]:
            metadata = NS(
                actual_seq_lengths_q=[q],
                num_decodes=decodes,
                num_prefills=prefills,
                attn_mask=bad_mask,
                seq_lens_list=[19 + q],
                block_tables=None,
            )
            with self.subTest(q=q, decodes=decodes, prefills=prefills), self.assertRaises((ValueError, RuntimeError)):
                resolve(provider, {"layer": metadata})

    def test_query_padding_preserves_single_real_request(self):
        modes = NS(FULL="full")
        ns = dict(prefix_graph_configured=prefix.prefix_graph_configured, CUDAGraphMode=modes)
        pad = LegacyDraftIntegration.method(
            "vllm_ascend/worker/v2/model_runner.py", "NPUModelRunner", "_pad_query_start_loc_for_fia", ns
        )
        obj = NS(vllm_config=config())
        for capacity in [1, 17, 33, 49, 65]:
            for q in range(1, capacity + 1):
                query_start = [0, q]
                actual, requests = pad(obj, capacity, 1, 1, query_start, modes.FULL)
                self.assertIs(actual, query_start)
                self.assertEqual(actual, [0, q])
                self.assertEqual(requests, 1)
        for requests, query_start in [(2, [0, 1, 2]), (1, [0, 0]), (1, [0, 66])]:
            with self.assertRaises(ValueError):
                pad(obj, 65, requests, requests, query_start, modes.FULL)


class TensorMasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("PyTorch is needed for real tensor mask contracts")
        import torch

        cls.torch = torch
        module = ast.parse((ROOT / "vllm_ascend/attention/attention_mask.py").read_text(encoding="utf-8"))
        node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "AttentionMaskBuilder")
        node.decorator_list = []
        namespace = dict(torch=torch)
        code = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(code), "<actual AttentionMaskBuilder>", "exec"), namespace)
        cls.builder_type = namespace["AttentionMaskBuilder"]

    def test_all_queries_mask_padding_and_storage_stability(self):
        torch = self.torch
        builder = self.builder_type(torch.device("cpu"))
        builder.configure_tree_mask(1, 65, 256)
        visibility = torch.zeros((1, 64, 64), dtype=torch.bool)
        ancestors = []
        for j in range(64):
            chain = {j}
            parent = j
            while parent:
                parent = (parent - 1) // 3
                chain.add(parent)
            ancestors.append(chain)
            visibility[0, j, list(chain)] = True
        pointers = None
        for q in list(range(1, 66)) + [65, 13, 1, 65, 17, 65]:
            with self.subTest(q=q):
                actual = builder.get_tree_attention_mask(visibility, torch.tensor([19 + q]), 1, [q])
                expected = torch.ones((1, 1, 65, 256), dtype=torch.bool)
                expected[0, 0, :q, :20] = False
                for row in range(1, q):
                    for ancestor in ancestors[row - 1]:
                        expected[0, 0, row, 20 + ancestor] = False
                self.assertTrue(torch.equal(actual, expected))
                capacity = next(g for g in [1, 17, 33, 49, 65] if g >= q)
                fia = builder.as_bsnd_fia_mask(actual, 1, capacity)
                self.assertEqual(tuple(fia.shape), (1, capacity, 256))
                self.assertTrue(torch.equal(fia, expected[:, 0, :capacity]))
                current = actual.data_ptr(), fia.data_ptr(), fia.stride()
                if pointers is None:
                    pointers = current
                self.assertEqual(current, pointers)

    def test_invalid_query_or_kv_lengths_fail(self):
        torch = self.torch
        builder = self.builder_type(torch.device("cpu"))
        builder.configure_tree_mask(1, 65, 256)
        visibility = torch.eye(64, dtype=torch.bool)[None]
        for lens, seq in [([0], 30), ([66], 100), ([5, 5], 30), ([20], 19), ([1], 257)]:
            with self.subTest(lens=lens, seq=seq), self.assertRaises(ValueError):
                builder.get_tree_attention_mask(visibility, torch.tensor([seq]), 1, lens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
