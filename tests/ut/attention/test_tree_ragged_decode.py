"""CPU regressions for target tree query packing and per-request masking.

Load the production definitions without importing the accelerator plugin, so
these tests can also run in an isolated CPU-only child process.
"""

import ast
import copy
import json
import os
import sys
import sysconfig
import unittest
from pathlib import Path
from types import SimpleNamespace

if __name__ == "__main__":
    for path in (sysconfig.get_path("purelib"), sysconfig.get_path("platlib")):
        if path not in sys.path:
            sys.path.append(path)

import torch

SOURCE_ROOT = Path(os.environ.get("TREE_RAGGED_SOURCE_ROOT", Path(__file__).resolve().parents[3]))
CAPACITY = 49


def load_definitions(filename, namespace, names, methods=None):
    source = ast.parse((SOURCE_ROOT / "vllm_ascend/attention" / filename).read_text(encoding="utf-8"))
    selected = []
    for node in source.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node = copy.deepcopy(node)
            if isinstance(node, ast.ClassDef):
                node.bases = []
                node.decorator_list = []
                if methods is not None:
                    node.body = [
                        method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in methods
                    ]
            selected.append(node)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)


def make_runtime():
    namespace = {
        "torch": torch,
        "_tree_spec_mask_caps": lambda: None,
        "tree_spec_enabled": lambda: True,
        "_EXTRA_CTX": SimpleNamespace(is_draft_model=False),
        "AscendAttentionState": SimpleNamespace(PrefillNoCache="prefill_no_cache"),
        "dummy_tree_mask_for_capture": lambda tokens, requests: tokens == requests * CAPACITY,
        "need_dummy_tree_visibility_for_capture": lambda: False,
    }
    load_definitions(
        "tree_spec.py",
        namespace,
        {"align_up", "tree_decode_query_lens", "pack_tree_queries", "unpack_tree_queries"},
    )
    load_definitions("attention_mask.py", namespace, {"AttentionMaskBuilder"})
    builder = namespace["AttentionMaskBuilder"](torch.device("cpu"))
    builder.configure_tree_mask(5, CAPACITY, 256)
    namespace["AttentionMaskBuilder"] = lambda device: builder
    namespace["tree_fia_bsnd_shape"] = lambda tokens: (tokens // CAPACITY, CAPACITY) if tokens % CAPACITY == 0 else None
    load_definitions(
        "attention_v1.py",
        namespace,
        {"AscendAttentionBackendImpl"},
        {
            "_tree_fia_bsnd_shape",
            "_tree_decode_fia_eligible",
            "_tnd_sparse34_mask",
            "_bsnd_tree_attn_mask",
            "_forward_tree_decode_fia",
            "full_graph_tree_fia",
            "_forward_fia_chunked_prefill_split",
        },
    )
    impl = namespace["AscendAttentionBackendImpl"]()
    impl.num_heads = impl.num_kv_heads = 1
    impl.head_size = 3
    impl.scale = 0.5
    impl.key_cache = impl.value_cache = None
    return namespace, builder, impl


def visibility(batch):
    result = torch.zeros(batch, CAPACITY - 1, CAPACITY - 1, dtype=torch.bool)
    for node in range(CAPACITY - 1):
        ancestor = node
        while ancestor >= 0:
            result[:, node, ancestor] = True
            ancestor = (ancestor - 1) // 2
    return result


def expected_mask(query_lens, seq_lens, visible):
    expected = torch.ones(len(query_lens), CAPACITY, 256, dtype=torch.bool)
    for batch, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
        prefix = seq_len - query_len
        for row in range(query_len):
            expected[batch, row, : prefix + 1] = False
            for node in range(query_len - 1):
                if row > 0 and visible[batch, row - 1, node]:
                    expected[batch, row, prefix + 1 + node] = False
    return expected


class TestTreeRaggedDecode(unittest.TestCase):
    def setUp(self):
        self.ns, self.builder, self.impl = make_runtime()

    def test_actual_boundaries_and_padding_validation(self):
        helper = self.ns["tree_decode_query_lens"]
        for lengths in ([49, 1], [1, 49], [1, 2, 7, 25, 49], [49, 49]):
            ends = torch.tensor(lengths).cumsum(0).tolist()
            self.assertEqual(helper(ends, len(lengths), sum(lengths), CAPACITY), lengths)
        for ends, requests, tokens in (([], 1, 1), ([0], 1, 0), ([50], 1, 50), ([49, 49], 2, 49), ([49, 50], 1, 50)):
            self.assertIsNone(helper(ends, requests, tokens, CAPACITY))

    def test_all_pruned_lengths_pack_and_restore(self):
        for length in range(1, CAPACITY + 1):
            lengths = [length, CAPACITY, 1]
            query = torch.arange(sum(lengths) * 3).reshape(-1, 1, 3)
            packed = self.ns["pack_tree_queries"](query, lengths, CAPACITY)
            self.assertEqual(tuple(packed.shape), (3, CAPACITY, 1, 3))
            torch.testing.assert_close(self.ns["unpack_tree_queries"](packed, lengths), query)
            self.assertEqual(torch.count_nonzero(packed[0, length:]).item(), 0)
            self.assertEqual(torch.count_nonzero(packed[2, 1:]).item(), 0)

    def test_uniform_capacity_uses_original_storage(self):
        query = torch.randn(98, 1, 3)
        packed = self.ns["pack_tree_queries"](query, [49, 49], CAPACITY)
        self.assertEqual(packed.data_ptr(), query.data_ptr())
        self.assertEqual(self.ns["unpack_tree_queries"](packed, [49, 49]).data_ptr(), query.data_ptr())

    def test_uniform_mask_preserves_existing_full_capacity_semantics(self):
        visible = visibility(2)
        seqs = torch.tensor([171, 255])
        legacy = self.builder.get_tree_attention_mask(visible, seqs, 2).clone()
        actual = self.builder.get_tree_attention_mask(visible, seqs, 2, [49, 49])
        torch.testing.assert_close(actual, legacy)
        torch.testing.assert_close(actual[:, 0], expected_mask([49, 49], seqs.tolist(), visible))

    def test_exact_mask_49_plus_1_and_inverse(self):
        for lengths, seqs in (([49, 1], [171, 255]), ([1, 49], [255, 171]), ([1, 2, 7, 25, 49], [256] * 5)):
            visible = visibility(len(lengths))
            mask = self.builder.get_tree_attention_mask(visible, torch.tensor(seqs), len(lengths), lengths)
            torch.testing.assert_close(mask[:, 0], expected_mask(lengths, seqs, visible))
            torch.testing.assert_close(self.builder.as_bsnd_fia_mask(mask, len(lengths), CAPACITY), mask[:, 0])

    def test_mask_buffer_refresh_after_request_lengths_change(self):
        visible = visibility(2)
        self.builder.get_tree_attention_mask(visible, torch.tensor([171, 255]), 2, [49, 1])
        pointer = self.builder._tree_fia_bsnd.data_ptr()
        mask = self.builder.get_tree_attention_mask(visible, torch.tensor([255, 171]), 2, [1, 49])
        self.assertEqual(pointer, self.builder._tree_fia_bsnd.data_ptr())
        torch.testing.assert_close(self.builder._tree_fia_bsnd[:2], expected_mask([1, 49], [255, 171], visible))
        self.assertTrue(mask[0, 0, 1:].all())

    def test_mask_rejects_impossible_lengths(self):
        for lengths, seqs in (([0], [100]), ([50], [100]), ([49], [48]), ([1], [257]), ([], [100])):
            with self.assertRaises(ValueError):
                self.builder.get_tree_attention_mask(visibility(1), torch.tensor(seqs), 1, lengths)

    def test_ragged_dispatch_preserves_paged_kv_and_output_order(self):
        for lengths in ([49, 1], [1, 49], [7, 25], [49, 49]):
            with self.subTest(lengths=lengths):
                seqs = [171, 255]
                visible = visibility(2)
                mask = self.builder.get_tree_attention_mask(visible, torch.tensor(seqs), 2, lengths)
                table = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]], dtype=torch.int32)
                metadata = SimpleNamespace(
                    attn_mask=mask,
                    actual_seq_lengths_q=torch.tensor(lengths).cumsum(0).tolist(),
                    num_prefills=0,
                    num_decodes=2,
                    block_tables=table,
                    attn_state="decode",
                )
                self.assertTrue(self.impl._tree_decode_fia_eligible(metadata))
                generator = torch.Generator().manual_seed(123)
                query = torch.randn(sum(lengths), 1, 3, generator=generator, dtype=torch.float64)
                key = torch.randn(4, 64, 3, generator=generator, dtype=torch.float64)
                value = torch.randn(4, 64, 3, generator=generator, dtype=torch.float64)
                calls = []

                def kernel(
                    calls=calls,
                    lengths=lengths,
                    seqs=seqs,
                    table=table,
                    expected_key=key,
                    expected_value=value,
                    **kwargs,
                ):
                    calls.append(kwargs)
                    result = torch.full_like(kwargs["query"], 1e6)
                    self.assertEqual(kwargs["input_layout"], "BSND")
                    self.assertEqual(kwargs["sparse_mode"], 0)
                    self.assertEqual(kwargs["actual_seq_lengths"], lengths)
                    self.assertEqual(kwargs["actual_seq_lengths_kv"], seqs)
                    self.assertEqual(kwargs["block_table"].data_ptr(), table.data_ptr())
                    self.assertEqual(kwargs["key"].data_ptr(), expected_key.data_ptr())
                    for batch, length in enumerate(lengths):
                        k = expected_key[table[batch].long()].flatten(0, 1)[: seqs[batch]]
                        v = expected_value[table[batch].long()].flatten(0, 1)[: seqs[batch]]
                        logits = kwargs["query"][batch, :length, 0] @ k.T * self.impl.scale
                        logits.masked_fill_(kwargs["atten_mask"][batch, :length, : seqs[batch]], -torch.inf)
                        result[batch, :length, 0] = logits.softmax(-1) @ v
                    return result, None

                self.ns["DeviceOperator"] = SimpleNamespace(npu_fused_infer_attention_score=kernel)
                actual = self.impl._forward_tree_decode_fia(
                    query, key, value, value, 64, table, metadata, seqs, sum(lengths), 2
                )
                expected = []
                reference_mask = expected_mask(lengths, seqs, visible)
                offset = 0
                for batch, length in enumerate(lengths):
                    k = key[table[batch].long()].flatten(0, 1)[: seqs[batch]]
                    v = value[table[batch].long()].flatten(0, 1)[: seqs[batch]]
                    for row in range(length):
                        allowed = ~reference_mask[batch, row, : seqs[batch]]
                        probabilities = (k[allowed] @ query[offset + row, 0] * self.impl.scale).softmax(0)
                        expected.append(probabilities @ v[allowed])
                    offset += length
                torch.testing.assert_close(actual[:, 0], torch.stack(expected), rtol=1e-12, atol=1e-12)
                self.assertEqual(len(calls), 1)

    def test_mixed_prefill_still_splits_tree_decode_from_causal_prefill(self):
        lengths = [49, 1]
        mask = self.builder.get_tree_attention_mask(visibility(2), torch.tensor([171, 255]), 2, lengths)
        table = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2], [0, 1, 2, 3]], dtype=torch.int32)
        metadata = SimpleNamespace(
            attn_mask=mask,
            actual_seq_lengths_q=[49, 50, 225],
            num_decode_tokens=50,
            num_prefills=1,
            num_decodes=2,
            seq_lens_list=[171, 255, 175],
        )
        query = torch.randn(225, 1, 3)
        cache = torch.randn(4, 64, 3)
        calls = []

        def kernel(**kwargs):
            calls.append(kwargs)
            return kwargs["query"] * 2, None

        self.ns["DeviceOperator"] = SimpleNamespace(npu_fused_infer_attention_score=kernel)
        output = self.impl._forward_fia_chunked_prefill_split(
            query, cache, cache, cache, cache, 64, table, metadata, torch.empty_like(query)
        )
        torch.testing.assert_close(output, query * 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual((calls[0]["input_layout"], calls[0]["sparse_mode"]), ("BSND", 0))
        self.assertEqual(calls[0]["actual_seq_lengths"], [49, 1])
        self.assertEqual((calls[1]["input_layout"], calls[1]["sparse_mode"]), ("TND", 3))
        self.assertEqual(calls[1]["actual_seq_lengths"], [175])
        self.assertEqual(calls[1]["actual_seq_lengths_kv"], [175])
        torch.testing.assert_close(calls[1]["block_table"], table[2:])
        self.assertEqual(tuple(calls[1]["atten_mask"].shape), (2048, 2048))

    def test_target_tree_cannot_fall_back_to_causal_tnd(self):
        with self.assertRaisesRegex(RuntimeError, "causal fallback"):
            self.impl._tnd_sparse34_mask(torch.ones(2, 1, 49, 256), torch.device("cpu"))
        self.ns["tree_spec_enabled"] = lambda: False
        causal_mask = self.impl._tnd_sparse34_mask(torch.ones(2, 1, 49, 256), torch.device("cpu"))
        self.assertEqual(tuple(causal_mask.shape), (2048, 2048))

    def test_ragged_graph_fails_explicitly(self):
        metadata = SimpleNamespace(actual_seq_lengths_q=[49, 50], num_decodes=2)
        self.impl._get_fia_params = lambda *args: (None, None, 64, None, None)
        with self.assertRaisesRegex(RuntimeError, "ragged graph capture is unsupported"):
            self.impl.full_graph_tree_fia(None, None, None, metadata, None)

    def test_metadata_passes_only_host_decode_query_lengths(self):
        source = (SOURCE_ROOT / "vllm_ascend/attention/attention_v1.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        builder = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "AscendAttentionMetadataBuilder"
        )
        build = next(node for node in builder.body if isinstance(node, ast.FunctionDef) and node.name == "build")
        call = next(
            node
            for node in ast.walk(build)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_attention_mask"
        )
        expression = next(keyword.value for keyword in call.keywords if keyword.arg == "query_lens")
        code = compile(ast.Expression(expression), "query_lens", "eval")
        for starts in ([0, 49, 224], [0, 49, 50, 129]):
            common = SimpleNamespace(for_cudagraph_capture=False)
            count = len(starts) - 2
            actual = eval(
                code,
                {
                    "common_attn_metadata": common,
                    "query_start_loc_cpu": torch.tensor(starts),
                    "num_decodes": count,
                },
            )
            self.assertEqual(actual, [49] if count == 1 else [49, 1])
        self.assertIn("_forward_fia_chunked_prefill_split", source)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTreeRaggedDecode)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    forbidden = sorted(
        name
        for name in sys.modules
        if name in ("torch_npu", "vllm", "vllm_ascend") or name.startswith(("torch_npu.", "vllm.", "vllm_ascend."))
    )
    audit = {
        "forbidden_imports": forbidden,
        "torch_device_backend_autoload": os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"),
        "ascend_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tests_run": result.testsRun,
    }
    print(json.dumps(audit, sort_keys=True))
    if (
        forbidden
        or audit["torch_device_backend_autoload"] != "0"
        or audit["ascend_visible_devices"] != ""
        or audit["cuda_visible_devices"] != ""
    ):
        raise SystemExit(91)
    raise SystemExit(0 if result.wasSuccessful() else 1)
