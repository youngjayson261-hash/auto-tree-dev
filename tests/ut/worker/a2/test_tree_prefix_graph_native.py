# SPDX-License-Identifier: Apache-2.0
"""Real backend KV padding and updatable BSND FIA tests on logical NPU0."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS


def _run_native_checks(root):
    import torch
    import torch_npu

    importlib.import_module("vllm_ascend.ops")
    from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
    from vllm_ascend.attention.attention_v1 import TreeFIAParamProvider
    from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph, register_task
    from vllm_ascend.device.device_op import DeviceOperator

    assert torch.npu.device_count() == 1
    torch.npu.set_device(0)
    dtype = torch.bfloat16
    cache_shape = (4, 128, 8, 128)
    kc = torch.full(cache_shape, -31, dtype=dtype, device="npu")
    vc = torch.full(cache_shape, -29, dtype=dtype, device="npu")
    keys = torch.ones((65, 8, 128), dtype=dtype, device="npu")
    vals = -torch.ones_like(keys)
    slots = torch.full((65,), -1, dtype=torch.int32, device="npu")
    rows = []

    def store(Q):
        DeviceOperator.reshape_and_cache(
            key=keys[:Q], value=vals[:Q], key_cache=kc, value_cache=vc, slot_mapping=slots[:Q], use_bnsd=False
        )

    def setup(q, generation):
        kc.fill_(-31)
        vc.fill_(-29)
        keys.fill_(generation)
        vals.fill_(-generation)
        slots.fill_(-1)
        slots[:q].copy_(torch.arange(128, 128 + q, dtype=torch.int32, device="npu"))

    def check_store(Q, q, generation, graph=None):
        setup(q, generation)
        if graph is None:
            store(Q)
        else:
            graph.replay()
        torch.npu.synchronize()
        ek = torch.full(cache_shape, -31, dtype=dtype)
        ev = torch.full(cache_shape, -29, dtype=dtype)
        ek[1, :q].fill_(generation)
        ev[1, :q].fill_(-generation)
        assert torch.equal(kc.cpu(), ek), ("unexpected key cache write", Q, q)
        assert torch.equal(vc.cpu(), ev), ("unexpected value cache write", Q, q)
        rows.append(dict(kind="KV_sentinel", Q=Q, q=q, graph=graph is not None, all_cache_slots_checked=True))

    check_store(65, 13, 1)
    graphs = {}
    for Q in (17, 65):
        setup(13, 2)
        store(Q)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            store(Q)
        graphs[Q] = graph
    for i, (Q, q) in enumerate([(65, 13), (65, 5), (65, 13), (17, 13), (65, 65), (65, 1)], 3):
        check_store(Q, q, i, graphs[Q])

    # Same persistent mask, physical Q layout, backend provider and update order
    # as ModelAclGraphManager. Eager short-query FIA is an independent shape.
    torch.manual_seed(123)
    query = torch.randn((65, 32, 128), dtype=dtype, device="npu") * 0.2
    kc.copy_(torch.randn(cache_shape, dtype=dtype, device="npu") * 0.2)
    vc.copy_(torch.randn(cache_shape, dtype=dtype, device="npu") * 0.2)
    key = kc.view(4, 128, 8 * 128)
    value = vc.view(4, 128, 8 * 128)
    table = torch.tensor([[1]], dtype=torch.int32, device="npu")
    builder = AttentionMaskBuilder(torch.device("npu", 0))
    builder.configure_tree_mask(1, 65, 256)
    visibility_cpu = torch.zeros((1, 64, 64), dtype=torch.bool)
    for j in range(64):
        p = j
        visibility_cpu[0, j, p] = True
        while p:
            p = (p - 1) // 3
            visibility_cpu[0, j, p] = True
    visibility = visibility_cpu.to("npu")

    def metadata(q):
        mask = builder.get_tree_attention_mask(visibility, torch.tensor([19 + q], dtype=torch.int32), 1, [q])
        return NS(
            actual_seq_lengths_q=[q],
            num_decodes=1,
            num_prefills=0,
            attn_mask=mask,
            seq_lens_list=[19 + q],
            block_tables=table,
        )

    def kwargs(Q, q, md):
        return dict(
            query=query[:Q].view(1, Q, 32, 128),
            key=key,
            value=value,
            atten_mask=builder.as_bsnd_fia_mask(md.attn_mask, 1, Q),
            block_table=table,
            input_layout="BSND",
            block_size=128,
            actual_seq_lengths=[q],
            actual_seq_lengths_kv=[19 + q],
            num_key_value_heads=8,
            num_heads=32,
            scale=128**-0.5,
            sparse_mode=0,
            pre_tokens=2147483647,
            next_tokens=2147483647,
        )

    fia_graphs = {}
    outputs = {}
    retained = []
    for Q in [1, 17, 33, 49, 65]:
        md = metadata(Q)
        kw = kwargs(Q, Q, md)
        ws = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**kw)
        output = torch.empty_like(kw["query"])
        lse = torch.empty(1, dtype=dtype, device="npu")
        op_kw = dict(kw, workspace=ws, out=[output, lse])
        torch_npu.npu_fused_infer_attention_score.out(**op_kw)
        torch.npu.synchronize()
        graph = UpdatableGraph()
        provider = TreeFIAParamProvider("layer", 1, Q, True)
        with torch.npu.graph(graph):
            register_task(torch_npu.npu_fused_infer_attention_score.out, op_kw, provider)
        fia_graphs[Q] = graph
        outputs[Q] = output
        retained.append((kw, ws, output, lse, provider))
    update_stream = torch.npu.Stream()
    for Q, q in [(65, 65), (65, 13), (65, 5), (65, 13), (17, 13), (1, 1), (33, 29), (49, 37), (65, 65)]:
        md = metadata(q)
        # Compute expected eagerly with exactly q rows, before graph replay.
        expected = torch_npu.npu_fused_infer_attention_score(**kwargs(q, q, md))[0]
        resolved = fia_graphs[Q].resolve_tasks(ContextSource({"layer": md}))
        update_stream.wait_stream(torch.npu.current_stream())
        fia_graphs[Q].replay()
        fia_graphs[Q].update(update_stream, resolved)
        torch.npu.current_stream().wait_stream(update_stream)
        torch.npu.synchronize()
        error = (outputs[Q][:, :q].float() - expected.float()).abs().max().item()
        assert torch.allclose(outputs[Q][:, :q], expected, atol=0.003, rtol=0.03), (Q, q, error)
        rows.append(
            dict(kind="FIA_graph_vs_eager", Q=Q, q=q, max_abs_error=error, physical_query_rows=Q, actual_query_rows=q)
        )
    result = dict(
        passed=True,
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        logical_device=torch.npu.current_device(),
        kv_padding_sentinel_checks=7,
        fia_update_checks=9,
        use_bnsd=False,
        checks=rows,
        model_loads=0,
    )
    (root / "native-checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


try:
    NATIVE_PACKAGE_AVAILABLE = importlib.util.find_spec("torch_npu") is not None
except (ImportError, ValueError):
    NATIVE_PACKAGE_AVAILABLE = False


@unittest.skipUnless(NATIVE_PACKAGE_AVAILABLE, "requires native torch_npu and one isolated NPU")
class NativePrefixGraphContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        if not visible or len(visible.split(",")) != 1:
            raise unittest.SkipTest("select exactly one isolated NPU externally")
        import torch

        importlib.import_module("torch_npu")

        if torch.npu.device_count() != 1:
            raise unittest.SkipTest("exactly one visible NPU is required")

    def test_padding_cache_sentinels_and_fia_replay(self):
        with tempfile.TemporaryDirectory(prefix="tree-prefix-graph-") as temporary:
            root = Path(temporary)
            _run_native_checks(root)
            result = json.loads((root / "native-checks.json").read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])
            self.assertEqual(result["kv_padding_sentinel_checks"], 7)
            self.assertEqual(result["fia_update_checks"], 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
