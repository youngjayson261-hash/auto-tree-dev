# SPDX-License-Identifier: Apache-2.0
"""Native ragged-BS graph checks; run with one explicitly selected NPU."""

import json
import os
import unittest
from types import SimpleNamespace as NS


@unittest.skipUnless(os.environ.get("ECHO_TEST_NPU") == "1", "explicit native validation only")
class NativeBatchPrefix(unittest.TestCase):
    def test_ragged_fia_updates_keep_request_boundaries_and_padding_kv(self):
        import torch
        import torch_npu

        from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionBackendImpl,
            TreeFIAParamProvider,
            _tree_fia_prefix_out,
        )
        from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph, register_task
        from vllm_ascend.device.device_op import DeviceOperator
        from vllm_ascend.utils import enable_custom_op

        self.assertEqual(torch.npu.device_count(), 1)
        torch.npu.set_device(0)
        self.assertTrue(enable_custom_op())
        dtype, device = torch.bfloat16, "npu"
        torch.manual_seed(17)
        cache_shape = (4, 128, 2, 128)
        key_cache = torch.full(cache_shape, -31, dtype=dtype, device=device)
        value_cache = torch.full(cache_shape, -29, dtype=dtype, device=device)
        keys = torch.ones((34, 2, 128), dtype=dtype, device=device)
        values = -torch.ones_like(keys)
        slots = torch.full((34,), -1, dtype=torch.int32, device=device)

        def store():
            DeviceOperator.reshape_and_cache(
                key=keys,
                value=values,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mapping=slots,
                use_bnsd=False,
            )

        store()
        sentinel_graph = torch.npu.NPUGraph()
        with torch.npu.graph(sentinel_graph):
            store()
        checks = []
        for lengths in ([17, 1], [1, 17], [5, 13], [1, 1], [17, 17]):
            key_cache.fill_(-31)
            value_cache.fill_(-29)
            slots.fill_(-1)
            start = 0
            for row, length in enumerate(lengths):
                slots[start : start + length].copy_(
                    torch.arange((row + 1) * 128, (row + 1) * 128 + length, dtype=torch.int32, device=device)
                )
                start += length
            sentinel_graph.replay()
            torch.npu.synchronize()
            expected_k = torch.full(cache_shape, -31, dtype=dtype)
            expected_v = torch.full(cache_shape, -29, dtype=dtype)
            for row, length in enumerate(lengths):
                expected_k[row + 1, :length].fill_(1)
                expected_v[row + 1, :length].fill_(-1)
            self.assertTrue(torch.equal(key_cache.cpu(), expected_k))
            self.assertTrue(torch.equal(value_cache.cpu(), expected_v))
            checks.append({"kind": "all_KV_padding_sentinels", "query_lengths": lengths})

        key_cache.copy_(torch.randn(cache_shape, dtype=dtype, device=device) * 0.2)
        value_cache.copy_(torch.randn(cache_shape, dtype=dtype, device=device) * 0.2)
        key = key_cache.view(4, 128, 256)
        value = value_cache.view(4, 128, 256)
        tables = torch.tensor([[1], [2]], dtype=torch.int32, device=device)
        query = torch.randn((34, 4, 128), dtype=dtype, device=device) * 0.2
        mask = torch.ones((2, 1, 17, 256), dtype=torch.bool, device=device)
        mask_builder = AttentionMaskBuilder(torch.device(device))
        mask_builder.configure_tree_mask(2, 17, 256)
        mask_builder._tree_attn_mask = mask
        history = [32, 64]

        def metadata(lengths, reverse=False):
            hist = history[::-1] if reverse else history
            cpu_mask = torch.ones((2, 1, 17, 256), dtype=torch.bool)
            for row, length in enumerate(lengths):
                cpu_mask[row, :, :length, : hist[row]] = False
                for position in range(length):
                    parent = position
                    while True:
                        cpu_mask[row, 0, position, hist[row] + parent] = False
                        if parent == 0:
                            break
                        parent = (parent - 1) // 2
            mask.copy_(cpu_mask)
            # Match the real metadata producer: the updatable FIA task retains
            # this stable 3D mask buffer, not the raw 4D tree mask.
            mask_builder._sync_tree_fia_bsnd(2, 17, 256)
            return NS(
                actual_seq_lengths_q=[lengths[0], sum(lengths)],
                num_decodes=2,
                num_prefills=0,
                attn_mask=mask,
                seq_lens_list=[hist[row] + n for row, n in enumerate(lengths)],
                block_tables=tables.flip(0).contiguous() if reverse else tables,
                query_start_loc_gpu=torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32, device=device),
            )

        common = dict(
            key=key,
            value=value,
            input_layout="BSND",
            block_size=128,
            num_key_value_heads=2,
            num_heads=4,
            scale=128**-0.5,
            sparse_mode=0,
            pre_tokens=2147483647,
            next_tokens=2147483647,
        )
        graphs, outputs, operations, retained = {}, {}, {}, []
        for capacity in (17, 9, 1):
            md = metadata([capacity, capacity])
            packed = torch.empty((2, capacity, 4, 128), dtype=dtype, device=device)
            packed_out = torch.empty_like(packed)
            flat_out = torch.empty((2 * capacity, 4, 128), dtype=dtype, device=device)
            params = dict(
                common,
                query=packed,
                block_table=md.block_tables,
                atten_mask=AscendAttentionBackendImpl._bsnd_tree_attn_mask(mask, 2, capacity),
                actual_seq_lengths=[capacity, capacity],
                actual_seq_lengths_kv=md.seq_lens_list,
            )
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**params)
            lse = torch.empty(1, dtype=dtype, device=device)
            operation = dict(
                params,
                query=query[: 2 * capacity],
                packed_query=packed,
                flat_output=flat_out,
                query_start_loc=md.query_start_loc_gpu,
                workspace=workspace,
                out=[packed_out, lse],
            )
            _tree_fia_prefix_out(**operation)
            torch.npu.synchronize()
            provider = TreeFIAParamProvider("layer", 2, capacity, True)
            graph = UpdatableGraph()
            with torch.npu.graph(graph):
                register_task(_tree_fia_prefix_out, operation, provider)
            graphs[capacity], outputs[capacity] = graph, flat_out
            operations[capacity] = operation
            retained.append((operation, provider, workspace, packed, packed_out))

        update_stream = torch.npu.Stream()
        cases = [
            (17, [17, 1], False),
            (17, [1, 17], False),
            (17, [5, 13], True),
            (9, [9, 2], False),
            (1, [1, 1], True),
            (17, [17, 17], False),
            (17, [1, 1], False),
            (9, [2, 9], True),
            (17, [13, 5], False),
        ]
        for capacity, lengths, reverse in cases:
            md = metadata(lengths, reverse)
            expected, start = [], 0
            for row, length in enumerate(lengths):
                params = dict(
                    common,
                    query=query[start : start + length].view(1, length, 4, 128),
                    block_table=md.block_tables[row : row + 1],
                    atten_mask=mask[row, :, :length].contiguous(),
                    actual_seq_lengths=[length],
                    actual_seq_lengths_kv=[md.seq_lens_list[row]],
                )
                expected.append(torch_npu.npu_fused_infer_attention_score(**params)[0].view(length, 4, 128))
                start += length
            expected = torch.cat(expected)
            eager_params = {
                **operations[capacity],
                **TreeFIAParamProvider("layer", 2, capacity, True).resolve({"layer": md}),
            }
            _tree_fia_prefix_out(**eager_params)
            torch.npu.synchronize()
            eager_error = (outputs[capacity][: sum(lengths)].float() - expected.float()).abs().max().item()
            print(json.dumps({"eager_error": eager_error, "lengths": lengths, "capacity": capacity}), flush=True)
            torch.testing.assert_close(outputs[capacity][: sum(lengths)], expected, atol=0.003, rtol=0.03)
            outputs[capacity].fill_(-99)
            resolved = graphs[capacity].resolve_tasks(ContextSource({"layer": md}))
            update_stream.wait_stream(torch.npu.current_stream())
            graphs[capacity].replay()
            graphs[capacity].update(update_stream, resolved)
            torch.npu.current_stream().wait_stream(update_stream)
            torch.npu.synchronize()
            actual = outputs[capacity][: sum(lengths)]
            error = (actual.float() - expected.float()).abs().max().item()
            torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.03)
            self.assertTrue(bool((outputs[capacity][sum(lengths) :] == 0).all()))
            checks.append(
                {
                    "kind": "ragged_graph_vs_independent_eager_requests",
                    "R": 2,
                    "capacity_per_request": capacity,
                    "query_lengths": lengths,
                    "request_reordered": reverse,
                    "max_abs_error": error,
                }
            )
        print(json.dumps({"status": "passed", "checks": checks}), flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
