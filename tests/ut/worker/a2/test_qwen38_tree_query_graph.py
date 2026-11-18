# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for GDN accepted-query transport; no NPU capture is claimed."""

import ast
import unittest
from dataclasses import dataclass, fields, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch


class Mode(Enum):
    FULL = "FULL"
    NONE = "NONE"


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    cg_mode: Mode = Mode.FULL
    num_reqs: int = 1
    uniform_token_count: int | None = None
    num_active_loras: int = 0
    num_ubatches: int = 1
    max_query_len: int | None = None


@dataclass
class Acceptance:
    path_node_ids: torch.Tensor
    num_sampled: torch.Tensor


class Stats:
    def record_lookup(self, *args):
        pass


class Graph:
    def __init__(self, tensors, qsl, acceptance, positions, **kwargs):
        self.replays = []

    def replay(self, qsl, acceptance):
        self.replays.append(qsl.clone())


def load_subject():
    # Execute the real host-only implementation without importing NPU plugins.
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/tree/qwen_query_graph.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = dict(
        torch=torch,
        dataclass=dataclass,
        fields=fields,
        TreeAcceptance=Acceptance,
        TreeGraphStats=Stats,
        TreeQueryGraph=Graph,
        CUDAGraphMode=Mode,
    )
    exec(compile(tree, str(path), "exec"), namespace)
    return NS(**namespace)


def fixture(q=13, p=17):
    gears = [1, 17, 33, 49, 65]
    descriptors = {n: Descriptor(n, uniform_token_count=n) for n in gears}
    manager = NS(
        _graphs_captured=True,
        tree_prefix_decode=True,
        compilation_config=NS(cudagraph_capture_sizes=gears),
        graphs={desc: object() for desc in descriptors.values()},
    )
    runner = NS(
        cudagraph_manager=manager,
        speculator=NS(tree_shape=NS(target_query_len=65)),
        tree_verification_plan={"graph_requested": True, "graph_descriptor": descriptors[p]},
    )
    batch = NS(
        num_reqs=1,
        num_reqs_after_padding=1,
        has_prefill=False,
        is_prefilling_np=np.array([False]),
        num_tokens=q,
        num_tokens_after_padding=p,
        num_scheduled_tokens=np.array([q]),
        query_start_loc_np=np.array([0, q]),
    )
    return runner, batch


def load_gdn_plan():
    root = Path(__file__).resolve().parents[4] / "vllm_ascend"
    names = {
        "TreeVerifyChunk", "TreeVerifyPlan", "TreeVerifyPlanBuffers", "tree_verify_chunk_count",
        "allocate_tree_verify_plan_buffers", "compute_tree_verify_plan", "_pack_tree_rows_to_tokens",
    }
    body = []
    for relative in ["worker/v2/spec_decode/tree/state_rollback.py", "ops/gdn_attn_builder.py"]:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        body.extend(node for node in tree.body if getattr(node, "name", None) in names)
    namespace = dict(torch=torch, dataclass=dataclass, replace=replace, GDN_RECURRENT_MAX_VERIFY_TOKENS=16,
                     NULL_BLOCK_ID=-1)
    exec(compile(ast.Module(body=body, type_ignores=[]), "gdn_plan_under_test", "exec"), namespace)
    return NS(**namespace)


class QueryGraphContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subject = load_subject()

    def capacity(self, runner, batch):
        return self.subject.qwen_tree_query_graph_capacity(runner, batch, 65)

    def test_existing_full_width_fixed_needs_no_prefix_runtime(self):
        runner, batch = fixture(65, 65)
        runner.tree_verification_plan = None
        runner.cudagraph_manager.tree_prefix_decode = False
        self.assertEqual(self.capacity(runner, batch), 65)

    def test_all_prefix_boundaries_select_exact_physical_gear(self):
        gears = [1, 17, 33, 49, 65]
        for q in [1, 2, 8, 15, 16, 17, 18, 32, 33, 34, 48, 49, 50, 64, 65]:
            p = min(n for n in gears if n >= q)
            with self.subTest(q=q, p=p):
                runner, batch = fixture(q, p)
                self.assertEqual(self.capacity(runner, batch), p)

    def test_short_prefix_requires_current_descriptor_and_runtime(self):
        for change in [
            lambda r, b: r.tree_verification_plan.pop("graph_descriptor"),
            lambda r, b: r.tree_verification_plan.update(graph_requested=False),
            lambda r, b: r.tree_verification_plan.update(mixed_batch=object()),
            lambda r, b: setattr(r.cudagraph_manager, "tree_prefix_decode", False),
            lambda r, b: setattr(r.cudagraph_manager, "_graphs_captured", False),
            lambda r, b: r.cudagraph_manager.graphs.clear(),
            lambda r, b: setattr(b, "num_tokens_after_padding", 33),
            lambda r, b: r.tree_verification_plan.update(graph_descriptor=Descriptor(33, uniform_token_count=33)),
            lambda r, b: r.tree_verification_plan.update(graph_descriptor=Descriptor(17, Mode.NONE, 1, 17)),
            lambda r, b: r.tree_verification_plan.update(graph_descriptor=Descriptor(17, uniform_token_count=17,
                                                                                    num_active_loras=1)),
            lambda r, b: r.tree_verification_plan.update(graph_descriptor=Descriptor(17, uniform_token_count=17,
                                                                                    num_ubatches=2)),
        ]:
            runner, batch = fixture()
            change(runner, batch)
            self.assertIsNone(self.capacity(runner, batch))

    def test_rejects_wrong_cpu_query_layout(self):
        for field, value in [
            ("num_reqs", 2),
            ("num_reqs_after_padding", 2),
            ("has_prefill", True),
            ("is_dummy", True),
            ("is_prefilling_np", np.array([True])),
            ("num_scheduled_tokens", np.array([12])),
            ("query_start_loc_np", np.array([1, 14])),
            ("query_start_loc_np", np.array([0, 12])),
            ("num_tokens", 0),
            ("num_tokens", 66),
        ]:
            runner, batch = fixture()
            setattr(batch, field, value)
            self.assertIsNone(self.capacity(runner, batch), field)

    def test_uncaptured_eager_descriptor_does_not_borrow_captured_full(self):
        runner, batch = fixture(65, 65)
        runner.tree_verification_plan["graph_descriptor"] = Descriptor(65, Mode.NONE, 1, 65)
        self.assertIsNone(self.capacity(runner, batch))
        runner.cudagraph_manager.tree_prefix_decode = False
        runner.tree_verification_plan = {"graph_requested": False}
        self.assertIsNone(self.capacity(runner, batch))

    def test_physical_gear_is_part_of_cache_key_and_failed_gate_resets_it(self):
        runner, batch = fixture()
        graphs = self.subject.QwenTreeQueryGraphs(runner)
        tensors, positions = [torch.empty(65, 2)], torch.arange(65)
        acceptance = Acceptance(torch.tensor([[0, 1]]), torch.tensor([2]))
        for q, p in [(13, 17), (15, 17), (18, 33)]:
            runner.tree_verification_plan["graph_descriptor"] = Descriptor(p, uniform_token_count=p)
            batch.num_tokens, batch.num_tokens_after_padding = q, p
            batch.num_scheduled_tokens[:1] = q
            batch.query_start_loc_np[:] = [0, q]
            self.assertTrue(graphs.eligible(batch))
            graphs.compact_queries(tensors, torch.tensor([0, q]), acceptance, positions)
        self.assertEqual(len(graphs.graphs), 2)
        self.assertEqual(sorted(key[0] for key in graphs.graphs), [17, 33])
        self.assertEqual(sorted(len(entry.graph.replays) for entry in graphs.graphs.values()), [1, 2])
        self.assertFalse(graphs.eligible(None))
        with self.assertRaisesRegex(RuntimeError, "eligible"):
            graphs.compact_queries(tensors, torch.tensor([0, 18]), acceptance, positions)


class GDNPrefixPlanContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subject = load_gdn_plan()

    def test_actual_query_lengths_pack_nodes_and_keep_capture_buffers_stable(self):
        subject = self.subject
        buffers = subject.allocate_tree_verify_plan_buffers(65, 1, torch.device("cpu"))
        addresses = tuple(getattr(buffers, field.name).data_ptr() for field in fields(buffers))
        state = torch.arange(100, 165, dtype=torch.int32).view(1, 65)
        parent = torch.cat([state[:, :1], state[:, :-1]], dim=1)
        for q in [1, 65, 13, 33, 1, 16, 17, 18, 32, 49, 50, 64]:
            p = min(n for n in [1, 17, 33, 49, 65] if n >= q)
            widths = torch.tensor([q])
            packed_state = subject._pack_tree_rows_to_tokens(state, widths, 65)
            packed_parent = subject._pack_tree_rows_to_tokens(parent, widths, 65)
            self.assertTrue(torch.equal(packed_state[0, :q], state[0, :q]))
            self.assertTrue((packed_state[0, q:] == -1).all())
            plan = subject.compute_tree_verify_plan(widths, packed_state, packed_parent, q).copy_into(buffers)
            self.assertEqual(plan.num_chunks, 5)
            visited = []
            for index, chunk in enumerate(plan.chunks):
                count = max(0, min(16, q - index * 16))
                self.assertEqual(chunk.cu_seqlens.tolist(), [0, count])
                self.assertTrue((chunk.token_indices >= 0).all())
                self.assertTrue((chunk.token_indices < q).all())
                self.assertTrue(torch.equal(chunk.state_indices[:count], state[0, index * 16:index * 16 + count]))
                self.assertTrue((chunk.state_indices[count:] == -1).all())
                self.assertTrue((chunk.init_state_indices[count:] == -1).all())
                self.assertTrue((chunk.out_indices[count:] >= q).all())
                self.assertTrue((chunk.out_indices < p + 16).all())
                visited.extend(chunk.out_indices[:count].tolist())
            self.assertEqual(visited, list(range(q)))
            self.assertEqual(addresses, tuple(getattr(buffers, field.name).data_ptr() for field in fields(buffers)))

    def test_no_spec_step_neutralizes_previous_chunk_lengths(self):
        subject = self.subject
        buffers = subject.allocate_tree_verify_plan_buffers(65, 1, torch.device("cpu"))
        state = torch.arange(65, dtype=torch.int32).view(1, 65)
        subject.compute_tree_verify_plan(torch.tensor([65]), state, state, 65).copy_into(buffers)
        self.assertEqual(buffers.cu_seqlens[:, 1].sum().item(), 65)
        buffers.zero_lengths()
        self.assertEqual(buffers.cu_seqlens.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
