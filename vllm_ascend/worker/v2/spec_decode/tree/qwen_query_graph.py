# SPDX-License-Identifier: Apache-2.0
"""Accepted query transport graphs for the Qwen GDN tree target."""

from dataclasses import dataclass, fields

import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.metrics import TreeGraphStats


class TreeQueryGraph:
    """Accepted hidden/aux rows and logical positions for the next draft."""

    def __init__(self, tensors, qsl, acceptance, positions, *, pool=None, stream=None, stats=None):
        from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_tree_query_along_path

        self.qsl = qsl.clone()
        self.acceptance = TreeAcceptance.from_path(
            torch.full_like(acceptance.path_node_ids, -1), torch.zeros_like(acceptance.num_sampled)
        )
        self.stats = stats if stats is not None else TreeGraphStats()
        self.replays = 0

        def run():
            compact_tree_query_along_path(tensors, self.qsl, self.acceptance, positions)

        # Disabled request rows make warmup/capture non-mutating.
        with torch.inference_mode():
            run()
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph, pool=pool, stream=stream):
                run()
        self.stats.record_capture("query")
        self.stats.set_workspace_bytes(
            "query",
            id(self),
            self.stats.tensor_bytes(
                [
                    self.qsl,
                    self.acceptance.path_node_ids,
                    self.acceptance.num_sampled,
                    self.acceptance.num_accepted,
                    self.acceptance.endpoint_node,
                    self.acceptance.active,
                ]
            ),
        )

    def replay(self, qsl, acceptance):
        self.qsl.copy_(qsl)
        for field in fields(TreeAcceptance):
            getattr(self.acceptance, field.name).copy_(getattr(acceptance, field.name))
        self.graph.replay()
        self.replays += 1
        self.stats.record_replay("query")


def _tensor_shape(tensor):
    return tuple(tensor.shape), tensor.dtype, tensor.device


def _tensor_view(tensor):
    return tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device


def qwen_tree_query_graph_capacity(runner, input_batch, query_len):
    """Return the captured physical gear for one real CPU-described query row.

    This is a transport capability check, not proof that target GDN capture is
    correct. Prefix rows require the *current* execution descriptor published
    by the runner. A configured FULL mode or an unrelated captured graph is
    insufficient. Missing proof keeps accepted-query transport eager.
    """
    if input_batch is None or input_batch.num_reqs != 1 or input_batch.has_prefill:
        return None
    if getattr(input_batch, "is_dummy", False):
        return None
    if input_batch.is_prefilling_np[:1].any() or input_batch.num_reqs_after_padding != 1:
        return None
    q, capacity = input_batch.num_tokens, input_batch.num_tokens_after_padding
    if not 1 <= q <= capacity <= query_len:
        return None
    scheduled = input_batch.num_scheduled_tokens[:1]
    qsl = input_batch.query_start_loc_np
    if scheduled.shape != (1,) or not (scheduled == q).all():
        return None
    if qsl.shape != (2,) or qsl[0] != 0 or qsl[1] != q:
        return None
    manager = runner.cudagraph_manager
    if manager is None or not manager._graphs_captured:
        return None
    prefix_runtime = False
    config = getattr(runner, "vllm_config", None)
    if config is not None:
        from vllm_ascend.worker.v2.spec_decode.tree.prefix_graph import (
            prefix_graph_configured,
            prefix_graph_runtime_proven,
        )

        prefix_configured = prefix_graph_configured(config)
        hybrid = getattr(config.model_config.hf_config, "model_type", None) == "qwen3_5"
        if prefix_configured:
            tp_group = None
            if hybrid:
                from vllm.distributed import get_tp_group

                tp_group = get_tp_group()
            prefix_runtime = prefix_graph_runtime_proven(config, runner.speculator, tp_group)
        if hybrid and (prefix_configured or getattr(manager, "tree_prefix_decode", False) or q != query_len):
            if not prefix_runtime:
                return None

    def matches(desc):
        return (
            desc.cg_mode == CUDAGraphMode.FULL
            and desc.num_reqs == 1
            and desc.num_tokens == capacity
            and (
                desc.uniform_token_count == capacity
                or (
                    desc.uniform_token_count is None
                    and prefix_runtime
                    and getattr(manager, "tree_prefix_decode", False) is True
                    and getattr(desc, "max_query_len", None) == capacity
                )
            )
            and getattr(desc, "num_active_loras", 0) == 0
            and getattr(desc, "num_ubatches", 1) == 1
        )

    plan = getattr(runner, "tree_verification_plan", None)
    descriptor = None if plan is None else plan.get("graph_descriptor")
    if plan is not None and plan.get("mixed_batch") is not None:
        return None
    if plan is not None and "graph_requested" in plan and plan["graph_requested"] is not True:
        return None
    if descriptor is not None and (
        plan.get("graph_requested") is not True or descriptor not in manager.graphs or not matches(descriptor)
    ):
        return None
    if getattr(manager, "tree_prefix_decode", False) or q != query_len:
        if (
            getattr(manager, "tree_prefix_decode", False) is not True
            or descriptor is None
            or plan.get("graph_requested") is not True
        ):
            return None
        compilation = getattr(manager, "compilation_config", None)
        gears = getattr(compilation, "cudagraph_capture_sizes", ())
        if not gears or any(type(n) is not int or not 1 <= n <= query_len for n in gears):
            return None
        candidates = [n for n in gears if n >= q]
        if not candidates or capacity != min(candidates):
            return None
    elif q != capacity:
        return None
    return capacity if any(matches(desc) for desc in manager.graphs) else None


@dataclass
class _QueryGraphEntry:
    graph: TreeQueryGraph
    # The recorded graph refers to these actual target outputs and positions.
    # Retain their storage even after the caller replaces its Python views.
    tensors: tuple[torch.Tensor, ...]
    positions: torch.Tensor


class QwenTreeQueryGraphs:
    """Reuse one private query graph for each stable target output layout."""

    def __init__(self, runner, *, stats=None):
        self.runner = runner
        self.stats = stats if stats is not None else TreeGraphStats()
        self.query_len = runner.speculator.tree_shape.target_query_len
        self.graphs = {}
        self._capacity = None

    def eligible(self, input_batch):
        # Clear the previous gear even when this batch is eager/prefill.
        self._capacity = qwen_tree_query_graph_capacity(self.runner, input_batch, self.query_len)
        return self._capacity is not None

    def compact_queries(self, tensors, qsl, acceptance, positions):
        if self._capacity is None:
            raise RuntimeError("Qwen tree query transport must be eligible for the current batch")
        key = (
            self._capacity,
            tuple(_tensor_view(tensor) for tensor in tensors),
            _tensor_view(positions),
            _tensor_shape(qsl),
            tuple(_tensor_shape(getattr(acceptance, field.name)) for field in fields(TreeAcceptance)),
        )
        entry = self.graphs.get(key)
        self.stats.record_lookup("query", entry is not None)
        if entry is None:
            # Capture failures propagate. The private acceptance starts inactive,
            # so warmup and capture cannot compact the live target outputs twice.
            graph = TreeQueryGraph(tensors, qsl, acceptance, positions, stats=self.stats)
            entry = _QueryGraphEntry(graph, tuple(tensors), positions)
            self.graphs[key] = entry
        entry.graph.replay(qsl, acceptance)
