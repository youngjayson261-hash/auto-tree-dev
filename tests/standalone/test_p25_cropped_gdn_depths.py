"""CPU contracts for P25-A, executing extracted production builder methods."""

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2]
GDN = SOURCE / "vllm_ascend/ops/gdn_attn_builder.py"
MAMBA = SOURCE / "vllm_ascend/worker/v2/model_states/mamba_hybrid.py"
ROLLBACK = SOURCE / "vllm_ascend/worker/v2/spec_decode/tree/state_rollback.py"


def extract_method(path, name, base=object, extra=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(methods) == 1
    cls = ast.ClassDef(name="ProductionMethod", bases=[ast.Name(id="Base", ctx=ast.Load())],
                       keywords=[], body=[copy.deepcopy(methods[0])], decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    namespace = dict(torch=torch, Base=base, __name__="p25_extracted_production", **(extra or {}))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["ProductionMethod"]


@pytest.fixture
def reference(monkeypatch):
    name = "vllm_ascend.worker.v2.spec_decode.tree.state_rollback"
    for size in range(1, len(name.split("."))):
        parent = ".".join(name.split(".")[:size])
        package = types.ModuleType(parent)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, parent, package)
    spec = importlib.util.spec_from_file_location(name, ROLLBACK)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def tree_batch():
    parents = torch.empty((3, 32), dtype=torch.int32)
    depths = torch.zeros((3, 33), dtype=torch.int32)
    for row in range(3):
        for node in range(1, 33):
            parent = node - 1 if row == 0 else (node - 1) // (row + 1)
            parents[row, node - 1] = parent
            depths[row, node] = depths[row, parent] + 1
    return parents, depths[:, 1:]


def builder_inputs(q):
    parents, depths = tree_batch()
    widths_all = torch.tensor([q, 33, min(q + 1, 33)], dtype=torch.int32)
    nodes = widths_all - 1
    selected = torch.tensor([2, 0], dtype=torch.int64)
    paths = torch.arange(16, dtype=torch.int32).repeat(3, 1)
    sampled = torch.tensor([1, 2, 3], dtype=torch.int32)
    widths = widths_all.index_select(0, selected)
    args = (torch.zeros(2, dtype=torch.int32), parents, nodes, selected, paths, sampled)
    return depths, args, widths


@pytest.mark.parametrize("q", [2, 9, 16, 17, 24, 25, 26, 29, 32, 33])
def test_existing_depths_avoid_reconstruction_and_keep_permuted_rows(reference, monkeypatch, q):
    depths, args, widths = builder_inputs(q)
    _, parents, nodes, selected, paths, sampled = args
    expected_windows = reference.compute_tree_conv_rows(
        parents.index_select(0, selected), nodes.index_select(0, selected), 3, 4, row_widths=widths)
    expected_prefix = reference.compute_tree_prefix_source(
        paths.index_select(0, selected), sampled.index_select(0, selected), 3)
    reconstruct = Mock(wraps=reference.tree_node_parents_depths)
    compute = Mock(wraps=reference.compute_tree_conv_rows)
    monkeypatch.setattr(reference, "tree_node_parents_depths", reconstruct)
    monkeypatch.setattr(reference, "compute_tree_conv_rows", compute)
    builder = extract_method(GDN, "_build_tree_conv_rows")()
    builder.spec_tree_slots = 33
    builder.spec_conv_window_rows = torch.empty((2, 33, 3), dtype=torch.int32)
    builder.spec_conv_width = 3
    windows, prefix = builder._build_tree_conv_rows(*args, row_widths=widths, tree_depths=depths)
    assert reconstruct.call_count == 0, "Current-step tree depths must be reused, not reconstructed per group"
    torch.testing.assert_close(compute.call_args.kwargs["tree_depths"], depths.index_select(0, selected))
    torch.testing.assert_close(windows, expected_windows)
    torch.testing.assert_close(prefix, expected_prefix)


def test_missing_depths_preserve_original_reconstruction(reference, monkeypatch):
    _, args, widths = builder_inputs(24)
    builder = extract_method(GDN, "_build_tree_conv_rows")()
    builder.spec_tree_slots = 33
    builder.spec_conv_window_rows = torch.empty((2, 33, 3), dtype=torch.int32)
    builder.spec_conv_width = 3
    reconstruct = Mock(wraps=reference.tree_node_parents_depths)
    monkeypatch.setattr(reference, "tree_node_parents_depths", reconstruct)
    windows, prefix = builder._build_tree_conv_rows(*args, row_widths=widths)
    assert reconstruct.call_count > 0
    assert windows is not None and prefix is not None


@pytest.fixture
def metadata_case(monkeypatch):
    class Base:
        def get_extra_attn_kwargs(self, builder, count):
            return {"num_accepted_tokens": torch.zeros(count, dtype=torch.int32)}

    class GDNBuilder:
        pass

    module = types.ModuleType("vllm_ascend.ops.gdn_attn_builder")
    module.AscendGDNAttentionMetadataBuilder = GDNBuilder
    monkeypatch.setitem(sys.modules, module.__name__, module)
    tree = ast.parse(MAMBA.read_text(encoding="utf-8"))
    pad = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pad_batch_rows")
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pad], type_ignores=[])), str(MAMBA), "exec"), namespace)
    cls = extract_method(MAMBA, "get_extra_attn_kwargs", Base, {"_pad_batch_rows": namespace["_pad_batch_rows"]})
    metadata = cls()
    parents, depths = tree_batch()
    metadata.tree_parents = parents[:2]
    metadata.tree_num_nodes = torch.tensor([29, 24], dtype=torch.int32)
    metadata.tree_depths = depths[:2]
    metadata.tree_prev_path = None
    metadata.tree_prev_num_sampled = None
    metadata.share_tree_metadata = False
    return metadata, GDNBuilder()


def test_metadata_depths_follow_the_same_padded_request_rows(metadata_case):
    metadata, builder = metadata_case
    kwargs = metadata.get_extra_attn_kwargs(builder, 3)
    assert "tree_depths" in kwargs, "The GDN builder must receive current-step depths"
    assert kwargs["tree_depths"].shape == (3, 32)
    torch.testing.assert_close(kwargs["tree_depths"][:2], metadata.tree_depths)
    assert torch.count_nonzero(kwargs["tree_depths"][2]).item() == 0
    assert kwargs["tree_parents"].shape[0] == kwargs["tree_num_nodes"].shape[0] == 3


def test_metadata_missing_depths_preserves_fallback(metadata_case):
    metadata, builder = metadata_case
    metadata.tree_depths = None
    assert "tree_depths" not in metadata.get_extra_attn_kwargs(builder, 3)


def test_non_gdn_metadata_keeps_existing_behavior(metadata_case):
    metadata, _ = metadata_case
    kwargs = metadata.get_extra_attn_kwargs(object(), 3)
    assert "tree_depths" not in kwargs
    assert "tree_parents" not in kwargs


def test_normal_and_host_builder_depth_keyword_contract():
    tree = ast.parse(GDN.read_text(encoding="utf-8"))
    names = ("AscendGDNAttentionMetadataBuilder", "AscendGDNHostMetadataBuilder")
    for name in names:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build")
        assert "tree_depths" in {a.arg for a in method.args.args + method.args.kwonlyargs}, name
        if name == "AscendGDNHostMetadataBuilder":
            calls = [n for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "build"]
            assert any("tree_depths" in {k.arg for k in call.keywords} for call in calls)
