"""CPU integration: execute production metadata forwarding and builder branches."""
import ast
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import pytest
import torch

from test_p25_shared_cropped_contract import (
    SOURCE, actual, equivalent_plan, layout, FIELDS,
)

MAMBA = SOURCE / "vllm_ascend/worker/v2/model_states/mamba_hybrid.py"
GDN = SOURCE / "vllm_ascend/ops/gdn_attn_builder.py"


def compile_nodes(path, nodes, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def metadata_type(actual, monkeypatch):
    @dataclass
    class Base:
        num_accepted_tokens: torch.Tensor
        num_decode_draft_tokens_cpu: torch.Tensor

        def get_extra_attn_kwargs(self, builder, rows):
            return {"num_accepted_tokens": self.num_accepted_tokens,
                    "num_decode_draft_tokens_cpu": self.num_decode_draft_tokens_cpu}

    class Builder:
        spec_tree_slots = 33
        spec_conv_width = 3

    package = types.ModuleType("vllm_ascend.ops")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = types.ModuleType("vllm_ascend.ops.gdn_attn_builder")
    module.AscendGDNAttentionMetadataBuilder = Builder
    monkeypatch.setitem(sys.modules, module.__name__, module)
    tree = ast.parse(MAMBA.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if getattr(n, "name", None) in
             ("_pad_batch_rows", "AscendMambaHybridAttnMetadata")]
    assert len(nodes) == 2
    namespace = types.ModuleType("p25_metadata_under_test")
    monkeypatch.setitem(sys.modules, namespace.__name__, namespace)
    namespace.__dict__.update(torch=torch, dataclass=dataclass, MambaHybridAttnMetadata=Base)
    compile_nodes(MAMBA, nodes, namespace.__dict__)
    return namespace.AscendMambaHybridAttnMetadata, Builder


def make_metadata(cls, q, capacity):
    args = layout(q, 2)
    metadata = cls(
        num_accepted_tokens=args["num_accepted_tokens"],
        num_decode_draft_tokens_cpu=torch.tensor([q-1], dtype=torch.int32),
        tree_parents=args["parents"], tree_num_nodes=args["num_nodes"],
        tree_prev_path=args["prev_path_node_ids"],
        tree_prev_num_sampled=args["prev_num_sampled"], tree_depths=args["depths"],
        share_tree_metadata=True, tree_query_width=q, tree_graph_capacity=capacity)
    return metadata


@pytest.mark.parametrize("q,capacity", [(2,9),(9,9),(16,17),(17,17),(24,25),
                                       (25,25),(26,33),(29,33),(32,33),(33,33)])
def test_forward_actual_host_widths_and_reuse_only_within_step(metadata_type, q, capacity):
    cls, builder_type = metadata_type
    metadata = make_metadata(cls, q, capacity)
    left = metadata.get_extra_attn_kwargs(builder_type(), 1)["shared_gdn_tree"]
    right = metadata.get_extra_attn_kwargs(builder_type(), 1)["shared_gdn_tree"]
    assert left is right
    assert (left.query_width, left.graph_capacity, left.slots) == (q, capacity, 33)
    following_step = make_metadata(cls, 33, 33)
    assert following_step.get_extra_attn_kwargs(builder_type(), 1)["shared_gdn_tree"] is not left
    different_conv = builder_type()
    different_conv.spec_conv_width = 4
    assert "shared_gdn_tree" not in metadata.get_extra_attn_kwargs(different_conv, 1)


@pytest.fixture
def builder_paths(actual):
    tree = ast.parse(GDN.read_text(encoding="utf-8"))
    build = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build")
    branches = [n for n in ast.walk(build) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "shared_gdn_tree is not None"]
    assert len(branches) == 1
    # Execute the unchanged production shared/generic branch, not a test rewrite.
    names = ("self", "shared_gdn_tree", "spec_state_indices_tensor", "spec_row_widths",
             "graph_request_count", "num_spec_decodes", "num_prefills", "num_decodes")
    function = ast.FunctionDef(
        name="bind_shared_branch",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in names],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[branches[0], ast.Return(value=ast.Tuple(elts=[
            ast.Name(id=n, ctx=ast.Load()) for n in
            ("spec_init_state_indices", "spec_state_indices_tensor", "spec_conv_window_rows",
             "spec_conv_prefix_src")], ctx=ast.Load()))], decorator_list=[])
    verify = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "_build_tree_verify_plan")
    cls = ast.ClassDef(name="ProductionVerifyBuilder", bases=[], keywords=[],
                       body=[verify], decorator_list=[])
    namespace = compile_nodes(GDN, [function, cls], dict(torch=torch))
    return namespace["bind_shared_branch"], namespace["ProductionVerifyBuilder"]


@pytest.mark.parametrize("q,capacity", [(2,9),(9,9),(16,17),(17,17),(24,25),
                                       (25,25),(26,33),(29,33),(32,33),(33,33)])
def test_builder_binds_and_packs_separate_group_pages(actual, builder_paths, q, capacity):
    reference, shared_type, pack = actual
    bind, _ = builder_paths
    args = layout(q, 2)
    shared = shared_type.create(**args, query_width=q, graph_capacity=capacity)
    widths = torch.tensor([q], dtype=torch.int32)
    builder = types.SimpleNamespace(spec_tree_slots=33, spec_conv_width=3)
    for group in (1, 2):
        pages = torch.arange(33, dtype=torch.int32)[None, :] + group*1000
        init, state, conv, prefix = bind(builder, shared, pages, widths, 1, 1, 0, 0)
        expected_init = pack(reference.compute_tree_init_state_indices(
            pages, args["num_accepted_tokens"], args["parents"], args["num_nodes"]), widths, 33)
        torch.testing.assert_close(init, expected_init)
        torch.testing.assert_close(state, pack(pages, widths, 33))
        assert conv is shared.conv_window_rows and prefix is shared.conv_prefix_src


def test_actual_builder_copies_all_chunks_to_existing_full_buffers(actual, builder_paths):
    reference, shared_type, pack = actual
    _, builder_type = builder_paths
    builder = builder_type()
    builder.spec_tree_slots = 33
    builder.use_full_cuda_graph = True
    builder.spec_verify_plan_buffers = reference.allocate_tree_verify_plan_buffers(33,1,torch.device("cpu"))
    pointers = {field: getattr(builder.spec_verify_plan_buffers, field).data_ptr() for field in FIELDS}
    for q, capacity in ((33,33),(29,33),(24,25),(33,33)):
        widths = torch.tensor([q], dtype=torch.int32)
        args = layout(q,2)
        shared = shared_type.create(**args, query_width=q, graph_capacity=capacity)
        pages = pack(torch.arange(33, dtype=torch.int32)[None,:]+1000,widths,33)
        init = shared.bind_init_state_indices(pages)
        got = builder._build_tree_verify_plan(widths,pages,init,47,shared)
        expected = reference.compute_tree_verify_plan(widths,pages,init,47)
        equivalent_plan(got,expected)
        assert {field: getattr(builder.spec_verify_plan_buffers,field).data_ptr() for field in FIELDS} == pointers


@pytest.mark.parametrize("q,capacity", [(29,25),(33,49),(2,0),(0,9),(34,33)])
def test_invalid_capacity_contract_is_rejected_on_host(actual,q,capacity):
    _, shared_type, _ = actual
    with pytest.raises(ValueError):
        shared_type.create(**layout(33,2),query_width=q,graph_capacity=capacity)
