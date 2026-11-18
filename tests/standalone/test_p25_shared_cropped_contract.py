"""CPU contracts for the current shared GDN production implementation.

P25_TEST_SOURCE may select an isolated snapshot instead of the current checkout.
These tests execute the real shared class, rollback helpers and table packing.
"""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest
import torch

SOURCE = Path(os.environ.get("P25_TEST_SOURCE", str(Path(__file__).resolve().parents[2])))
TREE = SOURCE / "vllm_ascend/worker/v2/spec_decode/tree"
FIELDS = ("token_indices", "state_indices", "init_state_indices", "out_indices", "cu_seqlens")
WIDTHS = (2, 9, 16, 17, 24, 25, 26, 29, 32, 33)


@pytest.fixture
def actual(monkeypatch):
    prefix = "vllm_ascend.worker.v2.spec_decode.tree"
    for size in range(1, len(prefix.split(".")) + 1):
        name = ".".join(prefix.split(".")[:size])
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    # The installed vLLM uses NULL_BLOCK_ID=0 and PAD_SLOT_ID=-1.
    for size in range(1, 6):
        name = ".".join("vllm.v1.attention.backends.utils".split(".")[:size])
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    utils = sys.modules["vllm.v1.attention.backends.utils"]
    utils.NULL_BLOCK_ID = 0
    utils.PAD_SLOT_ID = -1
    modules = []
    for stem in ("state_rollback", "shared_metadata"):
        name = prefix + "." + stem
        spec = importlib.util.spec_from_file_location(name, TREE / (stem + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    path = SOURCE / "vllm_ascend/ops/gdn_attn_builder.py"
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "_pack_tree_rows_to_tokens")
    namespace = dict(torch=torch, NULL_BLOCK_ID=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return modules[0], modules[1].SharedGDNTreeMetadata, namespace["_pack_tree_rows_to_tokens"]


def layout(q, branching):
    parents = torch.empty((1, 32), dtype=torch.int32)
    depths = torch.zeros((1, 33), dtype=torch.int32)
    for node in range(1, 33):
        parent = node - 1 if branching == 1 else (node - 1) // branching
        parents[0, node - 1] = parent
        depths[0, node] = depths[0, parent] + 1
    return dict(parents=parents, num_nodes=torch.tensor([q - 1], dtype=torch.int32),
                depths=depths[:, 1:], num_accepted_tokens=torch.tensor([3], dtype=torch.int32),
                prev_path_node_ids=torch.arange(16, dtype=torch.int32)[None, :],
                prev_num_sampled=torch.tensor([4], dtype=torch.int32), conv_width=3)


def equivalent_plan(got, expected):
    assert (got.rows, got.slots, got.width, got.num_chunks) == (1, 33, 16, 3)
    for left, right in zip(got.chunks, expected.chunks, strict=True):
        for field in FIELDS:
            torch.testing.assert_close(getattr(left, field), getattr(right, field))


def compare_group(reference, pack, shared, args, q, group):
    widths = torch.tensor([q], dtype=torch.int32)
    pages = torch.arange(33, dtype=torch.int32)[None, :] + 1000 * group
    expected_init = pack(reference.compute_tree_init_state_indices(
        pages, args["num_accepted_tokens"], args["parents"], args["num_nodes"]), widths, 33)
    got_init = shared.bind_init_state_indices(pages)
    torch.testing.assert_close(got_init, expected_init)
    assert bool((got_init[:, q:] == 0).all())
    packed_pages = pack(pages, widths, 33)
    # This independent scratch base deliberately differs from Q, C and S.
    scratch_base = 47
    expected = reference.compute_tree_verify_plan(widths, packed_pages, expected_init, scratch_base)
    got = shared.bind_verify_plan(packed_pages, got_init, scratch_base)
    equivalent_plan(got, expected)
    expected_conv = reference.compute_tree_conv_rows(
        args["parents"], args["num_nodes"], 3, 2, row_widths=widths, tree_depths=args["depths"])
    torch.testing.assert_close(shared.conv_window_rows[:q], expected_conv[:q])
    expected_prefix = reference.compute_tree_prefix_source(args["prev_path_node_ids"], args["prev_num_sampled"], 3)
    torch.testing.assert_close(shared.conv_prefix_src, expected_prefix)
    return got


@pytest.mark.parametrize("branching", [1, 2, 3])
def test_existing_complete_tree_api_remains_equivalent(actual, branching):
    reference, shared_type, pack = actual
    args = layout(33, branching)
    shared = shared_type.create(**args)
    compare_group(reference, pack, shared, args, 33, 1)


@pytest.mark.parametrize("branching", [1, 2, 3])
@pytest.mark.parametrize("q", WIDTHS)
def test_cropped_logical_metadata_matches_generic_for_separate_pages(actual, q, branching):
    reference, shared_type, pack = actual
    args = layout(q, branching)
    shared = shared_type.create(**args, query_width=q)
    first = compare_group(reference, pack, shared, args, q, 1)
    saved = [chunk.state_indices.clone() for chunk in first.chunks]
    second = compare_group(reference, pack, shared, args, q, 2)
    for index, (a, b) in enumerate(zip(first.chunks, second.chunks, strict=True)):
        assert a.token_indices.data_ptr() == b.token_indices.data_ptr(), "Logical geometry should be shared across cache groups"
        assert a.state_indices.data_ptr() != b.state_indices.data_ptr(), "Physical state pages must remain group-specific"
        torch.testing.assert_close(a.state_indices, saved[index])


@pytest.mark.parametrize("branching", [1, 2, 3])
def test_full_crop_full_copies_clear_old_values_without_changing_addresses(actual, branching):
    reference, shared_type, pack = actual
    buffers = reference.allocate_tree_verify_plan_buffers(33, 1, torch.device("cpu"))
    pointers = {field: getattr(buffers, field).data_ptr() for field in FIELDS}
    for q in (33, 29, 24, 33):
        args = layout(q, branching)
        shared = shared_type.create(**args, query_width=q)
        plan = compare_group(reference, pack, shared, args, q, 1)
        copied = plan.copy_into(buffers)
        equivalent_plan(copied, plan)
        assert {field: getattr(buffers, field).data_ptr() for field in FIELDS} == pointers
        assert [int(c.cu_seqlens[-1]) for c in copied.chunks] == [min(16, max(0, q - start)) for start in (0, 16, 32)]


@pytest.mark.parametrize("invalid_width", [0, 34])
def test_query_width_bounds_are_host_validated(actual, invalid_width):
    _, shared_type, _ = actual
    with pytest.raises(ValueError):
        shared_type.create(**layout(33, 2), query_width=invalid_width)
