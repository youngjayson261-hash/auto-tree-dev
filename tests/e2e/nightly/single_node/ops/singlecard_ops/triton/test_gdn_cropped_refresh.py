# SPDX-License-Identifier: Apache-2.0
"""Numeric equivalence of the fused cropped-row GDN refresh.

One ``GDNCroppedRefreshBatch.finish`` launch must leave every registered
group's fourteen destinations bit-identical to the per-group refresh of
``AscendGDNHostMetadataBuilder._refresh_cropped_tree_metadata``: the masked
state/init page tables, the five verify-plan buffers, the packed token order,
the query-span scalars, the accepted count and the two short-conv tables.
Sentinel-filled buffers catch writes past the per-group footprint.
"""

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.tree.gdn_tree_pages import (
    GDNCroppedRefreshBatch,
    GDNCroppedRefreshPointers,
)
from vllm_ascend.worker.v2.spec_decode.tree.shared_metadata import SharedGDNTreeMetadata
from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import allocate_tree_verify_plan_buffers

SENTINEL = -7


class _FakeBuilder:
    """Just the attributes ``GDNCroppedRefreshBatch`` reads off a builder."""

    def __init__(self, buffers, token_indx, qsl, asl):
        self.spec_verify_plan_buffers = buffers
        self.spec_token_indx = token_indx
        self.spec_query_start_loc = qsl
        self.spec_actual_seq_lengths = asl
        self.finalized = 0

    def _finalize_metadata(self, metadata):
        self.finalized += 1
        return metadata


class _FakeSpec:
    def __init__(self, init, conv_rows, conv_prefix):
        self.verify_plan = object()  # only None-ness is inspected
        self.init_state_indices = init
        self.conv_window_rows = conv_rows
        self.conv_prefix_src = conv_prefix


class _FakeMetadata:
    def __init__(self, state, num_accepted, spec):
        self.spec_state_indices_tensor = state
        self.num_accepted_tokens = num_accepted
        self.spec_decode_metadata = spec


class _Group:
    """One fake group's stable capture buffers (max_bs = 2 rows)."""

    def __init__(self, slots, conv_width):
        device = torch.device("npu")
        self.state_buf = torch.full((2, slots), SENTINEL, dtype=torch.int32, device=device)
        self.init_buf = torch.full((2, slots), SENTINEL, dtype=torch.int32, device=device)
        self.buffers = allocate_tree_verify_plan_buffers(slots, 2, device, width=slots)
        for t in (self.buffers.token_indices, self.buffers.out_indices):
            t.fill_(SENTINEL)
        for t in (self.buffers.state_indices, self.buffers.init_state_indices, self.buffers.cu_seqlens):
            t.fill_(SENTINEL)
        self.token_indx = torch.full((2 * slots,), SENTINEL, dtype=torch.int32, device=device)
        self.qsl = torch.full((3,), SENTINEL, dtype=torch.int32, device=device)
        self.qsl[0] = 0  # the pinned span base the refresh never touches
        self.asl = torch.full((3,), SENTINEL, dtype=torch.int32, device=device)
        self.num_accepted = torch.full((2,), SENTINEL, dtype=torch.int32, device=device)
        self.conv_rows_buf = torch.full((2 * slots, conv_width), SENTINEL, dtype=torch.int32, device=device)
        self.conv_prefix_buf = torch.full((2, conv_width - 1), SENTINEL, dtype=torch.int32, device=device)
        self.builder = _FakeBuilder(self.buffers, self.token_indx, self.qsl, self.asl)
        self.metadata = _FakeMetadata(
            self.state_buf[:1],
            self.num_accepted[:1],
            _FakeSpec(self.init_buf[:1], self.conv_rows_buf[:slots], self.conv_prefix_buf[:1]),
        )


def _make_tree(slots, conv_width, query_width):
    parents = torch.empty((1, slots - 1), dtype=torch.int32)
    depths = torch.zeros((1, slots), dtype=torch.int32)
    for node in range(1, slots):
        parent = (node - 1) // 2
        parents[0, node - 1] = parent
        depths[0, node] = depths[0, parent] + 1
    device = torch.device("npu")
    return SharedGDNTreeMetadata.create(
        parents.to(device),
        torch.tensor([slots - 1], dtype=torch.int32, device=device),
        depths[:, 1:].to(device),
        torch.tensor([3], dtype=torch.int32, device=device),
        torch.arange(max(slots - 1, 16), dtype=torch.int32, device=device)[None, :],
        torch.tensor([4], dtype=torch.int32, device=device),
        conv_width,
        query_width=query_width,
        graph_capacity=slots,
    )


def _reference_refresh(tree, group, pages, accepted):
    """The per-group refresh writes of _refresh_cropped_tree_metadata."""
    qw = tree.query_width
    state = tree.bind_state_indices(pages)
    init = tree.bind_init_state_indices(pages)
    group.state_buf[:1].copy_(state)
    group.init_buf[:1].copy_(init)
    tree.bind_and_copy_into(group.state_buf[:1], group.init_buf[:1], qw, group.buffers)
    group.token_indx[:qw].copy_(torch.arange(qw, dtype=torch.int32, device=pages.device))
    group.qsl[1] = qw
    group.asl[1] = qw
    group.num_accepted[:1].copy_(accepted)
    group.conv_rows_buf[: tree.slots].copy_(tree.conv_window_rows)
    group.conv_prefix_buf[:1].copy_(tree.conv_prefix_src)


def _assert_group_equal(fused, reference):
    torch.testing.assert_close(fused.state_buf, reference.state_buf)
    torch.testing.assert_close(fused.init_buf, reference.init_buf)
    for name in ("token_indices", "state_indices", "init_state_indices", "out_indices", "cu_seqlens"):
        torch.testing.assert_close(getattr(fused.buffers, name), getattr(reference.buffers, name))
    torch.testing.assert_close(fused.token_indx, reference.token_indx)
    torch.testing.assert_close(fused.qsl, reference.qsl)
    torch.testing.assert_close(fused.asl, reference.asl)
    torch.testing.assert_close(fused.num_accepted, reference.num_accepted)
    torch.testing.assert_close(fused.conv_rows_buf, reference.conv_rows_buf)
    torch.testing.assert_close(fused.conv_prefix_buf, reference.conv_prefix_buf)


def _random_pages(slots, groups):
    device = torch.device("npu")
    return [torch.randint(1, 10000, (1, slots + 3), dtype=torch.int32, device=device) for _ in range(groups)]


def run_case(slots, conv_width, query_width, groups=4):
    tree = _make_tree(slots, conv_width, query_width)
    accepted = torch.tensor([3], dtype=torch.int32, device="npu")
    pages_list = _random_pages(slots, groups)
    fused = [_Group(slots, conv_width) for _ in range(groups)]
    reference = [_Group(slots, conv_width) for _ in range(groups)]

    batch = GDNCroppedRefreshBatch(GDNCroppedRefreshPointers())
    for group, pages in zip(fused, pages_list):
        assert batch.add(tree, pages[:, :slots], group.builder, group.metadata)
    batch.finish(tree, accepted)
    assert all(group.builder.finalized == 1 for group in fused)

    for group, pages in zip(reference, pages_list):
        _reference_refresh(tree, group, pages[:, :slots], accepted)
    for fused_group, reference_group in zip(fused, reference):
        _assert_group_equal(fused_group, reference_group)


def test_cropped_refresh_matches_per_group_widths():
    for query_width in (2, 3, 17, 31, 32, 63):
        run_case(slots=64, conv_width=4, query_width=query_width)


def test_cropped_refresh_matches_per_group_shapes():
    run_case(slots=33, conv_width=3, query_width=9)
    run_case(slots=65, conv_width=5, query_width=64)
    run_case(slots=64, conv_width=2, query_width=2)


def test_cropped_refresh_second_step_reuses_pointer_table():
    slots, conv_width, qw = 64, 4, 29
    tree = _make_tree(slots, conv_width, qw)
    accepted = torch.tensor([2], dtype=torch.int32, device="npu")
    pages_list = _random_pages(slots, 3)
    fused = [_Group(slots, conv_width) for _ in range(3)]
    reference = [_Group(slots, conv_width) for _ in range(3)]
    pointers = GDNCroppedRefreshPointers()
    for step in range(2):  # fresh batch per step, stable pointer store
        if step:
            for pages in pages_list:
                pages.add_(1)  # same storage, new values: signature must NOT rebuild
            accepted = accepted.add(1)
        batch = GDNCroppedRefreshBatch(pointers)
        for group, pages in zip(fused, pages_list):
            assert batch.add(tree, pages[:, :slots], group.builder, group.metadata)
        batch.finish(tree, accepted)
    for group, pages in zip(reference, pages_list):
        _reference_refresh(tree, group, pages[:, :slots], accepted)
    for fused_group, reference_group in zip(fused, reference):
        _assert_group_equal(fused_group, reference_group)


def test_cropped_refresh_non_int32_accepted_falls_back():
    slots, conv_width, qw = 64, 4, 11
    tree = _make_tree(slots, conv_width, qw)
    accepted64 = torch.tensor([5], dtype=torch.int64, device="npu")
    accepted32 = accepted64.to(torch.int32)
    fused = _Group(slots, conv_width)
    reference = _Group(slots, conv_width)
    pages = _random_pages(slots, 1)[0]
    batch = GDNCroppedRefreshBatch(GDNCroppedRefreshPointers())
    assert batch.add(tree, pages[:, :slots], fused.builder, fused.metadata)
    batch.finish(tree, accepted64)
    _reference_refresh(tree, reference, pages[:, :slots], accepted32)
    _assert_group_equal(fused, reference)


def test_cropped_refresh_rejects_mismatched_group():
    slots, conv_width, qw = 64, 4, 11
    tree = _make_tree(slots, conv_width, qw)
    group = _Group(slots, conv_width)
    pages = _random_pages(slots, 1)[0]
    batch = GDNCroppedRefreshBatch(GDNCroppedRefreshPointers())
    # Wrong page dtype -> per-group fallback keeps ownership of this group.
    assert not batch.add(tree, pages[:, :slots].to(torch.int64), group.builder, group.metadata)
    # Wrong conv table shape -> fallback.
    group.metadata.spec_decode_metadata.conv_window_rows = group.conv_rows_buf[: slots - 1]
    assert not batch.add(tree, pages[:, :slots], group.builder, group.metadata)
    # A rejected group never reaches the finalizer list.
    batch.finish(tree, torch.tensor([1], dtype=torch.int32, device="npu"))
    assert group.builder.finalized == 0
