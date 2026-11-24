# SPDX-License-Identifier: Apache-2.0
"""Refresh the private recurrent-state pages of compatible GDN groups in one launch.

Also hosts the cropped-row counterpart: ``GDNCroppedRefreshBatch`` rewrites every
step-varying field of all cropped verification groups with one kernel launch.
"""

from typing import TYPE_CHECKING

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.envs import _bool_env

if TYPE_CHECKING:
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBuilder
    from vllm_ascend.worker.v2.spec_decode.tree.shared_metadata import SharedGDNTreeMetadata


@triton.jit
def _refresh_tree_pages(pointers, columns, SLOTS: tl.constexpr, BLOCK: tl.constexpr):
    # Each program owns both outputs of one group, including their partial tail.
    row = pointers + 3 * tl.program_id(0)
    src = tl.load(row).to(tl.pointer_type(tl.int32))
    dst = tl.load(row + 1).to(tl.pointer_type(tl.int32))
    init = tl.load(row + 2).to(tl.pointer_type(tl.int32))
    i = tl.arange(0, BLOCK)
    parent = tl.load(columns + i, mask=i < SLOTS, other=0).to(tl.int32)
    tl.store(dst + i, tl.load(src + i, mask=i < SLOTS, other=0), mask=i < SLOTS)
    tl.store(init + i, tl.load(src + parent, mask=i < SLOTS, other=0), mask=i < SLOTS)


class GDNTreePagePointers:
    """Model-owned address table. Page values and logical parents change every step."""

    def __init__(self):
        self.signature = ()
        self.pointers = None
        self.buffers = ()

    def refresh(self, columns, rows):
        signature = tuple((src.data_ptr(), dst.data_ptr(), init.data_ptr()) for src, dst, init in rows)
        if self.signature != signature:
            outputs = [ptr for _, dst, init in signature for ptr in (dst, init)]
            if len(set(outputs)) != len(outputs) or set(outputs).intersection(src for src, _, _ in signature):
                raise ValueError("GDN groups require private output page tables")
            host = torch.tensor(signature, device="cpu", dtype=torch.int64, pin_memory=True)
            self.pointers = host.to(columns.device, non_blocking=True)
            self.buffers = tuple(rows)  # Keep the allocations referenced by the device pointer table alive.
            self.signature = signature
        slots = columns.numel()
        _refresh_tree_pages[(len(rows),)](
            self.pointers, columns, slots, triton.next_power_of_2(slots), multibuffer=False, unit_flag=False
        )


class GDNTreePageBatch:
    """One complete verification row; finish page writes before graph replay."""

    # Only the 32-node tree plus root has been validated for this fast path.
    SLOTS = 33

    def __init__(self, pointers: GDNTreePagePointers):
        self.pointers = pointers
        self.rows = []
        self.finalizers = []

    def add(self, columns, src, builder: "AscendGDNHostMetadataBuilder", metadata: "GDNAttentionMetadata") -> bool:
        dst = metadata.spec_state_indices_tensor
        init = metadata.spec_decode_metadata.init_state_indices
        if (
            src.device.type != "npu"
            or not (src.device == columns.device == dst.device == init.device)
            or not (src.dtype == dst.dtype == init.dtype == torch.int32)
            or columns.dtype not in (torch.int64, torch.int32)
            or not (columns.shape == dst.shape == init.shape == (1, self.SLOTS))
            or src.ndim != 2
            or src.shape[0] != 1
            or src.shape[1] < self.SLOTS
            or not (src.stride(1) == columns.stride(1) == dst.stride(1) == init.stride(1) == 1)
        ):
            return False
        self.rows.append((src, dst, init))
        self.finalizers.append((builder, metadata))
        return True

    def finish(self, columns):
        if self.rows:
            self.pointers.refresh(columns, self.rows)
            # Convolution finalization can read the newly written page indices.
            for builder, metadata in self.finalizers:
                builder._finalize_metadata(metadata)
            self.rows.clear()
            self.finalizers.clear()


def cropped_refresh_enabled() -> bool:
    """``VLLM_ASCEND_GDN_CROPPED_BATCH=0`` restores the per-group refresh."""
    return _bool_env("VLLM_ASCEND_GDN_CROPPED_BATCH", True)


@triton.jit
def _cropped_tree_refresh(
    pointers,  # [G, 14] int64 source/destination address table
    columns,  # [S] int64 logical parent column of every slot
    invalid,  # [S] int8 view of the cropped row's inactive-slot mask
    owned,  # [S] int8 view of the verify geometry's ownership mask
    plan_tokens,  # [S] int64 packed token column of every cell (0 where unowned)
    plan_out_values,  # [S] int64 output row of every cell
    cu_values,  # [2] int32, (0, query_width)
    conv_rows_src,  # [S * CONV_W] int32 short-conv tap table of the tree
    conv_prefix_values,  # [CONV_W - 1] int32 committed-history taps
    accepted,  # [1] int32 accepted draft count of the step
    query_width,  # int32 cropped width
    null_block_id,  # int32 inactive-slot filler
    SLOTS: tl.constexpr,
    CONV_W: tl.constexpr,
    BLOCK: tl.constexpr,
    CONV_BLOCK: tl.constexpr,
    HAS_ACCEPTED: tl.constexpr,
):
    row = pointers + 14 * tl.program_id(0)
    src = tl.load(row).to(tl.pointer_type(tl.int32))
    dst_state = tl.load(row + 1).to(tl.pointer_type(tl.int32))
    dst_init = tl.load(row + 2).to(tl.pointer_type(tl.int32))
    plan_state = tl.load(row + 3).to(tl.pointer_type(tl.int32))
    plan_init = tl.load(row + 4).to(tl.pointer_type(tl.int32))
    plan_tok = tl.load(row + 5).to(tl.pointer_type(tl.int64))
    plan_out = tl.load(row + 6).to(tl.pointer_type(tl.int64))
    plan_cu = tl.load(row + 7).to(tl.pointer_type(tl.int32))
    token_indx = tl.load(row + 8).to(tl.pointer_type(tl.int32))
    qsl = tl.load(row + 9).to(tl.pointer_type(tl.int32))
    asl = tl.load(row + 10).to(tl.pointer_type(tl.int32))
    num_accepted = tl.load(row + 11).to(tl.pointer_type(tl.int32))
    conv_rows = tl.load(row + 12).to(tl.pointer_type(tl.int32))
    conv_prefix = tl.load(row + 13).to(tl.pointer_type(tl.int32))

    i = tl.arange(0, BLOCK)
    mask = i < SLOTS
    pages = tl.load(src + i, mask=mask, other=0)
    inactive = tl.load(invalid + i, mask=mask, other=1) != 0
    # bind_state_indices: inactive slots hold NULL_BLOCK_ID.
    tl.store(dst_state + i, tl.where(inactive, null_block_id, pages), mask=mask)
    parent = tl.load(columns + i, mask=mask, other=0).to(tl.int32)
    init = tl.load(src + parent, mask=mask, other=0)
    # bind_init_state_indices: the same mask over the gathered parent pages.
    tl.store(dst_init + i, tl.where(inactive, null_block_id, init), mask=mask)

    # _bind_stacked over the masked tables: an owned cell's token column is an
    # active slot (owned token columns precede query_width), so gathering the
    # raw pages yields the same value as gathering the masked tables.
    owns = tl.load(owned + i, mask=mask, other=0) != 0
    tok = tl.load(plan_tokens + i, mask=mask, other=0).to(tl.int32)
    tl.store(plan_state + i, tl.where(owns, tl.load(src + tok, mask=mask, other=0), -1), mask=mask)
    tok_parent = tl.load(columns + tok, mask=mask, other=0).to(tl.int32)
    tl.store(plan_init + i, tl.where(owns, tl.load(src + tok_parent, mask=mask, other=0), -1), mask=mask)
    # Geometry pass-through: the buffers replay the stacked geometry verbatim.
    tl.store(plan_tok + i, tl.load(plan_tokens + i, mask=mask, other=0), mask=mask)
    tl.store(plan_out + i, tl.load(plan_out_values + i, mask=mask, other=0), mask=mask)
    j = tl.arange(0, 2)
    tl.store(plan_cu + j, tl.load(cu_values + j))

    # The packed token order is the identity over the active prefix; the
    # per-group fallback copies the same arange into the same prefix.
    tl.store(token_indx + i, i, mask=i < query_width)
    tl.store(qsl, query_width)
    tl.store(asl, query_width)
    if HAS_ACCEPTED:
        tl.store(num_accepted, tl.load(accepted))

    rows = tl.arange(0, BLOCK)[:, None]
    taps = tl.arange(0, CONV_BLOCK)[None, :]
    conv_mask = (rows < SLOTS) & (taps < CONV_W)
    conv_offset = rows * CONV_W + taps
    tl.store(conv_rows + conv_offset, tl.load(conv_rows_src + conv_offset, mask=conv_mask, other=0), mask=conv_mask)
    prefix = tl.arange(0, CONV_BLOCK)
    prefix_mask = prefix < CONV_W - 1
    tl.store(
        conv_prefix + prefix,
        tl.load(conv_prefix_values + prefix, mask=prefix_mask, other=0),
        mask=prefix_mask,
    )


class GDNCroppedRefreshPointers:
    """Model-owned [G, 14] int64 address table of the fused cropped refresh.

    Every destination is one of a group's stable graph-capture buffers, so the
    table is rebuilt only when an address moves -- the per-step block table can
    move the source pages every step, which one pinned H2D rebuild covers.
    Output destinations must be private to their group and disjoint from every
    source, exactly like ``GDNTreePagePointers`` requires.
    """

    FIELDS = 14

    def __init__(self):
        self.signature = ()
        self.pointers = None
        self.buffers = ()

    def refresh(self, rows, device):
        signature = tuple(tuple(t.data_ptr() for t in row) for row in rows)
        if signature != self.signature:
            if any(len(row) != self.FIELDS for row in rows):
                raise ValueError("a cropped refresh row holds exactly 14 addresses")
            outputs = [ptr for row in signature for ptr in row[1:]]
            if len(set(outputs)) != len(outputs) or set(outputs).intersection(row[0] for row in signature):
                raise ValueError("cropped GDN groups require private, source-disjoint refresh destinations")
            host = torch.tensor(signature, device="cpu", dtype=torch.int64, pin_memory=True)
            self.pointers = host.to(device, non_blocking=True)
            # Keep the buffers the device table addresses alive.
            self.buffers = tuple(rows)
            self.signature = signature


class GDNCroppedRefreshBatch:
    """One cropped verification step; ``finish`` rewrites every group at once.

    ``AscendGDNHostMetadataBuilder._refresh_cropped_tree_metadata`` registers
    each warm group instead of issuing its ~55 per-group dispatches, applies
    the host-side descriptor updates itself and defers ``_finalize_metadata``
    to ``finish`` -- the same spot ``GDNTreePageBatch.finish`` finalizes the
    complete-row groups, after the layer loop and before the graph replay.
    Groups that fail a check keep the per-group path: every destination is
    private to its group, so the fused kernel and the fallback writes never
    overlap.
    """

    def __init__(self, pointers: GDNCroppedRefreshPointers):
        self.pointers = pointers
        self.rows: list[tuple[torch.Tensor, ...]] = []
        self.finalizers: list = []
        self._geometry = None

    def add(
        self,
        tree: "SharedGDNTreeMetadata",
        pages: torch.Tensor,
        builder: "AscendGDNHostMetadataBuilder",
        metadata: "GDNAttentionMetadata",
    ) -> bool:
        """Register one group's fourteen destinations; ``False`` keeps the fallback."""
        spec = metadata.spec_decode_metadata
        buffers = builder.spec_verify_plan_buffers
        if spec is None or spec.verify_plan is None or buffers is None or tree.invalid_slots is None:
            return False
        if self._geometry is None:
            geometry = tree._get_verify_geometry(tree.query_width)
            if geometry.num_chunks != 1:
                # bind_and_copy_into issues a single whole-tree chunk; keep the
                # fallback for any future multi-chunk geometry.
                return False
            self._geometry = geometry
        slots = tree.slots
        conv_width = tree.conv_width
        state = metadata.spec_state_indices_tensor
        init = spec.init_state_indices
        qsl = builder.spec_query_start_loc
        asl = builder.spec_actual_seq_lengths
        token_indx = builder.spec_token_indx
        num_accepted = metadata.num_accepted_tokens
        conv_rows = spec.conv_window_rows
        conv_prefix = spec.conv_prefix_src
        if (
            asl is None
            or init is None
            or conv_rows is None
            or conv_prefix is None
            or pages.device.type != "npu"
            or pages.device != tree.init_state_columns.device
            or not (pages.dtype == state.dtype == init.dtype == torch.int32)
            or not (pages.shape == state.shape == init.shape == (1, slots))
            or not (pages.stride(1) == state.stride(1) == init.stride(1) == 1)
            or state.device != pages.device
            or init.device != pages.device
        ):
            return False
        plan_tok = buffers.token_indices[0]
        plan_state = buffers.state_indices[0]
        plan_init = buffers.init_state_indices[0]
        plan_out = buffers.out_indices[0]
        plan_cu = buffers.cu_seqlens[0]
        if (
            not (plan_tok.dtype == plan_out.dtype == torch.int64)
            or not (plan_state.dtype == plan_init.dtype == plan_cu.dtype == torch.int32)
            or not (plan_tok.numel() == plan_state.numel() == plan_init.numel() == plan_out.numel())
            or plan_tok.numel() < slots
            or plan_cu.numel() < 2
            or not plan_tok.is_contiguous()
            or not plan_state.is_contiguous()
            or not plan_init.is_contiguous()
            or not plan_out.is_contiguous()
            or not plan_cu.is_contiguous()
            or plan_tok.device != pages.device
        ):
            return False
        if (
            token_indx.dtype != torch.int32
            or token_indx.numel() < slots
            or not token_indx.is_contiguous()
            or qsl.dtype != torch.int32
            or qsl.numel() < 2
            or not qsl.is_contiguous()
            or asl.dtype != torch.int32
            or asl.numel() < 2
            or not asl.is_contiguous()
            or num_accepted.dtype != torch.int32
            or num_accepted.numel() < 1
            or token_indx.device != pages.device
            or qsl.device != pages.device
            or asl.device != pages.device
            or num_accepted.device != pages.device
        ):
            return False
        if (
            conv_rows.dtype != torch.int32
            or conv_rows.shape != (slots, conv_width)
            or conv_rows.stride(0) != conv_width
            or conv_rows.stride(1) != 1
            or conv_prefix.dtype != torch.int32
            or conv_prefix.shape != (1, conv_width - 1)
            or not conv_prefix.is_contiguous()
            or conv_rows.device != pages.device
            or conv_prefix.device != pages.device
            or tree.conv_window_rows.shape != (slots, conv_width)
            or tree.conv_window_rows.dtype != torch.int32
            or not tree.conv_window_rows.is_contiguous()
            or tree.conv_prefix_src.shape != (1, conv_width - 1)
            or not tree.conv_prefix_src.is_contiguous()
            or tree.init_state_columns.shape != (1, slots)
            or not tree.init_state_columns.is_contiguous()
            or tree.invalid_slots.shape != (1, slots)
            or not tree.invalid_slots.is_contiguous()
        ):
            return False
        self.rows.append(
            (
                pages,
                state,
                init,
                plan_state,
                plan_init,
                plan_tok,
                plan_out,
                plan_cu,
                token_indx,
                qsl[1:],
                asl[1:],
                num_accepted,
                conv_rows,
                conv_prefix,
            )
        )
        self.finalizers.append((builder, metadata))
        return True

    def finish(self, tree: "SharedGDNTreeMetadata", accepted: torch.Tensor | None) -> None:
        """Write every registered group's fields in one launch, then finalize."""
        rows, finalizers = self.rows, self.finalizers
        self.rows, self.finalizers = [], []
        if not rows:
            return
        geometry = self._geometry
        device = tree.init_state_columns.device
        self.pointers.refresh(rows, device)
        has_accepted = (
            accepted is not None
            and accepted.dtype == torch.int32
            and accepted.numel() >= 1
            and accepted.device == device
        )
        _cropped_tree_refresh[(len(rows),)](
            self.pointers.pointers,
            tree.init_state_columns[0],
            tree.invalid_slots[0].view(torch.int8),
            geometry.owned[0].view(torch.int8),
            geometry.token_indices[0],
            geometry.out_indices[0],
            geometry.cu_seqlens[0],
            tree.conv_window_rows,
            tree.conv_prefix_src[0],
            accepted if has_accepted else tree.init_state_columns,
            tree.query_width,
            NULL_BLOCK_ID,
            SLOTS=tree.slots,
            CONV_W=tree.conv_width,
            BLOCK=triton.next_power_of_2(tree.slots),
            CONV_BLOCK=triton.next_power_of_2(tree.conv_width),
            HAS_ACCEPTED=has_accepted,
            multibuffer=False,
            unit_flag=False,
        )
        if not has_accepted:
            # Unexpected accepted layout: keep the per-group copy semantics.
            for builder, metadata in finalizers:
                metadata.num_accepted_tokens.copy_(accepted)
        # Convolution finalization can read the newly written page indices.
        for builder, metadata in finalizers:
            builder._finalize_metadata(metadata)
