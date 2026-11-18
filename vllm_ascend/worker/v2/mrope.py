# SPDX-License-Identifier: Apache-2.0
"""Prepare MRoPE in the existing buffers without losing packed-tree depths."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.mm.rope import RopeState
    from vllm.v1.worker.gpu.states import RequestState


@triton.jit
def _prepare_mrope_positions_kernel(
    positions_ptr,
    positions_stride0,
    positions_stride1,
    prefill_positions_ptr,
    prefill_stride0,
    prefill_stride1,
    prefill_delta_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    depth_positions_ptr,
    depth_stride,
    num_tokens,
    num_tokens_after_padding,
    NUM_DIMS: tl.constexpr,
    USE_TREE_POS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if row == num_reqs:
        for i in tl.range(num_tokens, num_tokens_after_padding, BLOCK_SIZE):
            token = i + tl.arange(0, BLOCK_SIZE)
            for dim in tl.static_range(NUM_DIMS):
                tl.store(
                    positions_ptr + dim * positions_stride0 + token * positions_stride1,
                    0,
                    mask=token < num_tokens_after_padding,
                )
        return

    state_idx = tl.load(idx_mapping_ptr + row)
    num_computed = tl.load(num_computed_tokens_ptr + state_idx)
    is_prefill = num_computed < tl.load(prefill_lens_ptr + state_idx)
    start = tl.load(query_start_loc_ptr + row)
    end = tl.load(query_start_loc_ptr + row + 1)
    delta = tl.load(prefill_delta_ptr + state_idx)
    for i in tl.range(0, end - start, BLOCK_SIZE):
        local = i + tl.arange(0, BLOCK_SIZE)
        mask = local < end - start
        token = start + local
        for dim in tl.static_range(NUM_DIMS):
            if is_prefill:
                pos = tl.load(
                    prefill_positions_ptr
                    + (state_idx * NUM_DIMS + dim) * prefill_stride0
                    + (num_computed + local) * prefill_stride1,
                    mask=mask,
                    other=0,
                ).to(tl.int64)
            else:
                if USE_TREE_POS:
                    pos = tl.load(depth_positions_ptr + token * depth_stride, mask=mask, other=0).to(tl.int64)
                else:
                    pos = (num_computed + local).to(tl.int64)
                pos = pos + delta
            tl.store(
                positions_ptr + dim * positions_stride0 + token * positions_stride1,
                pos,
                mask=mask,
            )


def prepare_mrope_positions(
    rope_state: RopeState,
    input_batch: InputBatch,
    req_states: RequestState,
    *,
    depth_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep original prefill planes and add each decode row's MRoPE delta.

    ``depth_positions`` is the runner's already prepared logical position, not
    a KV slot index. All output uses RopeState's persistent, strided buffer.
    The optional linear form has the upstream decode semantics; ordinary AR
    continues to use the upstream preparation method directly.
    """
    positions = rope_state.positions
    num_reqs = input_batch.num_reqs
    num_tokens = input_batch.num_tokens
    capacity = input_batch.num_tokens_after_padding
    if positions.ndim != 2 or positions.shape[0] != rope_state.num_dims:
        raise ValueError("MRoPE positions must preserve the configured planes")
    if not 0 <= num_tokens <= capacity < positions.shape[1]:
        raise ValueError("MRoPE tokens exceed the persistent position buffer")
    if input_batch.idx_mapping.shape[0] != num_reqs or input_batch.query_start_loc.shape[0] < num_reqs + 1:
        raise ValueError("MRoPE request rows do not match the input layout")
    if depth_positions is not None:
        if depth_positions.ndim != 1 or depth_positions.shape[0] < capacity:
            raise ValueError("Tree MRoPE requires one logical depth position per padded token")
        if depth_positions.device != positions.device:
            raise ValueError("Tree MRoPE positions must be on the persistent buffer device")
    prefill = rope_state.prefill_positions.gpu
    _prepare_mrope_positions_kernel[(num_reqs + 1,)](
        positions,
        positions.stride(0),
        positions.stride(1),
        prefill,
        prefill.stride(0),
        prefill.stride(1),
        rope_state.prefill_delta.gpu,
        input_batch.idx_mapping,
        input_batch.query_start_loc,
        req_states.prefill_len.gpu,
        req_states.num_computed_tokens.gpu,
        depth_positions if depth_positions is not None else positions,
        depth_positions.stride(0) if depth_positions is not None else 0,
        num_tokens,
        capacity,
        NUM_DIMS=rope_state.num_dims,
        USE_TREE_POS=depth_positions is not None,
        BLOCK_SIZE=1024,
    )
    return rope_state.get_positions(capacity)
