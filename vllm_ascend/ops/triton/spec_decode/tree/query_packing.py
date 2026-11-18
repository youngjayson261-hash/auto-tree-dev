# SPDX-License-Identifier: Apache-2.0
"""Persistent TND/BSND query transfers for ragged tree FULL graph tasks.

Graph task groups cannot contain device memcpy tasks on the supported CANN
runtime. These kernels consume the existing device query boundaries, keep a
fixed launch grid, and never copy padding rows into another request.
"""

import math

import torch
from vllm.triton_utils import tl, triton

_QUERY_TRANSFER_BLOCK_SIZE = 1024


@triton.jit
def _pack_kernel(src, starts, dst, G: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    begin = tl.load(starts + row)
    end = tl.load(starts + row + 1)
    token = offsets // F
    values = tl.load(src + (begin + token) * F + offsets % F, mask=(offsets < G * F) & (token < end - begin), other=0)
    tl.store(dst + row * G * F + offsets, values, mask=offsets < G * F)


@triton.jit
def _unpack_kernel(src, starts, dst, R: tl.constexpr, G: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = offsets // F
    row = tl.full((BLOCK,), 0, tl.int32)
    for index in tl.static_range(1, R):
        row += (token >= tl.load(starts + index)).to(tl.int32)
    begin = tl.load(starts + row)
    total = tl.load(starts + R)
    values = tl.load(
        src + (row * G + token - begin) * F + offsets % F, mask=(offsets < R * G * F) & (token < total), other=0
    )
    # Write the full physical output: its unused tail stays zero.
    tl.store(dst + offsets, values, mask=offsets < R * G * F)


def transfer_tree_queries(flat: torch.Tensor, starts: torch.Tensor, packed: torch.Tensor, *, unpack: bool) -> None:
    """Transfer into caller-owned buffers; real lengths were validated by metadata.

    ``starts`` is the existing zero-prefixed device query_start_loc, not a new
    host/device length buffer. Its first R+1 entries describe positive, packed
    queries of length at most G. The owning attention metadata validates that
    contract before capture/update; no device synchronization is needed here.
    """
    if packed.ndim < 3 or flat.shape != (packed.shape[0] * packed.shape[1], *packed.shape[2:]):
        raise ValueError("query transfer requires [R*G,...] and [R,G,...] buffers")
    if flat.dtype != packed.dtype or flat.device != packed.device or starts.device != flat.device:
        raise ValueError("query transfer payloads and boundaries must share a device and payload dtype")
    if starts.ndim != 1 or starts.numel() < packed.shape[0] + 1 or starts.dtype not in (torch.int32, torch.int64):
        raise ValueError("query transfer requires R+1 integer boundaries")
    if not flat.is_contiguous() or not packed.is_contiguous() or not starts.is_contiguous():
        raise ValueError("query transfer requires contiguous buffers")
    rows, capacity = packed.shape[:2]
    features, block = math.prod(packed.shape[2:]), _QUERY_TRANSFER_BLOCK_SIZE
    if unpack:
        _unpack_kernel[(triton.cdiv(rows * capacity * features, block),)](
            packed, starts, flat, R=rows, G=capacity, F=features, BLOCK=block
        )
    else:
        _pack_kernel[(triton.cdiv(capacity * features, block), rows)](
            flat, starts, packed, G=capacity, F=features, BLOCK=block
        )
