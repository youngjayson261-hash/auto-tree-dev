# SPDX-License-Identifier: Apache-2.0
"""CPU contracts of the fused cropped-row refresh bookkeeping.

The device-side value equivalence lives in the singlecard nightly suite
(``test_gdn_cropped_refresh.py`` there); these tests pin the host-side
contracts that need no NPU: pointer-table signature stability, destination
privacy and the ``VLLM_ASCEND_GDN_CROPPED_BATCH`` kill switch.
"""

import pytest
import torch

from vllm_ascend.ops.triton.spec_decode.tree.gdn_tree_pages import (
    GDNCroppedRefreshPointers,
    cropped_refresh_enabled,
)


def _row(offset=0):
    return tuple(torch.zeros(4, dtype=torch.int32) for _ in range(14))


def test_pointer_table_rebuilds_only_on_signature_change():
    pointers = GDNCroppedRefreshPointers()
    rows = [_row(), _row()]
    pointers.refresh(rows, torch.device("cpu"))
    table = pointers.pointers
    assert table is not None and table.shape == (2, 14)
    pointers.refresh(rows, torch.device("cpu"))  # same addresses: no rebuild
    assert pointers.pointers is table
    rows[0] = tuple(list(rows[0])[:-1] + [torch.zeros(4, dtype=torch.int32)])
    pointers.refresh(rows, torch.device("cpu"))  # a moved destination rebuilds
    assert pointers.pointers is not table


def test_pointer_table_rejects_shared_destinations():
    pointers = GDNCroppedRefreshPointers()
    shared = torch.zeros(4, dtype=torch.int32)
    row_a = (torch.zeros(4, dtype=torch.int32), shared) + tuple(torch.zeros(4, dtype=torch.int32) for _ in range(12))
    row_b = (torch.zeros(4, dtype=torch.int32), shared) + tuple(torch.zeros(4, dtype=torch.int32) for _ in range(12))
    with pytest.raises(ValueError, match="private"):
        pointers.refresh([row_a, row_b], torch.device("cpu"))


def test_pointer_table_rejects_source_destination_overlap():
    pointers = GDNCroppedRefreshPointers()
    src = torch.zeros(4, dtype=torch.int32)
    row = (src, src) + tuple(torch.zeros(4, dtype=torch.int32) for _ in range(12))
    with pytest.raises(ValueError, match="private"):
        pointers.refresh([row], torch.device("cpu"))


def test_cropped_refresh_kill_switch(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_GDN_CROPPED_BATCH", raising=False)
    assert cropped_refresh_enabled()
    monkeypatch.setenv("VLLM_ASCEND_GDN_CROPPED_BATCH", "1")
    assert cropped_refresh_enabled()
    monkeypatch.setenv("VLLM_ASCEND_GDN_CROPPED_BATCH", "0")
    assert not cropped_refresh_enabled()
