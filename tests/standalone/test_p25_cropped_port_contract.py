"""CPU contracts for the cropped-row shared GDN metadata port.

The cropped port (P25-A/B semantics on the tree261113-rebase architecture)
lets a truncated BS1 verification row share one logical metadata object
across every GDN cache group. These tests pin the contract:

* ``create`` keeps full ``slots`` shapes and records the host-known widths;
* cropped binds mask inactive slots with ``NULL_BLOCK_ID``;
* the verify geometry truncates at ``query_width`` and diverts inactive
  outputs to the scratch area, so a FULL graph replays stable buffers;
* cropped conv-window rows match the full table row for row on the prefix.
"""
import pytest
import torch
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.worker.v2.spec_decode.tree.shared_metadata import SharedGDNTreeMetadata

SLOTS = 33  # 1 + budget 32
CONV_WIDTH = 3
WIDTHS = (33, 24, 17, 9, 2)


def make_tree(branching=2):
    parents = torch.empty((1, SLOTS - 1), dtype=torch.int32)
    depths = torch.zeros((1, SLOTS), dtype=torch.int32)
    for node in range(1, SLOTS):
        parent = node - 1 if branching == 1 else (node - 1) // branching
        parents[0, node - 1] = parent
        depths[0, node] = depths[0, parent] + 1
    return dict(
        parents=parents,
        num_nodes=torch.tensor([SLOTS - 1], dtype=torch.int32),
        depths=depths[:, 1:],
        num_accepted_tokens=torch.tensor([3], dtype=torch.int32),
        prev_path_node_ids=torch.arange(16, dtype=torch.int32)[None, :],
        prev_num_sampled=torch.tensor([4], dtype=torch.int32),
    )


def create(tree, query_width=None):
    kwargs = {} if query_width is None else dict(query_width=query_width, graph_capacity=SLOTS)
    return SharedGDNTreeMetadata.create(
        tree["parents"],
        tree["num_nodes"],
        tree["depths"],
        tree["num_accepted_tokens"],
        tree["prev_path_node_ids"],
        tree["prev_num_sampled"],
        CONV_WIDTH,
        **kwargs,
    )


@pytest.mark.parametrize("branching", (1, 2, 4))
def test_full_row_defaults(branching):
    tree = make_tree(branching)
    full = create(tree)
    assert full.query_width == SLOTS and full.graph_capacity == SLOTS
    assert full.invalid_slots is None
    assert full.conv_window_rows.shape == (SLOTS, CONV_WIDTH)
    pages = torch.arange(SLOTS, dtype=torch.int32)[None, :] + 1000
    assert full.bind_state_indices(pages) is pages
    geometry = full._get_verify_geometry(scratch_base=SLOTS)
    assert geometry.num_chunks == 1
    assert int(geometry.cu_seqlens[0, 1]) == SLOTS
    assert bool(geometry.owned.all())


@pytest.mark.parametrize("width", WIDTHS)
def test_cropped_prefix_matches_full(width):
    tree = make_tree()
    full = create(tree)
    cropped = full if width == SLOTS else create(tree, query_width=width)
    # Same logical content on the active prefix, full shapes retained.
    assert cropped.conv_window_rows.shape == full.conv_window_rows.shape
    torch.testing.assert_close(cropped.conv_window_rows[:width], full.conv_window_rows[:width])
    torch.testing.assert_close(cropped.init_state_columns, full.init_state_columns)
    torch.testing.assert_close(cropped.conv_prefix_src, full.conv_prefix_src)
    if width == SLOTS:
        return
    assert cropped.query_width == width and cropped.graph_capacity == SLOTS
    assert cropped.invalid_slots is not None
    assert not bool(cropped.invalid_slots[0, :width].any())
    assert bool(cropped.invalid_slots[0, width:].all())


@pytest.mark.parametrize("width", (24, 17, 9, 2))
def test_cropped_binds_mask_inactive_slots(width):
    cropped = create(make_tree(), query_width=width)
    pages = torch.arange(SLOTS, dtype=torch.int32)[None, :] + 1000
    init = cropped.bind_init_state_indices(pages)
    assert init.dtype == torch.int32
    assert int((init[0, width:] == NULL_BLOCK_ID).all())
    bound = cropped.bind_state_indices(pages)
    torch.testing.assert_close(bound[0, :width], pages[0, :width])
    assert int((bound[0, width:] == NULL_BLOCK_ID).all())


@pytest.mark.parametrize("width", (24, 17, 9, 2))
def test_cropped_verify_geometry_truncates(width):
    cropped = create(make_tree(), query_width=width)
    scratch_base = 100
    geometry = cropped._get_verify_geometry(scratch_base)
    assert geometry.num_chunks == 1
    assert int(geometry.cu_seqlens[0, 0]) == 0
    assert int(geometry.cu_seqlens[0, 1]) == width
    assert bool(geometry.owned[0, :width].all())
    assert not bool(geometry.owned[0, width:].any())
    # Active cells address their own token column; inactive cells scratch.
    torch.testing.assert_close(geometry.out_indices[0, :width], torch.arange(width))
    torch.testing.assert_close(
        geometry.out_indices[0, width:], torch.arange(width, SLOTS) + scratch_base
    )
    # Geometry is cached per scratch base.
    assert cropped._get_verify_geometry(scratch_base) is geometry


def test_width_validation():
    tree = make_tree()
    with pytest.raises(ValueError):
        create(tree, query_width=0)
    with pytest.raises(ValueError):
        create(tree, query_width=SLOTS + 1)
    with pytest.raises(ValueError):
        SharedGDNTreeMetadata.create(
            tree["parents"], tree["num_nodes"], tree["depths"], tree["num_accepted_tokens"],
            tree["prev_path_node_ids"], tree["prev_num_sampled"], CONV_WIDTH,
            query_width=17, graph_capacity=9,
        )
