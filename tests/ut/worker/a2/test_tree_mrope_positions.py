# SPDX-License-Identifier: Apache-2.0
"""CPU execution of the real MRoPE kernel body and model-state routing.

The small pointer/Triton adapter executes integer loads/stores on CPU. It is
not an NPU compilation, graph-capture, or loaded-model numerical test.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]


def load_prepare_inputs():
    path = ROOT / "vllm_ascend/worker/v2/model_states/default.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendModelState")
    method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_inputs")
    cls = ast.ClassDef(
        name="Subject", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )

    class Base:
        def prepare_inputs(self, batch, req_states):
            self.super_calls += 1
            return {"positions": "upstream-linear", "upstream_marker": True}

    calls = []

    def prepare(rope_state, batch, req_states, *, depth_positions=None):
        calls.append((rope_state, batch, req_states, depth_positions))
        return "depth-mrope"

    namespace = dict(Base=Base, Any=object, prepare_mrope_positions=prepare)
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    subject = namespace["Subject"]()
    subject.rope_state = object()
    subject.super_calls = 0
    subject.prepare_engram_inputs = lambda batch, states: {"engram_marker": "kept"}
    return subject, calls


class TestModelStateRouting(unittest.TestCase):
    def test_tree_mrope_bypasses_linear_override(self):
        subject, calls = load_prepare_inputs()
        depth = object()
        batch = NS(tree_depths=object(), slot_positions=object(), positions=depth)
        states = object()
        result = subject.prepare_inputs(batch, states)
        self.assertEqual(result, {"positions": "depth-mrope", "engram_marker": "kept"})
        self.assertEqual(subject.super_calls, 0)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][3], depth)

    def test_ar_and_topk_one_keep_original_path(self):
        for attrs in (
            {},
            {"tree_depths": object(), "slot_positions": None},
            {"tree_depths": None, "slot_positions": object()},
        ):
            with self.subTest(attrs=attrs):
                subject, calls = load_prepare_inputs()
                result = subject.prepare_inputs(NS(**attrs), object())
                self.assertEqual(subject.super_calls, 1)
                self.assertEqual(calls, [])
                self.assertEqual(result["positions"], "upstream-linear")
                self.assertTrue(result["upstream_marker"])
                self.assertEqual(result["engram_marker"], "kept")

    def test_one_dimensional_tree_keeps_original_path(self):
        subject, calls = load_prepare_inputs()
        subject.rope_state = None
        subject.prepare_inputs(NS(tree_depths=object(), slot_positions=object()), object())
        self.assertEqual(subject.super_calls, 1)
        self.assertEqual(calls, [])


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor = tensor
        self.offset = offset

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)

    __radd__ = __add__


class TensorValue(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(TensorValue)


class TensorLanguage:
    constexpr = object()
    int64 = np.int64

    def __init__(self):
        self.row = 0
        self.rows = 0

    def program_id(self, axis):
        return self.row

    def num_programs(self, axis):
        return self.rows

    @staticmethod
    def arange(start, end):
        return np.arange(start, end, dtype=np.int64).view(TensorValue)

    range = staticmethod(range)
    static_range = staticmethod(range)

    @staticmethod
    def load(pointer, mask=None, other=0):
        offsets = np.asarray(pointer.offset)
        valid = np.broadcast_to(True if mask is None else mask, offsets.shape)
        selected = offsets[valid]
        flat = pointer.tensor.reshape(-1)
        if np.any(selected < 0) or np.any(selected >= flat.numel()):
            raise AssertionError("unmasked out-of-bounds kernel read")
        result = np.full(offsets.shape, other, dtype=flat.numpy().dtype)
        result[valid] = flat[selected].numpy()
        return result.item() if result.ndim == 0 else result.view(TensorValue)

    @staticmethod
    def store(pointer, value, mask=None):
        offsets = np.asarray(pointer.offset)
        valid = np.broadcast_to(True if mask is None else mask, offsets.shape)
        selected = offsets[valid]
        flat = pointer.tensor.reshape(-1)
        if np.any(selected < 0) or np.any(selected >= flat.numel()):
            raise AssertionError("unmasked out-of-bounds kernel write")
        values = np.broadcast_to(value, offsets.shape)[valid]
        flat[selected] = torch.as_tensor(values.copy(), dtype=flat.dtype)


def load_kernel_subject():
    path = ROOT / "vllm_ascend/worker/v2/mrope.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    language = TensorLanguage()

    class Kernel:
        def __init__(self, function):
            self.function = function
            self.launches = []

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.launches.append((args, kwargs))
                language.rows = grid[0]
                for row in range(grid[0]):
                    language.row = row
                    converted = [Pointer(x) if isinstance(x, torch.Tensor) else x for x in args]
                    self.function(*converted, **kwargs)

            return launch

    namespace = dict(torch=torch, tl=language, triton=NS(jit=Kernel), TYPE_CHECKING=False)
    exec(compile(tree, str(path), "exec", flags=__import__("__future__").annotations.compiler_flag), namespace)
    return NS(**namespace)


def fixture(
    lengths=(5,), mapping=(0,), computed=(100,), prefill_lens=(0,), deltas=(0,), depth=None, capacity=None, dims=3
):
    q = sum(lengths)
    capacity = q if capacity is None else capacity
    max_len = max(max(computed) + max(lengths) + 3, 20)
    max_reqs = len(computed)
    prefill = torch.arange(max_reqs * dims * max_len, dtype=torch.int32).reshape(max_reqs * dims, max_len)
    positions = torch.full((dims, capacity + 2), -999, dtype=torch.int64)
    rope = NS(
        num_dims=dims,
        positions=positions,
        prefill_positions=NS(gpu=prefill),
        prefill_delta=NS(gpu=torch.tensor(deltas, dtype=torch.int32)),
        get_positions=lambda n: positions[:, :n],
    )
    qsl = torch.tensor([0, *np.cumsum(lengths).tolist()], dtype=torch.int32)
    batch = NS(
        num_reqs=len(lengths),
        num_tokens=q,
        num_tokens_after_padding=capacity,
        idx_mapping=torch.tensor(mapping, dtype=torch.int32),
        query_start_loc=qsl,
    )
    states = NS(
        prefill_len=NS(gpu=torch.tensor(prefill_lens, dtype=torch.int32)),
        num_computed_tokens=NS(gpu=torch.tensor(computed, dtype=torch.int32)),
    )
    depth_tensor = None
    if depth is not None:
        depth_tensor = torch.full((capacity,), 777, dtype=torch.int64)
        depth_tensor[:q] = torch.tensor(depth, dtype=torch.int64)
    return rope, batch, states, depth_tensor


def oracle(rope, batch, states, depth):
    output = torch.zeros((rope.num_dims, batch.num_tokens_after_padding), dtype=torch.int64)
    for row, state_id in enumerate(batch.idx_mapping.tolist()):
        start, end = batch.query_start_loc[row : row + 2].tolist()
        computed = int(states.num_computed_tokens.gpu[state_id])
        if computed < int(states.prefill_len.gpu[state_id]):
            output[:, start:end] = rope.prefill_positions.gpu[
                state_id * rope.num_dims : (state_id + 1) * rope.num_dims, computed : computed + end - start
            ]
        else:
            logical = torch.arange(computed, computed + end - start) if depth is None else depth[start:end]
            output[:, start:end] = logical + int(rope.prefill_delta.gpu[state_id])
    return output


class TestMropeKernelBody(unittest.TestCase):
    def check(self, args):
        subject = load_kernel_subject()
        rope, batch, states, depth = args
        expected = oracle(*args)
        storage = rope.positions.data_ptr()
        actual = subject.prepare_mrope_positions(rope, batch, states, depth_positions=depth)
        self.assertTrue(torch.equal(actual, expected), (actual, expected))
        self.assertEqual(actual.data_ptr(), storage)
        self.assertEqual(actual.stride(), rope.positions.stride())
        self.assertEqual(len(subject._prepare_mrope_positions_kernel.launches), 1)
        self.assertTrue(torch.all(rope.positions[:, batch.num_tokens_after_padding :] == -999))
        return subject, args, actual

    def test_siblings_share_depth_in_all_three_planes(self):
        self.check(fixture(depth=[100, 101, 101, 102, 102], capacity=9))

    def test_nonzero_negative_delta_is_applied_per_request(self):
        self.check(
            fixture(
                lengths=(3, 2),
                mapping=(1, 0),
                computed=(100, 50),
                prefill_lens=(0, 0),
                deltas=(-7, 11),
                depth=[50, 51, 51, 100, 101],
                capacity=9,
            )
        )

    def test_prefill_keeps_distinct_original_planes(self):
        self.check(fixture(computed=(2,), prefill_lens=(12,), deltas=(99,), depth=[999] * 5, capacity=9))

    def test_mixed_prefill_decode_reordered_rows(self):
        self.check(
            fixture(
                lengths=(3, 4),
                mapping=(1, 0),
                computed=(100, 2),
                prefill_lens=(0, 12),
                deltas=(-7, 99),
                depth=[888, 888, 888, 100, 101, 101, 102],
                capacity=9,
            )
        )

    def test_four_rows_include_root_only_and_zero_length(self):
        self.check(
            fixture(
                lengths=(1, 0, 3, 2),
                mapping=(3, 1, 0, 2),
                computed=(100, 20, 3, 50),
                prefill_lens=(0, 0, 9, 0),
                deltas=(-7, 4, 99, 8),
                depth=[50, 100, 101, 101, 999, 999],
                capacity=9,
            )
        )

    def test_optional_linear_path_matches_upstream_formula(self):
        self.check(
            fixture(lengths=(2, 3), mapping=(1, 0), computed=(100, 50), prefill_lens=(0, 0), deltas=(7, -2), capacity=7)
        )

    def test_exact_capacity_and_other_num_dims(self):
        for dims in (1, 4):
            with self.subTest(dims=dims):
                self.check(fixture(depth=[100, 101, 101, 102, 102], dims=dims))

    def test_block_boundary_prefill_and_decode(self):
        for is_prefill in (False, True):
            with self.subTest(is_prefill=is_prefill):
                self.check(
                    fixture(
                        lengths=(1030,),
                        computed=(2,),
                        prefill_lens=(2000 if is_prefill else 0,),
                        depth=[2 + i // 3 for i in range(1030)],
                        capacity=1040,
                    )
                )

    def test_repeat_call_reuses_storage_and_clears_old_padding(self):
        subject, args, _ = self.check(fixture(depth=[100, 101, 101, 102, 102], capacity=9))
        rope, batch, states, depth = args
        before = rope.positions.data_ptr()
        batch.num_tokens = 3
        batch.query_start_loc[1] = 3
        depth[:3] = torch.tensor([100, 103, 103])
        states.num_computed_tokens.gpu[0] = 100
        rope.prefill_delta.gpu[0] = 5
        actual = subject.prepare_mrope_positions(rope, batch, states, depth_positions=depth)
        self.assertTrue(torch.equal(actual, oracle(*args)))
        self.assertEqual(rope.positions.data_ptr(), before)

    def test_zero_real_requests_only_clears_padding(self):
        args = fixture(depth=[100, 101, 101, 102, 102], capacity=9)
        rope, batch, states, depth = args
        batch.num_reqs = batch.num_tokens = 0
        batch.idx_mapping = torch.empty(0, dtype=torch.int32)
        batch.query_start_loc = torch.zeros(1, dtype=torch.int32)
        self.check(args)

    def test_invalid_layout_is_rejected_before_launch(self):
        for bad in ("depth_rank", "depth_length", "capacity", "request_rows", "planes"):
            with self.subTest(bad=bad):
                subject = load_kernel_subject()
                rope, batch, states, depth = fixture(depth=[100, 101, 101, 102, 102], capacity=9)
                if bad == "depth_rank":
                    depth = depth.unsqueeze(0)
                elif bad == "depth_length":
                    depth = depth[:4]
                elif bad == "capacity":
                    batch.num_tokens_after_padding = rope.positions.shape[1]
                elif bad == "request_rows":
                    batch.num_reqs = 2
                else:
                    rope.num_dims = 4
                with self.assertRaises(ValueError):
                    subject.prepare_mrope_positions(rope, batch, states, depth_positions=depth)
                self.assertEqual(subject._prepare_mrope_positions_kernel.launches, [])


if __name__ == "__main__":
    unittest.main()
