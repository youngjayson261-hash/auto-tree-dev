# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the LAYOUT_ERROR layout-sanitization guardrail.

The method is extracted from the speculator source so these contracts run
without an NPU worker or a vLLM installation; PyTorch is required for the
real tensor mutation checks.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "tree_budget_policy_for_sanitize",
    ROOT / "vllm_ascend/worker/v2/spec_decode/tree/budget_policy.py",
)
policy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(policy)


def load_sanitize(namespace):
    node = ast.parse(
        (ROOT / "vllm_ascend/worker/v2/spec_decode/tree/speculator.py").read_text(encoding="utf-8")
    )
    cls = next(n for n in node.body if isinstance(n, ast.ClassDef) and n.name == "AscendTreeSpeculator")
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_sanitize_bad_tree_rows"))

    class RemoveImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None

        def visit_Import(self, node):
            return None

    method = RemoveImports().visit(method)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<actual _sanitize_bad_tree_rows>", "exec"), namespace)
    return namespace["_sanitize_bad_tree_rows"]


class LayoutSanitize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("PyTorch is needed for real tensor mutation contracts")
        import torch

        cls.torch = torch
        # staticmethod: a plain function class attribute would bind the test
        # instance as the extracted method's ``self``.
        cls.sanitize = staticmethod(load_sanitize({"torch": torch, "PolicyStatus": policy.PolicyStatus}))

    def fixture(self, rows=3, budget=4):
        torch = self.torch
        tree = NS(
            num_nodes=torch.full((rows,), budget, dtype=torch.int32),
            depths=torch.arange(1, budget + 1, dtype=torch.int32).expand(rows, budget).contiguous(),
            parents=torch.zeros((rows, budget), dtype=torch.int32),
            tokens=torch.arange(100, 100 + rows * budget, dtype=torch.int64).reshape(rows, budget),
            first_child=torch.arange(rows * (budget + 1), dtype=torch.int32).reshape(rows, budget + 1),
            next_sibling=torch.arange(rows * (budget + 1), dtype=torch.int32).reshape(rows, budget + 1),
            visibility=torch.ones((rows, budget, budget), dtype=torch.bool),
        )
        packet = torch.zeros((rows, 2), dtype=torch.int32)
        packet[:, 0] = budget
        packet[:, 1] = int(policy.PolicyStatus.CONFIDENCE)
        return tree, packet

    def test_healthy_rows_untouched(self):
        tree, packet = self.fixture()
        snapshot = {k: v.clone() for k, v in vars(tree).items()}
        owner = NS(input_batch=NS(num_reqs=3), tree_budget_packet=packet, tree=tree)
        self.sanitize(owner)
        for key, before in snapshot.items():
            self.assertTrue(self.torch.equal(getattr(tree, key), before), key)

    def test_bad_row_degenerates_to_root_only(self):
        tree, packet = self.fixture()
        packet[1, 0] = 0
        packet[1, 1] = int(policy.PolicyStatus.LAYOUT_ERROR)
        snapshot = {k: v.clone() for k, v in vars(tree).items()}
        owner = NS(input_batch=NS(num_reqs=3), tree_budget_packet=packet, tree=tree)
        self.sanitize(owner)

        self.assertEqual(int(tree.num_nodes[1]), 0)
        self.assertTrue(bool((tree.depths[1] == 0).all()))
        self.assertTrue(bool((tree.parents[1] == -1).all()))
        self.assertTrue(bool((tree.tokens[1] == -1).all()))
        self.assertTrue(bool((tree.first_child[1] == -1).all()))
        self.assertTrue(bool((tree.next_sibling[1] == -1).all()))
        self.assertTrue(bool((~tree.visibility[1]).all()))
        for row in (0, 2):
            for key, before in snapshot.items():
                self.assertTrue(self.torch.equal(getattr(tree, key)[row], before[row]), f"{key}[{row}]")

    def test_only_active_rows_are_considered(self):
        tree, packet = self.fixture()
        # Row 2 is beyond the active batch; its stale LAYOUT_ERROR status must
        # not be acted upon while only two requests are live.
        packet[2, 1] = int(policy.PolicyStatus.LAYOUT_ERROR)
        snapshot = {k: v.clone() for k, v in vars(tree).items()}
        owner = NS(input_batch=NS(num_reqs=2), tree_budget_packet=packet, tree=tree)
        self.sanitize(owner)
        for key, before in snapshot.items():
            self.assertTrue(self.torch.equal(getattr(tree, key), before), key)

    def test_empty_batch_is_noop(self):
        tree, packet = self.fixture()
        packet[:, 1] = int(policy.PolicyStatus.LAYOUT_ERROR)
        snapshot = {k: v.clone() for k, v in vars(tree).items()}
        owner = NS(input_batch=NS(num_reqs=0), tree_budget_packet=packet[:0], tree=tree)
        self.sanitize(owner)
        for key, before in snapshot.items():
            self.assertTrue(self.torch.equal(getattr(tree, key), before), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
