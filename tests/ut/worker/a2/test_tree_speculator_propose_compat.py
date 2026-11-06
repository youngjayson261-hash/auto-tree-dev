# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real Tree entry point without importing device dependencies."""

from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/tree/speculator.py"


class ParentCalled(Exception):
    def __init__(self, args, kwargs):
        self.args_received = args
        self.kwargs_received = kwargs


def load_entry(legacy):
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    tree = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "AscendTreeSpeculator")
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "propose")

    class Parent:
        def propose(self, *args, **kwargs):
            if legacy and "num_speculative_tokens" in kwargs:
                raise TypeError("0.31 parent does not accept num_speculative_tokens")
            raise ParentCalled(args, kwargs)

    extracted = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            ast.ClassDef(
                name="AscendTreeSpeculator",
                bases=[ast.Name(id="AscendDFlashSpeculator", ctx=ast.Load())],
                keywords=[],
                body=[method],
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    namespace = {"AscendDFlashSpeculator": Parent, "vllm_version_is": lambda version: legacy}
    exec(compile(ast.fix_missing_locations(extracted), str(SOURCE), "exec"), namespace)
    instance = namespace["AscendTreeSpeculator"]()
    instance.num_speculative_steps = 16
    instance._dsv4_dspark_draft = False
    return instance


class TreeProposeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        package_name = "vllm_ascend.worker.v2.spec_decode.tree"
        package = types.ModuleType(package_name)
        chain = types.ModuleType(package_name + ".chain_pack")
        chain._LAYOUT = object()
        chain.clear_tree_chain_layout = lambda: None
        package.chain_pack = chain
        self.modules = patch.dict(sys.modules, {package_name: package, chain.__name__: chain})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.arguments = dict(
            input_batch=object(),
            attn_metadata={},
            slot_mappings={},
            last_hidden_states=object(),
            aux_hidden_states=None,
            num_sampled=object(),
            num_rejected=object(),
            last_sampled=object(),
            next_prefill_tokens=object(),
            temperature=object(),
            seeds=object(),
            dummy_run=True,
        )

    def dispatch(self, legacy, **extra):
        with self.assertRaises(ParentCalled) as captured:
            load_entry(legacy).propose(**(self.arguments | extra))
        return captured.exception

    def test_031_keyword_preserves_sync_object(self):
        sync = object()
        call = self.dispatch(True, dp_sync=sync)
        self.assertIs(call.args_received[11], sync)
        self.assertNotIn("num_speculative_tokens", call.kwargs_received)

    def test_031_none_sync_profiles_without_new_parent_keyword(self):
        call = self.dispatch(True, dp_sync=None, is_profile=True)
        self.assertIsNone(call.args_received[11])
        self.assertTrue(call.kwargs_received["is_profile"])
        self.assertNotIn("num_speculative_tokens", call.kwargs_received)

    def test_031_existing_state_argument_still_works(self):
        sync = object()
        call = self.dispatch(True, dp_sync_state=sync)
        self.assertIs(call.args_received[11], sync)
        self.assertNotIn("num_speculative_tokens", call.kwargs_received)

    def test_newer_parent_retains_token_count_and_sync_state(self):
        sync = object()
        call = self.dispatch(False, dp_sync_state=sync, num_speculative_tokens=16)
        self.assertIs(call.args_received[11], sync)
        self.assertEqual(call.kwargs_received["num_speculative_tokens"], 16)

    def test_conflicting_sync_aliases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sync"):
            load_entry(True).propose(**self.arguments, dp_sync=object(), dp_sync_state=object())

    def test_identical_sync_aliases_preserve_the_object(self):
        sync = object()
        call = self.dispatch(True, dp_sync=sync, dp_sync_state=sync)
        self.assertIs(call.args_received[11], sync)

    def test_dynamic_draft_depth_remains_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixed draft depth"):
            load_entry(True).propose(**self.arguments, num_speculative_tokens=15)


if __name__ == "__main__":
    unittest.main()
