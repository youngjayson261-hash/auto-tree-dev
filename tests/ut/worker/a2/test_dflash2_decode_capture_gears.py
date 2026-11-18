# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Ascend reshape method after upstream decode deduplication."""

import ast
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dflash2_confidence_admission as base
import test_tree_prefix_graph as legacy


class Mode(Enum):
    FULL = 1


@dataclass(frozen=True)
class Descriptor:
    cg_mode: Mode
    num_tokens: int
    num_reqs: int
    max_query_len: int | None = None
    uniform_token_count: int | None = None
    num_active_loras: int = 0
    num_ubatches: int = 1


def reshape_method():
    root = Path(__file__).resolve().parents[4]
    path = root / "vllm_ascend/worker/v2/aclgraph_utils.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_reshape_tree_verify_full_descs")
    scope = dict(replace=replace, CUDAGraphMode=Mode, BatchExecutionDescriptor=Descriptor)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[node.name]


def config():
    value = base.hybrid_candidate_config()
    value.parallel_config.tensor_parallel_size = 2
    value.additional_config["tree_spec_config"]["budget"] = 32
    value.compilation_config.cudagraph_mode.name = "FULL_DECODE_ONLY"
    value.compilation_config.cudagraph_capture_sizes = [1, 9, 17, 25, 33]
    return value


class TestHybridDecodeGearPlans(unittest.TestCase):
    def setUp(self):
        shape = NS(TreeShape=NS(from_config=lambda c: NS(target_query_len=c.additional_config["tree_spec_config"]["budget"] + 1)))
        self.modules = patch.dict(sys.modules, {
            "vllm_ascend.worker.v2.spec_decode.tree.shape": shape,
            "vllm_ascend.worker.v2.spec_decode.tree.prefix_graph": base.prefix,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.reshape = reshape_method()

    def manager(self, descs, cfg=None):
        cfg = config() if cfg is None else cfg
        return NS(vllm_config=cfg, compilation_config=cfg.compilation_config,
                  _capture_descs={Mode.FULL: list(descs)}, _candidates={"native": list(descs)})

    def test_real_single_nominal_descriptor_expands_to_all_configured_gears(self):
        original = Descriptor(Mode.FULL, 33, 1, uniform_token_count=33)
        manager = self.manager([original])
        self.reshape(manager)
        planned = manager._capture_descs[Mode.FULL]
        self.assertEqual([d.num_tokens for d in planned], [33, 25, 17, 9, 1])
        self.assertTrue(all(d.num_reqs == 1 and d.max_query_len == d.num_tokens and d.uniform_token_count is None for d in planned))
        self.assertEqual(original.uniform_token_count, 33)
        self.assertIs(manager._candidates["native"][0], planned[0])

    def test_already_complete_plans_do_not_gain_duplicate_gears(self):
        descs = [Descriptor(Mode.FULL, n, 1, max_query_len=n) for n in [33, 25, 17, 9, 1]]
        manager = self.manager(descs)
        self.reshape(manager)
        self.assertEqual([d.num_tokens for d in manager._capture_descs[Mode.FULL]], [33, 25, 17, 9, 1])
        self.assertTrue(all(a is b for a, b in zip(manager._candidates["native"], manager._capture_descs[Mode.FULL])))

    def test_every_query_length_dispatches_to_smallest_planned_gear(self):
        nominal = Descriptor(Mode.FULL, 33, 1, uniform_token_count=33)
        manager = self.manager([nominal])
        manager._candidates = {(n, 0): [nominal] for n in range(34)}
        self.reshape(manager)
        plans = manager._capture_descs[Mode.FULL]
        for n in range(1, 34):
            with self.subTest(query_tokens=n):
                candidates = manager._candidates[(n, 0)]
                expected = min(d.num_tokens for d in plans if d.num_tokens >= n)
                self.assertEqual([d.num_tokens for d in candidates], [expected])
                self.assertTrue(any(candidates[0] is d for d in plans))
        self.assertNotIn((34, 0), manager._candidates)

    def test_missing_nominal_capture_template_is_rejected(self):
        with self.assertRaises(ValueError):
            self.reshape(self.manager([Descriptor(Mode.FULL, 17, 1)]))

    def test_ambiguous_or_incompatible_templates_are_rejected(self):
        cases = [
            [Descriptor(Mode.FULL, 33, 1), Descriptor(Mode.FULL, 33, 1)],
            [Descriptor(Mode.FULL, 33, 2)],
            [Descriptor(Mode.FULL, 33, 1, uniform_token_count=16)],
            [Descriptor(Mode.FULL, 33, 1, num_active_loras=1)],
            [Descriptor(Mode.FULL, 33, 1, num_ubatches=2)],
            [Descriptor(Mode.FULL, 33, 1), Descriptor(Mode.FULL, 40, 1)],
        ]
        for descs in cases:
            with self.subTest(descs=descs), self.assertRaises(ValueError):
                self.reshape(self.manager(descs))

    def test_nonadmitted_batch_keeps_the_native_nominal_plan(self):
        cfg = config()
        cfg.scheduler_config.max_num_seqs = 2
        manager = self.manager([Descriptor(Mode.FULL, 33, 1, uniform_token_count=33)], cfg)
        self.reshape(manager)
        planned = manager._capture_descs[Mode.FULL]
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0].uniform_token_count, 33)

    def test_dense_domino_full_plan_is_unchanged(self):
        cfg = legacy.config()
        sizes = cfg.compilation_config.cudagraph_capture_sizes
        manager = self.manager([Descriptor(Mode.FULL, n, 1) for n in sizes], cfg)
        self.reshape(manager)
        self.assertEqual([d.num_tokens for d in manager._capture_descs[Mode.FULL]], sizes)
        self.assertTrue(all(d.max_query_len == d.num_tokens and d.uniform_token_count is None for d in manager._capture_descs[Mode.FULL]))


if __name__ == "__main__":
    unittest.main()
