# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run actual runner/policy bodies without importing torch or loading a model."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = load_file("parallel_actual_compat", HERE / "test_tree_confidence_runtime_compat.py")
policy = load_file("parallel_actual_policy", REPO / "vllm_ascend/worker/v2/spec_decode/tree/budget_policy.py")
FINE_FIELDS = (
    "oproj_tensor_parallel_size",
    "lmhead_tensor_parallel_size",
    "embedding_tensor_parallel_size",
    "mlp_tensor_parallel_size",
)


class TestConfidenceParallelAdmission(unittest.TestCase):
    def setUp(self):
        # Reuse import bindings, not their mocked policy decision: this class
        # always executes validate_runtime_facts from the production source.
        compat.TestTreeConfidenceRuntimeCompatibility.setUp(self)
        self.runner.vllm_config.parallel_config = SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        )
        self.runner.ascend_config = SimpleNamespace(
            finegrained_tp_config=SimpleNamespace(**dict.fromkeys(FINE_FIELDS, 0))
        )

    def record_facts(self, **facts):
        self.facts = facts
        policy.validate_runtime_facts(**facts)

    def test_normal_single_replica_and_disabled_module_parallelism_are_admitted(self):
        self.validate(self.runner)
        self.assertEqual(self.runner.speculator._tree_full_vocab_size, 8)

    def test_all_size_one_module_settings_are_admitted(self):
        self.runner.ascend_config.finegrained_tp_config = SimpleNamespace(**dict.fromkeys(FINE_FIELDS, 1))
        self.validate(self.runner)

    def test_real_runner_rejects_unadmitted_parallel_dimensions(self):
        for name in ("data_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
            with self.subTest(field=name):
                setattr(self.runner.vllm_config.parallel_config, name, 2)
                try:
                    with self.assertRaisesRegex(ValueError, "parallelism"):
                        self.validate(self.runner)
                finally:
                    setattr(self.runner.vllm_config.parallel_config, name, 1)

    def test_real_runner_rejects_each_multi_rank_module_setting(self):
        fine = self.runner.ascend_config.finegrained_tp_config
        for name in FINE_FIELDS:
            with self.subTest(field=name):
                setattr(fine, name, 2)
                try:
                    with self.assertRaisesRegex(ValueError, "finegrained_tp"):
                        self.validate(self.runner)
                finally:
                    setattr(fine, name, 0)

    def test_missing_parallel_metadata_is_not_assumed_single_replica(self):
        parallel = self.runner.vllm_config.parallel_config
        for name in ("data_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
            with self.subTest(field=name):
                delattr(parallel, name)
                try:
                    with self.assertRaisesRegex(ValueError, "parallelism"):
                        self.validate(self.runner)
                finally:
                    setattr(parallel, name, 1)

    def test_missing_finegrained_metadata_is_not_assumed_disabled(self):
        del self.runner.ascend_config
        with self.assertRaisesRegex(ValueError, "finegrained_tp"):
            self.validate(self.runner)

    def test_mutated_noninteger_or_negative_dimensions_are_rejected(self):
        for value in (True, False, None, "1", -1, 1.0):
            with self.subTest(value=value):
                self.runner.vllm_config.parallel_config.data_parallel_size = value
                with self.assertRaisesRegex(ValueError, "parallelism"):
                    self.validate(self.runner)
        self.runner.vllm_config.parallel_config.data_parallel_size = 1
        fine = self.runner.ascend_config.finegrained_tp_config
        for value in (True, None, "1", -1, 1.0):
            with self.subTest(fine_value=value):
                fine.oproj_tensor_parallel_size = value
                with self.assertRaisesRegex(ValueError, "finegrained_tp"):
                    self.validate(self.runner)

    def test_disabled_and_tau_zero_do_not_read_any_new_metadata(self):
        class Forbidden:
            def __getattribute__(self, name):
                raise AssertionError("disabled confidence read parallel metadata")

        for requested in (False, True):
            with self.subTest(requested=requested):
                self.runner._tree_budget_enabled = False
                self.runner._tree_confidence_requested = requested
                self.runner.vllm_config = Forbidden()
                self.runner.ascend_config = Forbidden()
                self.validate(self.runner)
                self.assertIsNone(self.facts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
