# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Actual repo Host counters/runner bodies, with explicitly local runtime doubles.

The local ABC is a test double, not proof of paired runtime ABC compatibility.
Selected runner/handler bodies execute without their runtime decorators; native
imports, device kernels, actual executor and performance remain separate tests.
"""

import abc
import ast
import importlib.util
import logging
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "vllm_ascend/worker/v2"
PACKAGE = ModuleType("_tree_budget_host_test")
PACKAGE.__path__ = [str(SOURCE / "spec_decode/tree")]
sys.modules[PACKAGE.__name__] = PACKAGE


def load(name):
    path = SOURCE / "spec_decode/tree" / (name + ".py")
    spec = importlib.util.spec_from_file_location(PACKAGE.__name__ + "." + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    with patch.dict(sys.modules, {"vllm.logger": logger_module}):
        spec.loader.exec_module(module)
    return module


load("budget_policy")
metrics = load("budget_metrics")


class HostAsyncOutput(abc.ABC):
    @abc.abstractmethod
    def get_output(self):
        pass


native_abc = HostAsyncOutput


def node(path, name):
    return next(
        n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef) and n.name == name
    )


def execute(nodes, namespace, path):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


output_path = SOURCE / "spec_decode/tree/budget_output.py"
TreeBudgetOutput = execute(
    [node(output_path, "TreeBudgetOutput")], {"AsyncModelRunnerOutput": HostAsyncOutput}, output_path
)["TreeBudgetOutput"]


class Spec(SimpleNamespace):
    pass


class Base:
    def sample_tokens(self, grammar_output):
        self.calls.append("native_sample")
        if self.native_error is not None:
            raise self.native_error
        return self.native_output

    def _remove_request(self, req_id):
        self.calls.append(("native_remove", req_id))
        return self.remove_result


runner_path = SOURCE / "model_runner.py"
original = node(runner_path, "NPUModelRunner")
selected = [
    n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in ("sample_tokens", "_remove_request")
]
for method in selected:
    method.decorator_list = []
candidate_class = ast.ClassDef(
    name="ActualRunner", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=selected, decorator_list=[]
)
ast.fix_missing_locations(candidate_class)
Runner = execute(
    [candidate_class],
    {
        "Base": Base,
        "lmhead_tp_enable": lambda: False,
        "AscendPCPManager": object,
        "AscendTreeSpeculator": Spec,
        "AsyncModelRunnerOutput": HostAsyncOutput,
        "TreeBudgetOutput": TreeBudgetOutput,
    },
    runner_path,
)["ActualRunner"]


class Poison:
    def __getattr__(self, name):
        raise AssertionError("collector read a device/unavailable input: " + name)

    def __iter__(self):
        raise AssertionError("disabled collector iterated input")


class HostMetricsTests(unittest.TestCase):
    def setUp(self):
        self.previous = metrics.logger.level
        metrics.logger.setLevel(logging.DEBUG)
        self.addCleanup(metrics.logger.setLevel, self.previous)
        self.stats = metrics.TreeBudgetStats()

    def test_matched_non_root_counts_and_separate_produced_cohort(self):
        self.stats.record_produced(6, 3)
        self.stats.record_published(6, [("a", 4, 1), ("b", 0, 1)])
        cohort = self.stats.snapshot_consumption(["a", "b"], [3, 0])
        self.stats.record_verified(cohort, ["a", "b"], [[10, 11, 12], [20]])
        result = self.stats.snapshot()
        self.assertEqual(result["produced_tree_non_root"], 18)
        self.assertEqual(result["delivered_s_non_root"], 12)
        self.assertEqual(
            result["matched_consumptions"],
            {"draft_non_root": 12, "published_non_root": 4, "verified_non_root": 3, "accepted_non_root": 2},
        )
        self.assertEqual(result["raw_sampler_accepted_non_root"], 2)
        self.assertEqual(result["unconsumed_deliveries"], 0)
        self.assertEqual(result["invalid_metric_rows"], 0)

    def test_reordered_host_output_uses_request_identity(self):
        self.stats.record_published(6, [("a", 4, 1), ("b", 2, 1)])
        cohort = self.stats.snapshot_consumption(["a", "b"], [3, 1])
        self.stats.record_verified(cohort, ["b", "a"], [[9], [1, 2, 3]])
        result = self.stats.snapshot()
        self.assertEqual(result["verified_non_root"], 4)
        self.assertEqual(result["raw_sampler_accepted_non_root"], 2)
        self.assertEqual(result["matched_requests"], 2)
        self.assertEqual(result["invalid_metric_rows"], 0)

    def test_unconsumed_other_request_survives_partial_consume(self):
        self.stats.record_published(6, [("a", 4, 1), ("b", 2, 1)])
        self.stats.record_verified(self.stats.snapshot_consumption(["a"], [3]), ["a"], [[1, 2]])
        self.assertEqual(self.stats.snapshot()["unconsumed_a_non_root"], 2)
        self.stats.record_verified(self.stats.snapshot_consumption(["b"], [0]), ["b"], [[9]])
        self.assertEqual(self.stats.snapshot()["unconsumed_deliveries"], 0)
        self.assertEqual(self.stats.snapshot()["matched_requests"], 2)

    def test_terminal_unconsumed_delivery_is_reported_as_discard_not_accepted(self):
        self.stats.record_produced(6, 1)
        self.stats.record_published(6, [("terminal", 4, 1)])
        self.stats.discard("terminal")
        self.stats.discard("terminal")
        result = self.stats.snapshot()
        self.assertEqual(result["discarded_publications"], 1)
        self.assertEqual(result["discarded_s_non_root"], 6)
        self.assertEqual(result["discarded_a_non_root"], 4)
        self.assertEqual(result["raw_sampler_accepted_non_root"], 0)
        self.assertEqual(result["matched_requests"], 0)

    def test_overwritten_unconsumed_budget_cannot_supply_new_consumption(self):
        self.stats.record_published(6, [("a", 4, 1)])
        self.stats.record_published(6, [("a", 1, 1)])
        self.stats.record_verified(self.stats.snapshot_consumption(["a"], [1]), ["a"], [[7, 8]])
        result = self.stats.snapshot()
        self.assertEqual(result["matched_consumptions"]["published_non_root"], 1)
        self.assertEqual(result["discarded_a_non_root"], 4)
        self.assertEqual(result["discarded_publications"], 1)

    def test_score_fallback_batch_is_counted_once_for_multiple_rows(self):
        self.stats.record_published(6, [("a", 6, 3), ("b", 6, 3), ("bootstrap", 6, 2)])
        result = self.stats.snapshot()
        self.assertEqual(result["score_fallback_batches"], 1)
        self.assertEqual(result["status_requests"], {3: 2, 2: 1})

    def test_bootstrap_unmatched_and_invalid_rows_are_diagnostics_not_admission(self):
        self.stats.record_verified(self.stats.snapshot_consumption(["first"], [0]), ["first"], [[5]])
        self.assertEqual(self.stats.snapshot()["unmatched_verifications"], 1)
        self.stats.record_published(6, [("bad", 1, 1)])
        self.stats.record_verified(self.stats.snapshot_consumption(["bad"], [3]), ["bad"], [[1, 2]])
        self.assertEqual(self.stats.snapshot()["invalid_metric_rows"], 1)

    def test_debug_off_does_not_read_inputs_or_enable_other_module_logs(self):
        other = logging.getLogger("p17_unrelated_module_test")
        other.setLevel(logging.INFO)
        self.addCleanup(other.setLevel, logging.NOTSET)
        self.assertFalse(other.isEnabledFor(logging.DEBUG))
        metrics.logger.setLevel(logging.INFO)
        self.stats.record_produced(Poison(), Poison())
        self.stats.record_published(Poison(), Poison())
        self.stats.record_verified(Poison(), Poison(), Poison())
        self.assertEqual(self.stats.snapshot_consumption(Poison(), Poison()), ())
        self.assertEqual(self.stats.snapshot()["produced_tree_non_root"], 0)
        metrics.logger.setLevel(logging.DEBUG)
        self.assertTrue(self.stats.enabled)
        self.assertFalse(other.isEnabledFor(logging.DEBUG))

    def test_output_adapter_delegates_once_then_counts_existing_host_ids(self):
        self.stats.record_published(6, [("a", 4, 1)])
        host = SimpleNamespace(req_ids=["a"], sampled_token_ids=[[1, 2, 3]])
        calls = []

        class NativeOutput:
            device_value = Poison()

            def get_output(self):
                calls.append("native_get_output")
                return host

        output = TreeBudgetOutput(NativeOutput(), self.stats, self.stats.snapshot_consumption(["a"], [3]))
        self.assertIs(output.get_output(), host)
        self.assertEqual(calls, ["native_get_output"])
        self.assertEqual(self.stats.snapshot()["raw_sampler_accepted_non_root"], 2)

    def test_original_output_exception_is_preserved_without_counting(self):
        sentinel = RuntimeError("native output failure")

        class NativeOutput:
            def get_output(self):
                raise sentinel

        output = TreeBudgetOutput(NativeOutput(), self.stats, ())
        with self.assertRaises(RuntimeError) as raised:
            output.get_output()
        self.assertIs(raised.exception, sentinel)
        self.assertEqual(self.stats.snapshot()["verification_requests"], 0)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        previous = metrics.logger.level
        metrics.logger.setLevel(logging.DEBUG)
        self.addCleanup(metrics.logger.setLevel, previous)
        self.stats = metrics.TreeBudgetStats()

    def test_repeating_same_consumption_does_not_count_raw_totals_twice(self):
        self.stats.record_published(6, [("a", 4, 1)])
        cohort = self.stats.snapshot_consumption(["a"], [3])
        self.stats.record_verified(cohort, ["a"], [[1, 2, 3]])
        self.stats.record_verified(cohort, ["a"], [[1, 2, 3]])
        result = self.stats.snapshot()
        self.assertEqual(result["verified_non_root"], 3)
        self.assertEqual(result["raw_sampler_accepted_non_root"], 2)
        self.assertEqual(result["verification_requests"], 1)
        self.assertEqual(result["invalid_metric_rows"], 1)

    def test_mismatched_snapshot_lengths_are_unknown_not_truncated(self):
        self.stats.record_published(6, [("a", 4, 1), ("b", 2, 1)])
        self.assertEqual(self.stats.snapshot_consumption(["a", "b"], [3]), ())
        self.assertEqual(self.stats.snapshot()["invalid_metric_rows"], 1)

    def test_duplicate_delivery_identity_does_not_count_as_two_publications(self):
        self.stats.record_published(6, [("a", 4, 3), ("a", 4, 3)])
        result = self.stats.snapshot()
        self.assertEqual(result["published_requests"], 0)
        self.assertEqual(result["unconsumed_deliveries"], 0)
        self.assertEqual(result["score_fallback_batches"], 0)
        self.assertEqual(result["invalid_metric_rows"], 1)

    def test_mismatched_output_rows_are_unknown_not_partially_counted(self):
        self.stats.record_published(6, [("a", 4, 1), ("b", 2, 1)])
        cohort = self.stats.snapshot_consumption(["a", "b"], [3, 1])
        self.stats.record_verified(cohort, ["a", "b"], [[1, 2, 3]])
        self.assertEqual(self.stats.snapshot()["verification_requests"], 0)
        self.assertEqual(self.stats.snapshot()["invalid_metric_rows"], 1)


class SeamTests(unittest.TestCase):
    def setUp(self):
        previous = metrics.logger.level
        metrics.logger.setLevel(logging.DEBUG)
        self.addCleanup(metrics.logger.setLevel, previous)
        self.stats = metrics.TreeBudgetStats()

    def runner(self, output):
        value = Runner()
        value.calls = []
        value.pcp_manager = None
        value.is_last_pp_rank = True
        value.native_output = output
        value.native_error = None
        value.remove_result = object()
        fields = ("depths", "parents", "visibility", "num_nodes", "first_child", "next_sibling")
        value.speculator = Spec(
            tree_budget_stats=self.stats,
            tree=SimpleNamespace(**{k: object() for k in fields}),
            input_batch=SimpleNamespace(idx_mapping="slot"),
        )
        value.req_states = SimpleNamespace(**{"tree_" + k: {} for k in fields})
        value.draft_tokens_handler = SimpleNamespace(seal_budget_production=lambda: value.calls.append("seal"))
        value._restore_replicated_draft_target_states = lambda: value.calls.append("restore")
        value._finalize_dump_data = lambda: value.calls.append("finalize")
        value.execute_model_state = SimpleNamespace(
            input_batch=SimpleNamespace(req_ids=["a"], tree_verified_counts_cpu=[3])
        )
        return value

    def test_runner_reuses_host_output_once_for_statistics(self):
        self.stats.record_published(6, [("a", 4, 1)])
        host = SimpleNamespace(req_ids=["a"], sampled_token_ids=[[10, 151645, 11, 12]])
        calls = []

        class Native(native_abc):
            def get_output(self):
                calls.append("original_get_output")
                return host

        value = self.runner(Native())
        output = value.sample_tokens(None)
        self.assertIsInstance(output, native_abc)
        self.assertEqual(calls, [])
        self.assertEqual(value.calls, ["restore", "native_sample", "seal", "finalize"])
        self.assertIs(output.get_output(), host)
        self.assertIs(output.get_output(), host)
        self.assertEqual(calls, ["original_get_output", "original_get_output"])
        self.assertEqual(self.stats.snapshot()["verification_requests"], 1)
        self.assertEqual(self.stats.snapshot()["raw_sampler_accepted_non_root"], 3)

    def test_repeated_native_exception_is_not_swallowed_by_reported_flag(self):
        host = SimpleNamespace(req_ids=["a"], sampled_token_ids=[[1]])
        sentinel = RuntimeError("native second-call failure")

        class Native(native_abc):
            calls = 0

            def get_output(self):
                self.calls += 1
                if self.calls == 2:
                    raise sentinel
                return host

        output = TreeBudgetOutput(Native(), self.stats, self.stats.snapshot_consumption(["a"], [0]))
        self.assertIs(output.get_output(), host)
        with self.assertRaises(RuntimeError) as caught:
            output.get_output()
        self.assertIs(caught.exception, sentinel)
        self.assertEqual(self.stats.snapshot()["verification_requests"], 1)

    def test_disabled_actual_runner_skips_host_inputs_wrapper_and_metadata(self):
        metrics.logger.setLevel(logging.INFO)
        self.stats._delivered = Poison()
        host = object()
        value = self.runner(host)
        value.execute_model_state = SimpleNamespace(input_batch=Poison())
        self.assertIs(value.sample_tokens(None), host)
        self.assertEqual(value.calls, ["restore", "native_sample", "seal", "finalize"])
        self.assertIs(value._remove_request("a"), value.remove_result)
        self.assertEqual(value.calls[-1], ("native_remove", "a"))

    def test_original_sample_exception_preserves_identity(self):
        value = self.runner(object())
        sentinel = RuntimeError("native sample failure")
        value.native_error = sentinel
        with self.assertRaises(RuntimeError) as caught:
            value.sample_tokens(None)
        self.assertIs(caught.exception, sentinel)
        self.assertEqual(value.calls, ["restore", "native_sample"])

    def test_actual_remove_clears_only_diagnostic_publication_once(self):
        self.stats.record_published(6, [("a", 4, 1), ("b", 2, 1)])
        value = self.runner(object())
        self.assertIs(value._remove_request("a"), value.remove_result)
        value._remove_request("a")
        result = self.stats.snapshot()
        self.assertEqual(result["discarded_publications"], 1)
        self.assertEqual(result["unconsumed_a_non_root"], 2)

    def test_actual_builder_host_count_hook_uses_selected_work_not_budget(self):
        builder_path = REPO / "vllm_ascend/worker/v2/spec_decode/tree/builder.py"
        parsed = ast.parse(builder_path.read_text(encoding="utf-8"))
        assignments = [
            n
            for n in ast.walk(parsed)
            if isinstance(n, ast.Assign)
            and (
                isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "num_nodes"
                and isinstance(n.value, ast.Call)
                and (getattr(n.value.func, "id", None) == "min")
                or (
                    isinstance(n.targets[0], ast.Attribute)
                    and n.targets[0].attr == "num_selected_nodes"
                    and isinstance(n.value, ast.Name)
                    and (n.value.id == "num_nodes")
                )
            )
        ]
        builder = SimpleNamespace(budget=48)
        execute(
            sorted(assignments, key=lambda n: n.lineno),
            {"self": builder, "state": SimpleNamespace(num_nodes=7)},
            builder_path,
        )
        self.assertEqual(builder.num_selected_nodes, 7)
        path = SOURCE / "spec_decode/tree/speculator.py"
        hook = next(
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.If)
            and any(
                isinstance(x, ast.Call)
                and isinstance(x.func, ast.Attribute)
                and (x.func.attr == "record_builder_produced")
                for x in ast.walk(n)
            )
            and isinstance(n.test, ast.BoolOp)
            and ("tree_budget_stats.enabled" in ast.unparse(n.test))
        )
        execute(
            [hook],
            {
                "self": SimpleNamespace(tree_builder=builder, tree_budget_stats=self.stats, tree=Poison()),
                "input_batch": SimpleNamespace(num_reqs=2),
            },
            path,
        )
        self.assertEqual(self.stats.snapshot()["produced_tree_non_root"], 14)


class DiagnosticExceptionContract(unittest.TestCase):
    def setUp(self):
        self.old_level = metrics.logger.level
        self.old_handlers = list(metrics.logger.handlers)
        self.old_propagate = metrics.logger.propagate
        metrics.logger.setLevel(logging.DEBUG)
        metrics.logger.propagate = False
        metrics.logger.handlers = []
        self.addCleanup(self.restore_logger)
        self.stats = metrics.TreeBudgetStats()
        self.stats.record_published(6, [("a", 4, 1)])
        self.host = SimpleNamespace(req_ids=["a"], sampled_token_ids=[[10, 11, 12]])

    def restore_logger(self):
        metrics.logger.level = self.old_level
        metrics.logger.handlers = self.old_handlers
        metrics.logger.propagate = self.old_propagate

    def failing_diagnostic_logger(self):
        class FailingHandler(logging.Handler):
            def emit(self, record):
                raise RuntimeError("independent diagnostic logging failure")

        metrics.logger.addHandler(FailingHandler())

    def test_successful_native_async_output_survives_diagnostic_logging_failure(self):
        self.failing_diagnostic_logger()
        host, calls = (self.host, [])

        class Native(native_abc):
            def get_output(self):
                calls.append("original_get_output")
                return host

        wrapped = TreeBudgetOutput(Native(), self.stats, self.stats.snapshot_consumption(["a"], [3]))
        self.assertIs(wrapped.get_output(), host)
        self.assertEqual(calls, ["original_get_output"])

    def test_successful_native_sync_output_and_finalize_survive_diagnostic_logging_failure(self):
        self.failing_diagnostic_logger()
        fixture = SeamTests()
        fixture.stats = self.stats
        runner = fixture.runner(self.host)
        self.assertIs(runner.sample_tokens(None), self.host)
        self.assertEqual(runner.calls, ["restore", "native_sample", "seal", "finalize"])

    def test_original_async_exception_still_propagates_by_identity(self):
        self.failing_diagnostic_logger()
        sentinel = RuntimeError("original native failure")

        class Native(native_abc):
            def get_output(self):
                raise sentinel

        wrapped = TreeBudgetOutput(Native(), self.stats, ())
        with self.assertRaises(RuntimeError) as caught:
            wrapped.get_output()
        self.assertIs(caught.exception, sentinel)


class DiagnosticPoison:
    def __getattr__(self, name):
        raise RuntimeError("diagnostic Host descriptor failure")

    def __iter__(self):
        raise RuntimeError("diagnostic Host iterator failure")

    def __len__(self):
        raise RuntimeError("diagnostic Host length failure")


seams = SimpleNamespace(SOURCE=SOURCE, SeamTests=SeamTests)


class DiagnosticIsolationTests(unittest.TestCase):
    def setUp(self):
        self.level, self.handlers, self.propagate = (
            metrics.logger.level,
            metrics.logger.handlers,
            metrics.logger.propagate,
        )
        metrics.logger.setLevel(logging.DEBUG)
        metrics.logger.handlers, metrics.logger.propagate = ([], False)
        self.addCleanup(self.restore)
        self.stats = metrics.TreeBudgetStats()

    def restore(self):
        metrics.logger.setLevel(self.level)
        metrics.logger.handlers, metrics.logger.propagate = (self.handlers, self.propagate)

    def assert_unknown(self):
        snapshot = self.stats.snapshot()
        self.assertIs(snapshot["diagnostic_unknown"], True)
        self.assertGreater(snapshot["diagnostic_errors"], 0)
        self.assertIs(snapshot["metrics_comparable"], False)
        self.assertEqual(snapshot["last_diagnostic_error_type"], "RuntimeError")

    def test_all_new_descriptor_and_collector_failures_are_contained(self):
        checks = [
            lambda: self.stats.record_builder_produced(DiagnosticPoison(), 1),
            lambda: self.stats.record_fixed_delivery(DiagnosticPoison(), DiagnosticPoison(), 0),
            lambda: self.stats.record_budget_delivery(DiagnosticPoison(), DiagnosticPoison(), [], []),
            lambda: self.stats.capture_consumption(DiagnosticPoison()),
            lambda: self.stats.record_published(6, DiagnosticPoison()),
            lambda: self.stats.snapshot_consumption(DiagnosticPoison(), []),
            lambda: self.stats.record_host_output((), DiagnosticPoison()),
            lambda: self.stats.record_verified(DiagnosticPoison(), [], []),
        ]
        for check in checks:
            with self.subTest(check=check):
                self.assertIsNone(check())
                self.assert_unknown()
        self.assertEqual(self.stats.diagnostic_errors, len(checks))

    def test_snapshot_failure_still_runs_native_sample_and_finalize(self):
        fixture = seams.SeamTests()
        fixture.stats = self.stats
        host = SimpleNamespace(req_ids=["a"], sampled_token_ids=[[1]])
        runner = fixture.runner(host)
        runner.execute_model_state = DiagnosticPoison()
        self.assertIs(runner.sample_tokens(None), host)
        self.assertEqual(runner.calls, ["restore", "native_sample", "seal", "finalize"])
        self.assert_unknown()

    def test_host_output_field_failure_still_returns_exact_native_output(self):
        fixture = seams.SeamTests()
        fixture.stats = self.stats
        runner = fixture.runner(DiagnosticPoison())
        self.assertIs(runner.sample_tokens(None), runner.native_output)
        self.assertEqual(runner.calls[-1], "finalize")
        self.assert_unknown()

    def test_diagnostic_fault_does_not_replace_later_native_exception(self):
        fixture = seams.SeamTests()
        fixture.stats = self.stats
        runner = fixture.runner(None)
        runner.execute_model_state = DiagnosticPoison()
        sentinel = RuntimeError("native failure")
        runner.native_error = sentinel
        with self.assertRaises(RuntimeError) as error:
            runner.sample_tokens(None)
        self.assertIs(error.exception, sentinel)
        self.assertEqual(runner.calls, ["restore", "native_sample"])
        self.assert_unknown()

    def test_enabled_property_failure_is_sticky_without_logging_recursion(self):
        original = metrics.logger.isEnabledFor

        def broken(*args):
            raise RuntimeError("logger metadata failure")

        metrics.logger.isEnabledFor = broken
        try:
            self.assertIs(self.stats.enabled, False)
        finally:
            metrics.logger.isEnabledFor = original
        self.assert_unknown()
        self.stats.record_produced(6, 1)
        self.assert_unknown()

    def test_discard_failure_cannot_replace_native_remove_result(self):
        fixture = seams.SeamTests()
        fixture.stats = self.stats
        runner = fixture.runner(None)
        self.stats._delivered = DiagnosticPoison()
        self.assertIs(runner._remove_request("a"), runner.remove_result)
        self.assertIs(self.stats.diagnostic_unknown, True)
        self.assertEqual(self.stats.diagnostic_errors, 1)

    def test_debug_off_avoids_new_metadata_traversal(self):
        metrics.logger.setLevel(logging.INFO)
        self.stats.capture_consumption(DiagnosticPoison())
        self.stats.record_host_output(DiagnosticPoison(), DiagnosticPoison())
        self.stats.record_fixed_delivery(DiagnosticPoison(), DiagnosticPoison(), 0)
        self.stats.record_budget_delivery(
            DiagnosticPoison(), DiagnosticPoison(), DiagnosticPoison(), DiagnosticPoison()
        )
        self.stats.record_builder_produced(DiagnosticPoison(), DiagnosticPoison())
        self.assertEqual(self.stats.diagnostic_errors, 0)
        self.assertIs(self.stats.diagnostic_unknown, False)

    def test_actual_fixed_handler_native_return_and_exception_unchanged(self):
        path = seams.SOURCE / "spec_decode/tree/draft_tokens.py"
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        owner = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "TreeDraftTokensHandler")
        method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "get_draft_tokens")
        method.returns = None
        method.decorator_list = []

        class Base:
            def get_draft_tokens(self):
                self.calls += 1
                if self.native_error is not None:
                    raise self.native_error
                return self.native_result

        cls = ast.ClassDef(
            name="ActualHandler",
            bases=[ast.Name(id="Base", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        ast.fix_missing_locations(cls)
        namespace = {"Base": Base, "PolicyStatus": SimpleNamespace(FIXED=0)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        handler = namespace["ActualHandler"]()
        handler.calls, handler.native_error = (0, None)
        handler.native_result = DiagnosticPoison()
        handler._budget_enabled, handler._budget_stats = (False, self.stats)
        handler.speculator = SimpleNamespace(tree_builder=DiagnosticPoison())
        self.assertIs(handler.get_draft_tokens(), handler.native_result)
        self.assertEqual(handler.calls, 1)
        self.assert_unknown()
        sentinel = RuntimeError("native handler failure")
        handler.native_error = sentinel
        with self.assertRaises(RuntimeError) as error:
            handler.get_draft_tokens()
        self.assertIs(error.exception, sentinel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
