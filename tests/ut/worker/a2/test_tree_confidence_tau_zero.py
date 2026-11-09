# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run actual source methods/guards with stdlib-only tau-zero probes."""

import ast
import builtins
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "vllm_ascend/worker/v2/model_runner.py"
SPECULATOR = ROOT / "vllm_ascend/worker/v2/spec_decode/tree/speculator.py"
POLICY = ROOT / "vllm_ascend/worker/v2/spec_decode/tree/budget_policy.py"


def source_method(path, name, namespace=None):
    nodes = [
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(nodes) != 1:
        raise AssertionError(f"expected one real source method: {name}")
    scope = {"SchedulerOutput": object, "BatchReqState": object}
    scope.update(namespace or {})
    exec(compile(ast.Module(body=[nodes[0]], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def source_guard(path, method_name, marker, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == method_name)
    nodes = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr in ("_tree_confidence_requested", "_tree_budget_enabled")
        and marker in ast.unparse(node)
    ]
    if len(nodes) != 1:
        raise AssertionError(f"expected one actual guard in {method_name}: {marker}")
    shell = ast.parse("def check(self, scheduler_output=None, logits=None, draft_candidate_ids=None):\n    pass\n")
    shell.body[0].body = [copy.deepcopy(nodes[0])]
    ast.fix_missing_locations(shell)
    scope = dict(namespace or {})
    exec(compile(shell, str(path), "exec"), scope)
    return scope["check"]


def metadata_guard(namespace):
    tree = ast.parse(SPECULATOR.read_text(encoding="utf-8"))
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(child, ast.Assign) and "self._tree_confidence_metadata" in ast.unparse(child.targets)
            for child in ast.walk(node)
        )
    ]
    if len(methods) != 1:
        raise AssertionError("expected one real speculator confidence metadata producer")
    return source_guard(SPECULATOR, methods[0].name, "self._tree_confidence_metadata", namespace)


class WatchedState(SimpleNamespace):
    def __getattribute__(self, name):
        if name not in {"_reads", "_tree_confidence_requested", "_tree_budget_enabled", "__dict__", "__class__"}:
            object.__getattribute__(self, "_reads").append(name)
        return super().__getattribute__(name)


class ShapeProbe:
    def __init__(self, width=8):
        self.reads = []
        self.width = width

    @property
    def shape(self):
        self.reads.append("shape")
        return (1, 16, self.width)


class DominoModel:
    pass


class IndependentScorer:
    pass


class DominoScorer:
    pass


class LogitBiasState:
    pass


class PenaltiesState:
    pass


class BadWordsState:
    pass


class TestTauZeroRuntimeAdmission(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("tau_zero_actual_policy", POLICY)
        self.policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.policy)
        self.validate = source_method(RUNNER, "_validate_tree_confidence_runtime")
        model = DominoModel()
        model.config = SimpleNamespace(vocab_size=8, draft_vocab_size=8)
        model.draft_id_to_target_id = None
        model.logits_processor = SimpleNamespace(vocab_size=8, org_vocab_size=8)
        self.runner = WatchedState(
            _reads=[],
            _tree_confidence_requested=True,
            _tree_budget_enabled=True,
            speculator=SimpleNamespace(
                model=model,
                tree_builder=SimpleNamespace(scorer=IndependentScorer()),
                _tree_graphs={},
                enable_adaptive_verification=False,
            ),
            sampler=SimpleNamespace(
                logits_processors=[LogitBiasState(), PenaltiesState(), BadWordsState()],
                thinking_budget_state=SimpleNamespace(enabled=False),
                trace_replay_state=None,
            ),
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(async_scheduling=False),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                    prefill_context_parallel_size=1,
                    decode_context_parallel_size=1,
                ),
                max_concurrent_batches=1,
                model_config=SimpleNamespace(enforce_eager=True),
            ),
            model_state=SimpleNamespace(num_new_sampled_tokens_per_step=1),
            ascend_config=SimpleNamespace(
                finegrained_tp_config=SimpleNamespace(
                    oproj_tensor_parallel_size=0,
                    lmhead_tensor_parallel_size=0,
                    embedding_tensor_parallel_size=0,
                    mlp_tensor_parallel_size=0,
                )
            ),
            adaptive_verification=None,
        )
        bindings = {
            "vllm.sampling_params": {"SamplingType": SimpleNamespace(GREEDY=0)},
            "vllm.v1.worker.gpu.sample.sampler": {
                "LogitBiasState": LogitBiasState,
                "PenaltiesState": PenaltiesState,
                "BadWordsState": BadWordsState,
            },
            "vllm_ascend.models.qwen3_domino": {"AscendDominoQwen3ForCausalLM": DominoModel},
            "vllm_ascend.worker.v2.spec_decode.tree.scoring": {
                "IndependentScorer": IndependentScorer,
                "DominoScorer": DominoScorer,
            },
            "vllm_ascend.worker.v2.spec_decode.tree.budget_policy": {
                "admit_base_confidence_scorer": self.policy.admit_base_confidence_scorer,
                "validate_runtime_facts": self.policy.validate_runtime_facts,
            },
        }
        modules = {}
        for name, values in bindings.items():
            parts = name.split(".")
            for count in range(1, len(parts) + 1):
                parent = ".".join(parts[:count])
                if parent not in modules:
                    modules[parent] = ModuleType(parent)
                    modules[parent].__path__ = []
            modules[name].__dict__.update(values)
        context = patch.dict(sys.modules, modules)
        context.start()
        self.addCleanup(context.stop)

    def test_tau_zero_does_not_import_or_read_confidence_state(self):
        self.runner._tree_budget_enabled = False
        imports = []
        original = builtins.__import__

        def record(name, *args, **kwargs):
            imports.append(name)
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=record):
            self.validate(self.runner)
        self.assertEqual(imports, [], "tau0 entered actual confidence-only imports")
        self.assertEqual(self.runner._reads, [], "tau0 inspected confidence model/runtime state")

    def test_positive_threshold_still_establishes_vocab(self):
        self.validate(self.runner)
        self.assertEqual(self.runner.speculator._tree_full_vocab_size, 8)
        self.assertIn("speculator", self.runner._reads)

    def test_positive_threshold_still_rejects_async(self):
        self.runner.vllm_config.scheduler_config.async_scheduling = True
        with self.assertRaisesRegex(ValueError, "async_scheduling"):
            self.validate(self.runner)

    def test_positive_threshold_still_rejects_unadmitted_scorer(self):
        self.runner.speculator.tree_builder.scorer = object()
        with self.assertRaisesRegex(ValueError, "scorer"):
            self.validate(self.runner)


class TestTauZeroRunnerInputs(unittest.TestCase):
    def setUp(self):
        self.verified = source_method(RUNNER, "_tree_verified_counts")
        self.add_guard = source_guard(RUNNER, "add_requests", "_validate_tree_confidence_runtime")

    def test_tau_zero_verified_path_does_not_read_inputs(self):
        state = WatchedState(_reads=[], _tree_confidence_requested=True, _tree_budget_enabled=False)
        self.assertIsNone(self.verified(state, object(), object()))
        self.assertEqual(state._reads, [])

    def test_positive_verified_counts_keep_zero_and_batch_order(self):
        state = SimpleNamespace(
            _tree_budget_enabled=True,
            speculator=SimpleNamespace(budget=8),
            req_states=SimpleNamespace(tree_next_budget_cpu=[0, 3]),
        )
        output = SimpleNamespace(scheduled_spec_decode_tokens={"a": [-1, -1], "b": []})
        batch = SimpleNamespace(
            req_ids=["a", "b"], idx_mapping_np=[1, 0], is_prefilling_np=[False, False], num_scheduled_tokens=[3, 1]
        )
        self.assertEqual(self.verified(state, output, batch), [2, 0])
        state.req_states.tree_next_budget_cpu[1] = 1
        with self.assertRaisesRegex(ValueError, "available budget"):
            self.verified(state, output, batch)

    def test_tau_zero_add_requests_skips_sampling_admission(self):
        calls = []
        state = SimpleNamespace(
            _tree_confidence_requested=True,
            _tree_budget_enabled=False,
            _validate_tree_confidence_runtime=lambda *args: calls.append(args),
        )
        output = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(sampling_params="sample")])
        self.add_guard(state, scheduler_output=output)
        self.assertEqual(calls, [], "tau0 entered actual add_requests confidence guard")

    def test_positive_add_requests_checks_actual_sampling_params(self):
        calls = []
        state = SimpleNamespace(
            _tree_confidence_requested=True,
            _tree_budget_enabled=True,
            _validate_tree_confidence_runtime=lambda *args: calls.append(args),
        )
        output = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(sampling_params="sample")])
        self.add_guard(state, scheduler_output=output)
        self.assertEqual(calls, [(("sample",),)])


class TestTauZeroSpeculatorGuards(unittest.TestCase):
    def setUp(self):
        self.logs = []
        self.meta = metadata_guard(
            {
                "model": SimpleNamespace(config=SimpleNamespace(draft_vocab_size=8)),
                "get_ascend_config": lambda: SimpleNamespace(
                    tree_spec_config=SimpleNamespace(verification_threshold=0.0)
                ),
                "confidence_source_metadata": lambda **values: values,
                "logger": SimpleNamespace(info=lambda *args: self.logs.append(args)),
            }
        )
        self.logits = source_guard(SPECULATOR, "_run_tree_finalize", "confidence requires full base-vocabulary logits")

    def state(self, enabled):
        return WatchedState(
            _reads=[],
            _tree_confidence_requested=True,
            _tree_budget_enabled=enabled,
            draft_backend="dflash",
            _dsv4_dspark_draft=False,
            vocab_size=8,
            tree_builder=SimpleNamespace(scorer=SimpleNamespace(name="independent"), _frontier_width=lambda size: 2),
            _tree_confidence_source="base_full_vocab",
            _tree_full_vocab_size=8,
        )

    def test_tau_zero_metadata_does_not_read_model_or_builder(self):
        state = self.state(False)
        self.meta(state)
        self.assertEqual(state._reads, [], "tau0 entered actual metadata producer")
        self.assertEqual(self.logs, [])

    def test_positive_metadata_still_records_source(self):
        state = self.state(True)
        self.meta(state)
        self.assertEqual(state._tree_confidence_metadata["vocab_size"], 8)
        self.assertEqual(len(self.logs), 1)

    def test_tau_zero_logits_does_not_read_vocab_or_layout(self):
        state = self.state(False)
        logits = ShapeProbe()
        self.logits(state, logits=logits)
        self.assertEqual(logits.reads, [], "tau0 entered actual full-vocabulary logit guard")
        self.assertEqual(state._reads, [])

    def test_positive_logits_still_accepts_full_vocab(self):
        self.logits(self.state(True), logits=ShapeProbe())

    def test_positive_logits_still_rejects_wrong_vocab(self):
        with self.assertRaisesRegex(ValueError, "full base-vocabulary logits"):
            self.logits(self.state(True), logits=ShapeProbe(4))

    def test_positive_logits_still_rejects_candidate_mapping(self):
        with self.assertRaisesRegex(ValueError, "full base-vocabulary logits"):
            self.logits(self.state(True), logits=ShapeProbe(), draft_candidate_ids=object())


if __name__ == "__main__":
    unittest.main()
