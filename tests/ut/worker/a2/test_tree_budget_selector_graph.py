# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real selector math; fake graph backend tests lifecycle, not NPU execution."""

import ast
import gc
import importlib.util
import sys
import weakref
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

TREE_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/tree"
_package = ModuleType("_p20_tree_test")
_package.__path__ = [str(TREE_PATH)]
sys.modules[_package.__name__] = _package


def load(name):
    spec = importlib.util.spec_from_file_location(f"{_package.__name__}.{name}", TREE_PATH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


policy = load("budget_policy")
graph_module = load("budget_graph")
SelectorBinding = graph_module.SelectorBinding
TreeBudgetSelectorGraphRunner = graph_module.TreeBudgetSelectorGraphRunner


class FakeGraph:
    def __init__(self):
        self.operation = None
        self.failure = None

    def replay(self):
        self.operation()
        if self.failure:
            raise self.failure


class FakeBackend:
    """Record the reference call instead of pretending CPU has ACLGraph."""

    def __init__(self):
        self.supported = True
        self.active = False
        self.current = None
        self.captures = 0
        self.syncs = 0
        self.failure = None
        self.last_graph = None

    def supports(self, device):
        return self.supported

    def is_capturing(self):
        return self.active

    def new_graph(self):
        self.last_graph = FakeGraph()
        return self.last_graph

    @contextmanager
    def capture(self, graph, *, pool):
        assert not self.active
        self.active, self.current = True, graph
        self.captures += 1
        try:
            yield
            if self.failure:
                raise self.failure
        finally:
            self.active, self.current = False, None

    def synchronize(self, device):
        self.syncs += 1

    def memory(self, device):
        return 0, 0


def make_binding(rows=2, *, tau=0.1, device="cpu"):
    layout = SimpleNamespace(
        depths=torch.tensor([[1, 1, 2, 2]] * rows, dtype=torch.int32, device=device),
        parents=torch.tensor([[0, 0, 1, 2]] * rows, dtype=torch.int32, device=device),
        num_nodes=torch.full((rows,), 4, dtype=torch.int32, device=device),
    )
    summary = policy.TreeScoreSummary(
        torch.full((rows, 2), -1.0, device=device),
        torch.ones((rows, 2), dtype=torch.bool, device=device),
        torch.zeros(rows, dtype=torch.bool, device=device),
    )
    return SelectorBinding(
        layout,
        summary,
        torch.zeros(rows, dtype=torch.bool, device=device),
        torch.full((rows, 2), -77, dtype=torch.int32, device=device),
        tau,
    )


def reference(binding):
    expected = torch.empty_like(binding.packet)
    policy.select_tree_prefix_device(
        binding.layout, binding.summary, expected, tau=binding.tau, bootstrap_mask=binding.bootstrap_mask
    )
    return expected


@pytest.fixture
def case(monkeypatch):
    backend = FakeBackend()

    def record(*args, **kwargs):
        if backend.current is not None:
            backend.current.operation = lambda: policy.select_tree_prefix_device(*args, **kwargs)
        return policy.select_tree_prefix_device(*args, **kwargs)

    monkeypatch.setattr(graph_module, "select_tree_prefix_device", record)
    return TreeBudgetSelectorGraphRunner(backend=backend), make_binding(), backend


def test_replay_reads_new_values_and_has_no_hot_sync(case):
    runner, binding, backend = case
    before = binding.packet.clone()
    inputs = [t.clone() for t in binding.tensors()[:-1]]
    assert runner.prepare(binding, pool=None)
    assert torch.equal(binding.packet, before)
    assert all(torch.equal(a, b) for a, b in zip(binding.tensors()[:-1], inputs))
    syncs = backend.syncs
    for score in [-100.0, -1.0, -3.0, -0.1]:
        binding.summary.max_path_logp.fill_(score)
        expected = reference(binding)
        assert runner.try_replay(binding)
        assert torch.equal(binding.packet, expected)
    assert backend.captures == 1 and backend.syncs == syncs
    assert runner.diagnostics()["replays"] == 4


def test_repeated_prepare_and_new_wrappers_reuse_binding(case):
    runner, binding, backend = case
    assert runner.prepare(binding, pool=None)
    wrapped = replace(
        binding,
        layout=SimpleNamespace(**vars(binding.layout)),
        summary=policy.TreeScoreSummary(*binding.summary),
        packet=binding.packet.view_as(binding.packet),
    )
    assert runner.prepare(wrapped, pool=None)
    assert runner.try_replay(wrapped)
    assert backend.captures == 1


@pytest.mark.parametrize("index", range(8))
def test_each_tensor_address_change_invalidates_without_recapture(case, index):
    runner, binding, backend = case
    assert runner.prepare(binding, pool=None)
    values = list(binding.tensors())
    values[index] = values[index].clone()
    changed = replace(
        binding,
        layout=SimpleNamespace(depths=values[0], parents=values[1], num_nodes=values[2]),
        summary=policy.TreeScoreSummary(*values[3:6]),
        bootstrap_mask=values[6],
        packet=values[7],
    )
    assert not runner.try_replay(changed)
    assert not runner.try_replay(binding)
    assert backend.captures == 1
    assert runner.diagnostics()["entries"][2]["state"] == "STALE"


@pytest.mark.parametrize("change", ["shape", "stride", "dtype", "device", "tau"])
def test_metadata_changes_miss_before_replay(case, change):
    runner, binding, backend = case
    assert runner.prepare(binding, pool=None)
    if change == "shape":
        changed = replace(binding, bootstrap_mask=binding.bootstrap_mask[:1])
    elif change == "stride":
        changed = replace(binding, packet=binding.packet.t().contiguous().t())
    elif change == "dtype":
        changed = replace(binding, bootstrap_mask=binding.bootstrap_mask.to(torch.int32))
    elif change == "device":
        changed = replace(binding, packet=torch.empty_like(binding.packet, device="meta"))
    else:
        changed = replace(binding, tau=0.2)
    assert not runner.try_replay(changed)
    assert backend.captures == 1


def test_unknown_rows_zero_rows_and_tau_zero(case):
    runner, binding, backend = case
    assert not runner.try_replay(binding)
    assert runner.prepare(binding, pool=None)
    assert not runner.try_replay(make_binding(1))
    assert not runner.prepare(make_binding(0), pool=None)
    assert not runner.try_replay(make_binding(0))
    assert not runner.prepare(make_binding(tau=0), pool=None)
    assert not runner.try_replay(make_binding(tau=0))
    assert backend.captures == 1


def test_entry_holds_tensor_references(case):
    runner, binding, _ = case
    ref = weakref.ref(binding.layout.depths)
    assert runner.prepare(binding, pool=None)
    del case, binding
    gc.collect()
    assert ref() is not None


def test_explicit_invalidation_and_quiescent_reprepare(case):
    runner, binding, backend = case
    assert runner.prepare(binding, pool=None)
    runner.invalidate(reason="buffer rebuild", num_reqs=2)
    assert not runner.try_replay(binding)
    assert runner.prepare(binding, pool=None)
    assert runner.try_replay(binding)
    assert backend.captures == 2


def test_missing_backend_does_not_capture(case):
    runner, binding, backend = case
    backend.supported = False
    assert not runner.prepare(binding, pool=None)
    assert not runner.try_replay(binding)
    assert runner.diagnostics()["entries"][2]["state"] == "UNAVAILABLE"
    assert backend.captures == 0


@pytest.mark.parametrize("error", [NotImplementedError("unsupported"), RuntimeError("fatal")])
def test_capture_errors_restore_packet_and_only_unsupported_falls_back(case, error):
    runner, binding, backend = case
    before = binding.packet.clone()
    backend.failure = error
    if isinstance(error, NotImplementedError):
        assert not runner.prepare(binding, pool=None)
    else:
        with pytest.raises(RuntimeError, match="fatal"):
            runner.prepare(binding, pool=None)
    assert torch.equal(binding.packet, before)
    assert not runner.try_replay(binding)
    assert not backend.is_capturing()


def test_replay_partial_failure_propagates_without_eager_retry(case, monkeypatch):
    runner, binding, backend = case
    assert runner.prepare(binding, pool=None)
    backend.last_graph.failure = RuntimeError("replay failed after write")
    monkeypatch.setattr(graph_module, "select_tree_prefix_device", lambda *a, **k: pytest.fail("eager retry"))
    with pytest.raises(RuntimeError, match="replay failed after write"):
        runner.try_replay(binding)


def test_nested_capture_and_input_errors_are_not_capability_fallback(case):
    runner, binding, backend = case
    backend.active = True
    with pytest.raises(RuntimeError, match="captur"):
        runner.prepare(binding, pool=None)
    backend.active = False
    with pytest.raises(ValueError, match="packet_out"):
        runner.prepare(replace(binding, packet=binding.packet.to(torch.int64)), pool=None)
    assert not runner.try_replay(binding)


def test_failed_restore_is_fatal(case, monkeypatch):
    runner, binding, backend = case
    backend.failure = NotImplementedError("unsupported")
    original = torch.Tensor.copy_

    def fail_restore(destination, source, *args, **kwargs):
        if destination is binding.packet:
            raise RuntimeError("restore failed")
        return original(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_restore)
    with pytest.raises(RuntimeError, match="restore failed"):
        runner.prepare(binding, pool=None)
    assert not runner.try_replay(binding)


def make_runtime(runner, *, tau=0.1):
    source = TREE_PATH / "speculator.py"
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "_TreeBudgetRuntime"
    )
    namespace = dict(
        torch=torch,
        TreeScoreSummary=policy.TreeScoreSummary,
        validate_policy_config=policy.validate_policy_config,
        select_tree_prefix_device=policy.select_tree_prefix_device,
        SelectorBinding=SelectorBinding,
        TreeBudgetSelectorGraphRunner=lambda: runner,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    runtime = namespace["_TreeBudgetRuntime"](4, 2, torch.device("cpu"), tau=tau)
    if tau > 0:
        runtime._summary.max_path_logp.fill_(-1)
        runtime._summary.valid_depths.fill_(True)
        runtime._summary.invalid_scores.fill_(False)
    return runtime


def test_runtime_prepare_preserves_bootstrap_and_uses_active_rows(case):
    runner, binding, backend = case
    runtime = make_runtime(runner)
    packet = torch.full((4, 2), -9, dtype=torch.int32)
    before = packet.clone()
    assert runtime.prepare_selector_graph(binding.layout, packet, 2, pool=None)
    assert runtime._batch_rows is None
    assert not hasattr(runtime, "_bootstrap_key")
    assert runtime.bootstrap_mask.tolist() == [True] * 4
    assert torch.equal(packet, before)
    runtime.update_batch(np.zeros(2, dtype=np.bool_), 2)
    runtime.publish(binding.layout, packet[:2])
    assert packet.tolist() == [[4, int(policy.PolicyStatus.CONFIDENCE)]] * 2 + [[-9, -9]] * 2
    runtime._summary.max_path_logp.fill_(-100)
    runtime.publish(binding.layout, packet[:2])
    assert packet[:2, 0].tolist() == [0, 0]
    assert runner.diagnostics()["replays"] == 2 and backend.captures == 1


def test_runtime_unknown_rows_and_invalid_bindings_use_reference(case):
    runner, binding, _ = case
    runtime = make_runtime(runner)
    packet = torch.zeros((4, 2), dtype=torch.int32)
    assert runtime.prepare_selector_graph(binding.layout, packet, 2, pool=None)
    runtime.update_batch(np.zeros(1, dtype=np.bool_), 1)
    runtime.publish(binding.layout, packet[:1])
    assert packet[0, 0] == 4
    runtime.update_batch(np.zeros(2, dtype=np.bool_), 2)
    runtime.bootstrap_mask = runtime.bootstrap_mask.to(torch.int32)
    with pytest.raises(ValueError, match="bootstrap_mask"):
        runtime.publish(binding.layout, packet[:2])
    assert runner.diagnostics()["replays"] == 0


def test_runtime_stride_and_batch_checks_precede_replay(case):
    runner, binding, _ = case
    runtime = make_runtime(runner)
    packet = torch.zeros((4, 2), dtype=torch.int32)
    assert runtime.prepare_selector_graph(binding.layout, packet, 2, pool=None)
    runtime.update_batch(np.zeros(2, dtype=np.bool_), 2)
    with pytest.raises(ValueError, match="current active batch"):
        runtime.publish(binding.layout, packet[:1])
    with pytest.raises(ValueError, match="stride"):
        runtime.publish(binding.layout, packet[:2].t().contiguous().t())
    assert runner.diagnostics()["replays"] == 0


def test_runtime_tau_zero_retains_original_path(case):
    runner, binding, backend = case
    runtime = make_runtime(runner, tau=0)
    packet = torch.zeros((4, 2), dtype=torch.int32)
    assert not runtime.prepare_selector_graph(binding.layout, packet, 2, pool=None)
    runtime.update_batch(np.zeros(2, dtype=np.bool_), 2)
    runtime.publish(binding.layout, packet[:2])
    assert packet[:2].tolist() == [[4, int(policy.PolicyStatus.FIXED)]] * 2
    assert runtime._selector_graph_runner is None and backend.captures == 0
