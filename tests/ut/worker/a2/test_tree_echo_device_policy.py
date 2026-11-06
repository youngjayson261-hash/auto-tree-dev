# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU Torch differential tests; no model or NPU fixture is needed."""

import ast
import importlib.util
import inspect
import math
import random
import subprocess
import sys
from ctypes import c_float
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/tree/budget_policy.py"
_spec = importlib.util.spec_from_file_location("_echo_budget_policy_device_test", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
policy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = policy
_spec.loader.exec_module(policy)
select_device = policy.select_tree_prefix_device
SENTINEL = -77


def row(depths=(1, 1, 2, 2), parents=(0, 0, 1, 2), scores=(-0.1, -2.0), valid=None, invalid=False, bootstrap=False):
    return dict(
        depths=list(depths),
        parents=list(parents),
        scores=list(scores),
        valid=[True] * len(scores) if valid is None else list(valid),
        invalid=invalid,
        bootstrap=bootstrap,
    )


def make_case(rows, *, active_rows=None, capacity_rows=None, capacity=None, strided=False):
    """Mirror TreeLayout's three consumed fields and the real D03 summary."""
    r = len(rows) if active_rows is None else active_rows
    cr = len(rows) if capacity_rows is None else capacity_rows
    b = max((len(x["depths"]) for x in rows), default=0) if capacity is None else capacity
    d = max((len(x["scores"]) for x in rows), default=0)
    depths = torch.full((cr, b), -999, dtype=torch.int64, device="cpu")
    parents = torch.full((cr, b), -999, dtype=torch.int64, device="cpu")
    counts = torch.zeros(cr, dtype=torch.int32, device="cpu")
    maxima = torch.full((cr, d), -torch.inf, dtype=torch.float32, device="cpu")
    valid = torch.zeros((cr, d), dtype=torch.bool, device="cpu")
    invalid = torch.zeros(cr, dtype=torch.bool, device="cpu")
    bootstrap = torch.zeros(cr, dtype=torch.bool, device="cpu")
    oracle_rows = []
    for i, x in enumerate(rows):
        n, nd = len(x["depths"]), len(x["scores"])
        depths[i, :n] = torch.tensor(x["depths"], dtype=torch.int64, device="cpu")
        parents[i, :n] = torch.tensor(x["parents"], dtype=torch.int64, device="cpu")
        counts[i] = n
        maxima[i, :nd] = torch.tensor(x["scores"], dtype=torch.float32, device="cpu")
        valid[i, :nd] = torch.tensor(x["valid"], dtype=torch.bool, device="cpu")
        invalid[i], bootstrap[i] = x["invalid"], x["bootstrap"]
        oracle_rows.append(
            dict(
                depths=x["depths"],
                parents=x["parents"],
                max_path_logp=[c_float(v).value for v in x["scores"]] + [-math.inf] * (d - nd),
                valid_depths=x["valid"] + [False] * (d - nd),
                invalid_scores=x["invalid"],
                bootstrap=x["bootstrap"],
            )
        )
    if strided:
        storage = torch.full((cr * 2, 4), SENTINEL, dtype=torch.int32, device="cpu")
        packet = storage[::2, 1:3][:r]
    else:
        storage = torch.full((cr, 2), SENTINEL, dtype=torch.int32, device="cpu")
        packet = storage[:r]
    return SimpleNamespace(
        layout=SimpleNamespace(depths=depths, parents=parents, num_nodes=counts),
        summary=policy.TreeScoreSummary(maxima, valid, invalid),
        bootstrap=bootstrap,
        packet=packet,
        storage=storage,
        rows=oracle_rows[:r],
    )


def check(case, tau):
    expected = policy.select_tree_prefix_batch_reference(case.rows, policy="confidence", tau=tau)
    before = case.storage.clone()
    pointer, strides = case.packet.data_ptr(), case.packet.stride()
    assert select_device(case.layout, case.summary, case.packet, tau=tau, bootstrap_mask=case.bootstrap) is None
    assert (case.packet.data_ptr(), case.packet.stride()) == (pointer, strides)
    assert case.packet.tolist() == [[a, int(status)] for a, status in expected]
    changed = case.storage != before
    allowed = torch.zeros_like(case.storage, dtype=torch.bool)
    if case.packet.stride(0) == 2:
        allowed[: len(case.rows)] = True
    else:
        allowed[::2, 1:3][: len(case.rows)] = True
    assert not torch.any(changed & ~allowed)
    for original, (a, status) in zip(case.rows, expected):
        if status != policy.PolicyStatus.LAYOUT_ERROR:
            s, b = len(original["depths"]), case.layout.depths.shape[1]
            policy.validate_tree_counts(s, a, 0, b)
            policy.validate_tree_counts(s, a, a, b)
    return expected


@pytest.mark.parametrize("tau", [0.0, 0.5, 1.0, float.fromhex("0x0.0000000000001p-1022")])
def test_reference_parity_and_original_node_counts(tau):
    result = check(make_case([row()]), tau)
    if tau == 0.5:
        assert result == [(2, policy.PolicyStatus.CONFIDENCE)]


@pytest.mark.parametrize("direction, expected", [("below", 0), ("equal", 1), ("above", 1)])
def test_fp32_threshold_exact_equality(direction, expected):
    tau = 0.25
    boundary = torch.tensor(c_float(math.log(tau)).value, dtype=torch.float32, device="cpu")
    if direction != "equal":
        boundary = torch.nextafter(boundary, torch.tensor(-math.inf if direction == "below" else math.inf))
    result = check(make_case([row((1,), (0,), (boundary.item(),))]), tau)
    assert result == [(expected, policy.PolicyStatus.CONFIDENCE)]


@pytest.mark.parametrize("scores, expected", [((-2.0, -0.1, -0.1), 0), ((-0.1, -2.0, -0.1), 1), ((-math.inf,), 0)])
def test_first_failure_stops_consecutive_depths(scores, expected):
    n = len(scores)
    result = check(make_case([row(range(1, n + 1), range(n), scores)]), 0.5)
    assert result == [(expected, policy.PolicyStatus.CONFIDENCE)]


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.01])
def test_bad_score_falls_back_for_whole_active_confidence_batch(bad):
    rows = [row(scores=(bad, -2.0)), row(), row(invalid=True, bootstrap=True)]
    result = check(make_case(rows), 0.5)
    assert result == [
        (4, policy.PolicyStatus.SCORE_FALLBACK),
        (4, policy.PolicyStatus.SCORE_FALLBACK),
        (4, policy.PolicyStatus.BOOTSTRAP_FIXED),
    ]


def test_bootstrap_bad_scores_do_not_trigger_batch_fallback():
    result = check(make_case([row(), row(scores=(math.nan, math.inf), invalid=True, bootstrap=True)]), 0.5)
    assert result[0] == (2, policy.PolicyStatus.CONFIDENCE)
    assert result[1] == (4, policy.PolicyStatus.BOOTSTRAP_FIXED)


@pytest.mark.parametrize("valid", [(False, False), ()])
def test_missing_scores_fall_back_instead_of_becoming_layout_error(valid):
    scores = (-math.inf,) * len(valid)
    result = check(make_case([row(scores=scores, valid=valid), row()]), 0.5)
    assert result == [(4, policy.PolicyStatus.SCORE_FALLBACK)] * 2


@pytest.mark.parametrize(
    "bad_row",
    [
        row((1, 2), (0, 2), (-0.1, -0.2)),
        row((1, 2, 1), (0, 1, 0), (-0.1, -0.2)),
        row((1, 3), (0, 1), (-0.1, -0.2, -0.3)),
        row((0,), (0,), (-0.1,)),
        row((1,), (-1,), (-0.1,)),
        row(valid=(False, True)),
        row(scores=(-0.1,), valid=(True,)),
    ],
)
def test_layout_errors_match_reference(bad_row):
    result = check(make_case([bad_row]), 0.5)
    assert result == [(0, policy.PolicyStatus.LAYOUT_ERROR)]


@pytest.mark.parametrize("tau, bootstrap", [(0.0, False), (0.5, False), (0.5, True)])
def test_real_topology_error_precedes_fixed_bootstrap_and_bad_scores(tau, bootstrap):
    bad = row((1, 3), (0, 1), (math.nan, math.inf), invalid=True, bootstrap=bootstrap)
    result = check(make_case([bad, row()]), tau)
    assert result[0] == (0, policy.PolicyStatus.LAYOUT_ERROR)
    expected = (4, policy.PolicyStatus.FIXED) if tau == 0 else (2, policy.PolicyStatus.CONFIDENCE)
    assert result[1] == expected


@pytest.mark.parametrize("count", [-1, 5])
def test_invalid_num_nodes_returns_fatal_zero_budget(count):
    case = make_case([row()], capacity=4)
    case.layout.num_nodes[0] = count
    select_device(case.layout, case.summary, case.packet, tau=0.5, bootstrap_mask=case.bootstrap)
    assert case.packet.tolist() == [[0, int(policy.PolicyStatus.LAYOUT_ERROR)]]


@pytest.mark.parametrize("strided", [False, True])
def test_active_rows_only_and_packet_views(strided):
    case = make_case(
        [row(), row(bootstrap=True), row(scores=(math.nan, math.inf), invalid=True)],
        active_rows=2,
        capacity_rows=5,
        strided=strided,
    )
    if not strided:
        assert case.packet[:, 0].stride() == (2,)
    else:
        assert not case.packet.is_contiguous()
    check(case, 0.5)


@pytest.mark.parametrize("bootstrap", [False, True])
def test_root_only_and_zero_capacity(bootstrap):
    check(make_case([row((), (), (), bootstrap=bootstrap)], capacity=0), 0.5)


def test_no_active_requests_does_not_write_capacity_rows():
    check(make_case([row(invalid=True)], active_rows=0, capacity_rows=3), 0.5)


def test_int32_layout_and_original_buffers_are_preserved():
    case = make_case([row()])
    case.layout.depths = case.layout.depths.to(torch.int32)
    case.layout.parents = case.layout.parents.to(torch.int32)
    originals = (case.layout.depths, case.layout.parents, case.layout.num_nodes, *case.summary, case.bootstrap)
    copies = [value.clone() for value in originals]
    check(case, 0.5)
    assert all(torch.equal(value, before) for value, before in zip(originals, copies))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: setattr(c, "packet", c.packet.to(torch.int64)),
        lambda c: setattr(c, "packet", torch.empty((1, 3), dtype=torch.int32)),
        lambda c: setattr(c.layout, "depths", c.layout.depths.float()),
        lambda c: setattr(c.layout, "parents", c.layout.parents[:, :1]),
        lambda c: setattr(c.layout, "num_nodes", c.layout.num_nodes[:0]),
        lambda c: setattr(c, "summary", c.summary._replace(max_path_logp=c.summary.max_path_logp.double())),
        lambda c: setattr(c, "summary", c.summary._replace(valid_depths=c.summary.valid_depths[:, :1])),
        lambda c: setattr(c, "summary", c.summary._replace(invalid_scores=c.summary.invalid_scores.int())),
        lambda c: setattr(c, "bootstrap", c.bootstrap.int()),
    ],
)
def test_metadata_rejected_without_packet_writes(mutation):
    case = make_case([row()])
    before = case.storage.clone()
    mutation(case)
    with pytest.raises(ValueError):
        select_device(case.layout, case.summary, case.packet, tau=0.5, bootstrap_mask=case.bootstrap)
    assert torch.equal(case.storage, before)


@pytest.mark.parametrize("tau", [True, -0.1, 1.1, math.nan, math.inf])
def test_invalid_threshold_is_rejected_before_writes(tau):
    case = make_case([row()])
    with pytest.raises(ValueError):
        select_device(case.layout, case.summary, case.packet, tau=tau, bootstrap_mask=case.bootstrap)
    assert torch.all(case.packet == SENTINEL)


def test_deterministic_small_tree_differential():
    rng = random.Random(20261106)
    for _ in range(40):
        rows = []
        for _ in range(rng.randint(1, 5)):
            depths, parents, previous = [], [], [0]
            dmax = rng.randint(1, 4)
            for depth in range(1, dmax + 1):
                new_ids = []
                for _ in range(rng.randint(1, 3)):
                    depths.append(depth)
                    parents.append(rng.choice(previous))
                    new_ids.append(len(depths))
                previous = new_ids
            scores = [rng.uniform(-5.0, 0.0) for _ in range(dmax)]
            prefix = rng.randint(0, dmax)
            rows.append(
                row(
                    depths,
                    parents,
                    scores,
                    [i < prefix for i in range(dmax)],
                    invalid=rng.random() < 0.1,
                    bootstrap=rng.random() < 0.2,
                )
            )
        check(make_case(rows), rng.choice([0.0, 0.01, 0.5, 1.0]))


def test_selector_uses_no_host_tensor_readback(monkeypatch):
    case = make_case([row(), row(bootstrap=True)])

    def forbidden(*args, **kwargs):
        raise AssertionError("selector attempted a host tensor readback")

    with monkeypatch.context() as guarded:
        for name in ("item", "tolist", "cpu", "numpy", "__bool__"):
            guarded.setattr(torch.Tensor, name, forbidden)
        select_device(case.layout, case.summary, case.packet, tau=0.5, bootstrap_mask=case.bootstrap)
    assert case.packet.tolist() == [
        [2, int(policy.PolicyStatus.CONFIDENCE)],
        [4, int(policy.PolicyStatus.BOOTSTRAP_FIXED)],
    ]
    tree = ast.parse(inspect.getsource(select_device))
    assert not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "bool" for n in ast.walk(tree)
    )


def test_reference_module_remains_importable_without_torch():
    script = """
import importlib.util, sys
assert 'torch' not in sys.modules
spec = importlib.util.spec_from_file_location('_stdlib_echo_budget', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert module.select_tree_prefix_reference(depths=[1], parents=[0], max_path_logp=[-0.1],
    valid_depths=[True], invalid_scores=False, policy='confidence', tau=0.5,
    bootstrap=False) == (1, module.PolicyStatus.CONFIDENCE)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(MODULE_PATH)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
