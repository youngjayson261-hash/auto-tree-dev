# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def make_runner(budgets):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner._tree_budget_enabled = True
    runner.speculator = SimpleNamespace(budget=8)
    runner.req_states = SimpleNamespace(tree_next_budget_cpu=budgets)
    return runner


def batch(prefill=(False, False), scheduled=(3, 1)):
    return SimpleNamespace(
        req_ids=["a", "b"],
        idx_mapping_np=np.array([1, 0]),
        is_prefilling_np=np.array(prefill),
        num_scheduled_tokens=np.array(scheduled),
    )


def test_verified_counts_follow_batch_order_and_snapshot():
    runner = make_runner([0, 5])
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={"a": [-1, -1], "b": []})
    counts = runner._tree_verified_counts(scheduled, batch())
    assert counts == [2, 0]
    runner.req_states.tree_next_budget_cpu[:] = [8, 8]
    assert counts == [2, 0]


@pytest.mark.parametrize("budgets", [[0, 1], [0, 9], [0, None]])
def test_oversized_or_missing_decode_budget_rejected(budgets):
    runner = make_runner(budgets)
    with pytest.raises(ValueError, match="budget"):
        runner._tree_verified_counts(SimpleNamespace(scheduled_spec_decode_tokens={"a": [1, 2]}), batch())


def test_prefill_bootstrap_and_all_zero():
    runner = make_runner([0, None])
    output = SimpleNamespace(scheduled_spec_decode_tokens={})
    assert runner._tree_verified_counts(output, batch(prefill=(True, False), scheduled=(12, 1))) == [0, 0]
    runner.req_states.tree_next_budget_cpu[:] = [0, 0]
    assert runner._tree_verified_counts(output, batch(scheduled=(1, 1))) == [0, 0]


def test_decode_query_must_be_root_plus_verified_prefix():
    with pytest.raises(ValueError, match="query"):
        make_runner([0, 5])._tree_verified_counts(
            SimpleNamespace(scheduled_spec_decode_tokens={"a": [1, 2]}), batch(scheduled=(2, 1))
        )


def test_root_only_sampling_clears_previous_acceptance(monkeypatch):
    runner = make_runner([0, 0])
    runner.rejection_sampler = SimpleNamespace(acceptance=object())
    input_batch = SimpleNamespace(num_draft_tokens=0, tree_acceptance=object())

    def native_sample(self, hidden, batch, grammar):
        assert self.rejection_sampler.acceptance is None
        assert batch.tree_acceptance is None
        return "sampled root"

    monkeypatch.setattr(GPUModelRunner, "sample", native_sample)
    monkeypatch.setattr("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", lambda: False)
    assert runner.sample(None, input_batch, None) == "sampled root"
