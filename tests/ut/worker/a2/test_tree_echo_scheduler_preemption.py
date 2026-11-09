# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the Scheduler -> native Worker removal -> Ascend state boundary.

The Scheduler, KVCacheManager, Worker removal methods, RequestState and Handler
are real. Model outputs and proposal budgets are controlled component inputs.
Only model/LoRA cleanup collaborators record calls; no Engine, model forward or
KV/hidden commit is exercised. The Worker finish substep is called on its own,
without claiming coverage of its add/update or full execute_model sequence.
"""

import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from tests.ut.kv_offload.utils import create_request, create_scheduler, create_vllm_config
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.spec_decode.tree import budget_metrics
from vllm_ascend.worker.v2.spec_decode.tree.budget_metrics import TreeBudgetStats
from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.draft_tokens import TreeDraftTokensHandler
from vllm_ascend.worker.v2.spec_decode.tree.shape import TreeShape
from vllm_ascend.worker.v2.spec_decode.tree.speculator import AscendTreeSpeculator
from vllm_ascend.worker.v2.states import AscendRequestState
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper


def _sample_one_token(scheduler, scheduler_output, requests):
    """Supply real ModelRunnerOutput metadata without loading model weights."""
    req_ids = [request.request_id for request in requests]
    assert set(req_ids) <= scheduler_output.num_scheduled_tokens.keys()
    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=[[0] for _ in req_ids],
        ),
    )


def _natural_preemption():
    """Use actual finite KV capacity instead of replacing allocate/preempt."""
    config = create_vllm_config(max_num_seqs=3, max_num_batched_tokens=200, block_size=16)
    # The common helper defaults to a connector when its argument is None.
    config.kv_transfer_config = None
    config.cache_config.enable_prefix_caching = False
    config.scheduler_config.async_scheduling = False
    config.scheduler_config.policy = "priority"
    scheduler = create_scheduler(config, num_blocks=6)
    assert isinstance(scheduler, Scheduler)
    assert isinstance(scheduler.kv_cache_manager, KVCacheManager)
    assert hasattr(scheduler, "use_v2_model_runner")
    scheduler.use_v2_model_runner = True

    low = create_request(request_id=1, num_tokens=32, max_tokens=8, block_size=16)
    low.priority, low.arrival_time = 5, 1.0
    scheduler.add_request(low)
    first = scheduler.schedule()
    _sample_one_token(scheduler, first, [low])

    high = create_request(request_id=2, num_tokens=32, max_tokens=8, block_size=16)
    high.priority, high.arrival_time = 0, 2.0
    scheduler.add_request(high)
    second = scheduler.schedule()
    _sample_one_token(scheduler, second, [low, high])
    preemption = scheduler.schedule()

    assert low.status == RequestStatus.PREEMPTED
    assert low in scheduler.waiting
    assert high in scheduler.running
    assert low.request_id in preemption.preempted_req_ids
    assert high.request_id not in preemption.preempted_req_ids
    assert not preemption.finished_req_ids
    return scheduler, low, high, preemption


class _ModelRemovalLog:
    """Observe ordering without removing the request or changing its budget."""

    def __init__(self, state, calls):
        self.state = state
        self.calls = calls

    def remove_request(self, req_id):
        self.calls.append(("model", req_id, self.state.req_id_to_index.get(req_id)))


class _LoraRemovalLog:
    def __init__(self, state, calls):
        self.state = state
        self.calls = calls

    def remove_request(self, req_id):
        self.calls.append(("lora", req_id, self.state.req_id_to_index.get(req_id)))


def _removal_worker(state, calls, speculator):
    """Initialize only collaborators used by the pinned native finish substep."""
    worker = NPUModelRunner.__new__(NPUModelRunner)
    worker.req_states = state
    assert isinstance(speculator, AscendTreeSpeculator)
    worker.speculator = speculator
    worker.model_state = _ModelRemovalLog(state, calls)
    worker.lora_state = _LoraRemovalLog(state, calls)
    worker.pooling_runner = None
    worker.pp_handler = None
    worker.encoder_cache = None
    worker.prompt_logprobs_worker = None
    # The dispatch stays native. P17 overrides _remove_request only to add
    # passive diagnostics; its delegation is observed below, not replaced.
    assert NPUModelRunner.finish_requests is GPUModelRunner.finish_requests
    return worker


@pytest.mark.skipif(os.environ.get("ECHO_TEST_NPU") != "1", reason="requires explicitly reserved NPU")
def test_native_scheduler_preemption_clears_budget_and_rejects_stale_publication(monkeypatch, request):
    pytest.importorskip("torch_npu")
    from vllm_ascend.patch.worker.patch_v2 import patch_uva  # noqa: F401

    scheduler, low, high, preemption = _natural_preemption()
    stats = TreeBudgetStats()
    previous_level = budget_metrics.logger.level
    budget_metrics.logger.setLevel(logging.DEBUG)
    # Restore through pytest even if a device assertion fails.
    request.addfinalizer(lambda: budget_metrics.logger.setLevel(previous_level))
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    with torch_cuda_wrapper():
        state = AscendRequestState(3, 64, 64, 4, 128, device, TreeShape(4, 8, 2))
        # Seed valid native state through its real add API. This is not a claim
        # that the Worker's separate add_requests substep has been exercised.
        for request in (low, high):
            state.add_request(request.request_id, 32, list(request.prompt_token_ids), 0, 8)
        # Real type, initialized only for this removal/publication component.
        # No model constructor or forward runs. S=8 is a controlled input.
        speculator = AscendTreeSpeculator.__new__(AscendTreeSpeculator)
        speculator.budget = 8
        speculator.tree_builder = TreeBuilder(budget=8, topk=2)
        speculator.tree_builder.num_selected_nodes = 8
        speculator.tree_budget_stats = stats
        speculator.tree_budget_packet = torch.zeros((2, 2), dtype=torch.int32, device=device)
        handler = TreeDraftTokensHandler(
            device, state, speculator, verification_policy="confidence", threshold=0.1, budget_stats=stats
        )
    req_ids = [low.request_id, high.request_id]
    producer = torch.npu.Stream(device=device)

    def seal(budgets):
        batch = SimpleNamespace(
            req_ids=req_ids,
            idx_mapping_np=np.array([state.req_id_to_index[req_id] for req_id in req_ids]),
            has_structured_output_reqs=False,
        )
        with torch.npu.stream(producer):
            speculator.tree_budget_packet.copy_(torch.tensor([[budget, 0] for budget in budgets], dtype=torch.int32))
            handler.set_draft_tokens(batch, None)
            handler.seal_budget_production()

    seal([3, 1])
    assert handler.get_draft_tokens().draft_token_ids == [[-1] * 3, [-1]]
    low_slot = state.req_id_to_index[low.request_id]
    high_slot = state.req_id_to_index[high.request_id]
    seal([4, 2])
    calls = []
    worker = _removal_worker(state, calls, speculator)
    native_calls = []
    native_results = []
    discard_calls = []
    original_remove = GPUModelRunner._remove_request
    original_discard = stats.discard

    def observe_base(self, req_id):
        native_calls.append(req_id)
        # Delegate exactly once to the genuine GPU implementation. Exceptions
        # propagate before any diagnostic code in the NPU override can run.
        result = original_remove(self, req_id)
        native_results.append(result)
        return result

    def observe_discard(req_id):
        discard_calls.append((req_id, state.req_id_to_index.get(req_id)))
        return original_discard(req_id)

    monkeypatch.setattr(GPUModelRunner, "_remove_request", observe_base)
    monkeypatch.setattr(stats, "discard", observe_discard)

    # The real SchedulerOutput reaches the real Worker dispatch and real state
    # removal. No manual remove_request call stands in for this bridge.
    worker.finish_requests(preemption)
    assert native_calls == [low.request_id]
    assert native_results == [True]
    assert discard_calls == [(low.request_id, None)]
    assert stats.snapshot()["discarded_publications"] == 1
    assert stats.snapshot()["discarded_a_non_root"] == 3
    assert stats.snapshot()["invalid_metric_rows"] == 0
    assert not stats.snapshot()["diagnostic_unknown"]
    assert low.request_id not in state.req_id_to_index
    assert state.tree_next_budget_cpu[low_slot] is None
    assert state.req_id_to_index[high.request_id] == high_slot
    assert state.tree_next_budget_cpu[high_slot] == 1
    assert calls == [("model", low.request_id, low_slot), ("lora", low.request_id, None)]

    # Isolate the same-slot/new-ID edge with the real state allocation API;
    # this deliberately does not claim coverage of Worker.add_requests.
    replacement = "replacement"
    with torch_cuda_wrapper():
        state.add_request(replacement, 2, [1, 2], 0, 8)
    assert state.req_id_to_index[replacement] == low_slot
    assert state.tree_next_budget_cpu[low_slot] is None
    delivered = handler.get_draft_tokens()
    assert delivered.req_ids == [high.request_id]
    assert delivered.draft_token_ids == [[-1] * 2]
    assert state.tree_next_budget_cpu[low_slot] is None
    assert state.tree_next_budget_cpu[high_slot] == 2
    assert handler.get_draft_tokens() is None

    # Also exercise finished_req_ids from an actual Scheduler cancellation.
    scheduler.finish_requests([high.request_id], RequestStatus.FINISHED_ABORTED)
    cancellation = scheduler.schedule()
    assert high.request_id in cancellation.finished_req_ids

    # DEBUG off must not traverse or mutate diagnostic lifecycle metadata.
    class DisabledPoison(dict):
        def pop(self, *args, **kwargs):
            raise AssertionError("disabled removal touched diagnostic metadata")

    delivered = stats._delivered
    stats._delivered = DisabledPoison(delivered)
    expected_discard_calls = list(discard_calls)
    budget_metrics.logger.setLevel(logging.INFO)
    worker.finish_requests(cancellation)
    assert native_calls == [low.request_id, high.request_id]
    assert native_results == [True, True]
    assert discard_calls == expected_discard_calls
    stats._delivered = delivered
    budget_metrics.logger.setLevel(logging.DEBUG)
    assert high.request_id not in state.req_id_to_index
    assert state.tree_next_budget_cpu[high_slot] is None
    assert state.req_id_to_index[replacement] == low_slot
    assert state.tree_next_budget_cpu[low_slot] is None
    assert ("model", high.request_id, high_slot) in calls
    assert ("lora", high.request_id, None) in calls

    # A logger predicate failure must not replace a genuine successful native
    # return, nor the False result for an already absent request.
    def broken_enabled(_level):
        raise RuntimeError("controlled logger predicate failure")

    with monkeypatch.context() as logging_failure:
        logging_failure.setattr(budget_metrics.logger, "isEnabledFor", broken_enabled)
        assert worker._remove_request(replacement) is native_results[-1] is True
        assert replacement not in state.req_id_to_index
        assert state.tree_next_budget_cpu[low_slot] is None
        assert worker._remove_request(replacement) is native_results[-1] is False
    assert native_calls == [low.request_id, high.request_id, replacement, replacement]
    assert native_results == [True, True, True, False]
    assert stats.diagnostic_unknown and stats.diagnostic_errors == 2

    # A diagnostic lifecycle failure happens only after the real state removal
    # and is marked unknown; no logger is used to handle this error.
    diagnostic_req = "diagnostic-failure"
    with torch_cuda_wrapper():
        state.add_request(diagnostic_req, 2, [1, 2], 0, 8)

    class FailingPop(dict):
        def pop(self, *args, **kwargs):
            raise RuntimeError("controlled diagnostic lifecycle failure")

    stats._delivered = FailingPop(stats._delivered)
    assert worker._remove_request(diagnostic_req) is native_results[-1] is True
    assert diagnostic_req not in state.req_id_to_index
    assert discard_calls[-1] == (diagnostic_req, None)
    assert stats.diagnostic_unknown and stats.diagnostic_errors == 3

    # An original collaborator exception inside the genuine GPU body remains
    # identical. The NPU override must not retry or enter diagnostics afterward.
    native_error = RuntimeError("controlled native collaborator failure")
    before_calls = len(native_calls)
    before_discards = len(discard_calls)
    before_errors = stats.diagnostic_errors

    def fail_model_remove(_req_id):
        raise native_error

    monkeypatch.setattr(worker.model_state, "remove_request", fail_model_remove)
    with pytest.raises(RuntimeError) as raised:
        worker._remove_request("native-error")
    assert raised.value is native_error
    assert native_calls[before_calls:] == ["native-error"]
    assert len(native_results) == before_calls
    assert len(discard_calls) == before_discards
    assert stats.diagnostic_errors == before_errors
