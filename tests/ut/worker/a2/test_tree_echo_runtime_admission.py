# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
from types import SimpleNamespace

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.worker.gpu.sample.sampler import BadWordsState, LogitBiasState, PenaltiesState

from vllm_ascend.ascend_config import FinegrainedTPConfig, TreeSpecConfig
from vllm_ascend.models.qwen3_domino import AscendDominoQwen3ForCausalLM
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.scoring import DominoScorer
from vllm_ascend.worker.v2.spec_decode.tree.speculator import AscendTreeSpeculator


@pytest.fixture(scope="module")
def runtime_config():
    # Real 0.31 configuration construction, without loading any weights.
    return EngineArgs(
        model="/mnt/share/weight/Qwen3-8B",
        enforce_eager=True,
        async_scheduling=False,
        tensor_parallel_size=1,
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        speculative_config={"model": "/mnt/share/weight/Qwen3-8B-Domino-b16", "num_speculative_tokens": 16},
        additional_config={"tree_spec_config": {"enabled": True, "budget": 48, "topk": 8}},
    ).create_engine_config()


def make_runner(config):
    # Preserve real class identities at the model/scorer boundary, no weights.
    model = AscendDominoQwen3ForCausalLM.__new__(AscendDominoQwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = config.speculative_config.draft_model_config.hf_config
    # Match DFlashQwen3ForCausalLM.__init__ without loading weights.
    if getattr(model.config, "draft_vocab_size", None) is None:
        model.config.draft_vocab_size = model.config.vocab_size
    model.draft_id_to_target_id = None
    with set_current_vllm_config(config):
        model.logits_processor = LogitsProcessor(model.config.vocab_size)
    speculator = AscendTreeSpeculator.__new__(AscendTreeSpeculator)
    speculator.model = model
    speculator.tree_builder = TreeBuilder(48, 8, DominoScorer(None))
    speculator._tree_graphs = {}
    speculator.enable_adaptive_verification = False
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = config
    runner.ascend_config = SimpleNamespace(finegrained_tp_config=FinegrainedTPConfig())
    runner._tree_budget_enabled = True
    runner._tree_confidence_requested = True
    runner.speculator = speculator
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.adaptive_verification = None
    runner.sampler = SimpleNamespace(
        logits_processors=[cls.__new__(cls) for cls in (LogitBiasState, PenaltiesState, BadWordsState)],
        thinking_budget_state=SimpleNamespace(enabled=False),
        trace_replay_state=None,
    )
    return runner


def test_default_and_tau_zero_keep_fixed_budget():
    assert TreeSpecConfig().verification_policy == "fixed"
    assert TreeSpecConfig().verification_threshold == 0.0
    for policy in ("fixed", "confidence"):
        config = TreeSpecConfig(enabled=True, budget=48, topk=8, verification_policy=policy, verification_threshold=0.0)
        assert not config.confidence_verification_enabled
        assert config.confidence_verification_requested is (policy == "confidence")
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner._tree_budget_enabled = False
    runner._tree_confidence_requested = False
    runner._validate_tree_confidence_runtime()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_threshold_is_rejected_by_real_config(value):
    with pytest.raises(ValueError, match="verification_threshold"):
        TreeSpecConfig(verification_threshold=value)


def test_invalid_policy_and_positive_enablement():
    with pytest.raises(ValueError, match="verification_policy"):
        TreeSpecConfig(verification_policy="other")
    assert TreeSpecConfig(
        enabled=True, budget=48, topk=8, verification_policy="confidence", verification_threshold=0.1
    ).confidence_verification_enabled


def test_actual_runtime_configuration_is_single_eager_batch(runtime_config):
    assert runtime_config.max_concurrent_batches == 1
    assert runtime_config.parallel_config.tensor_parallel_size == 1
    runner = make_runner(runtime_config)
    runner._validate_tree_confidence_runtime((SamplingParams(temperature=0),))
    assert runner.speculator._tree_full_vocab_size == 151936


@pytest.mark.parametrize("field,value", [("async_scheduling", True), ("enforce_eager", False)])
def test_unsupported_engine_facts(runtime_config, field, value):
    target = runtime_config.scheduler_config if field == "async_scheduling" else runtime_config.model_config
    previous = getattr(target, field)
    try:
        setattr(target, field, value)
        with pytest.raises(ValueError, match="confidence unsupported"):
            make_runner(runtime_config)._validate_tree_confidence_runtime()
    finally:
        setattr(target, field, previous)


@pytest.mark.parametrize(
    "case", ["bonus", "graph", "adaptive", "custom", "thinking", "replay", "scorer", "model", "vocab"]
)
def test_unsupported_loaded_runtime(runtime_config, case):
    runner = make_runner(runtime_config)
    if case == "bonus":
        runner.model_state.num_new_sampled_tokens_per_step = 2
    elif case == "graph":
        runner.speculator._tree_graphs[1] = object()
    elif case == "adaptive":
        runner.adaptive_verification = object()
    elif case == "custom":
        runner.sampler.logits_processors.append(object())
    elif case == "thinking":
        runner.sampler.thinking_budget_state.enabled = True
    elif case == "replay":
        runner.sampler.trace_replay_state = object()
    elif case == "scorer":

        class Child(DominoScorer):
            pass

        runner.speculator.tree_builder.scorer = Child(None)
    elif case == "model":
        runner.speculator.model = torch.nn.Identity()
    else:
        runner.speculator.model.logits_processor.org_vocab_size -= 1
    with pytest.raises(ValueError, match="confidence unsupported"):
        runner._validate_tree_confidence_runtime()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.8},
        {"temperature": 0, "presence_penalty": 0.1},
        {"temperature": 0, "frequency_penalty": 0.1},
        {"temperature": 0, "repetition_penalty": 1.1},
        {"temperature": 0, "min_tokens": 1},
        {"temperature": 0, "logit_bias": {0: 0.5}},
        {"temperature": 0, "allowed_token_ids": [0]},
        {"temperature": 0, "bad_words": ["x"]},
        {"temperature": 0, "structured_outputs": StructuredOutputsParams(choice=["x"])},
    ],
)
def test_request_sampling_is_checked_before_use(runtime_config, kwargs):
    with pytest.raises(ValueError, match="confidence unsupported"):
        make_runner(runtime_config)._validate_tree_confidence_runtime((SamplingParams(**kwargs),))


@pytest.mark.parametrize("value", [True, False])
def test_bool_threshold_rejected(value):
    with pytest.raises(ValueError, match="verification_threshold"):
        TreeSpecConfig(verification_threshold=value)


@pytest.mark.parametrize(
    "field",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
def test_real_parallel_gate(runtime_config, monkeypatch, field):
    monkeypatch.setattr(runtime_config.parallel_config, field, 2)
    with pytest.raises(ValueError, match="confidence"):
        make_runner(runtime_config)._validate_tree_confidence_runtime()


@pytest.mark.parametrize(
    "field",
    [
        "oproj_tensor_parallel_size",
        "lmhead_tensor_parallel_size",
        "embedding_tensor_parallel_size",
        "mlp_tensor_parallel_size",
    ],
)
def test_real_finegrained_gate(runtime_config, monkeypatch, field):
    runner = make_runner(runtime_config)
    # Mutate an otherwise legal loaded configuration: constructor rejection
    # alone would not exercise the positive-confidence runner guard.
    monkeypatch.setattr(runner.ascend_config.finegrained_tp_config, field, 2)
    with pytest.raises(ValueError, match="finegrained_tp"):
        runner._validate_tree_confidence_runtime()


def test_request_rejected_before_native_add(runtime_config, monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    calls = []
    monkeypatch.setattr(GPUModelRunner, "add_requests", lambda *args: calls.append(args))
    runner = make_runner(runtime_config)
    output = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(sampling_params=SamplingParams(temperature=1))])
    with pytest.raises(ValueError, match="confidence"):
        runner.add_requests(output)
    assert calls == []
