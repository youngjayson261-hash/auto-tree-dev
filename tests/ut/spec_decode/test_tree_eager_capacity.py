# SPDX-License-Identifier: Apache-2.0
"""Explicit eager capacity, admission failures and untouched live query rows."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_eager_tree_capacity", ROOT / "vllm_ascend/worker/v2/spec_decode/tree/prefix_graph.py"
)
prefix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prefix)


def method(path, owner, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("rows,tokens,expected", [(1, 3, 17), (1, 15, 17), (1, 17, 17), (2, 18, 34), (3, 35, 51)])
def test_live_queries_fit_full_capacity_without_new_requests(rows, tokens, expected):
    tree = NS(verification_eager_pad_to_full_tree=True, budget=16)
    assert prefix.eager_tree_verification_capacity(tree, 1024, rows, tokens, decode=True) == expected
    assert vars(tree) == {"verification_eager_pad_to_full_tree": True, "budget": 16}


@pytest.mark.parametrize(
    "enabled,decode,rows,tokens", [(False, True, 1, 15), (True, False, 1, 20), (True, True, 2, 2), (True, True, 0, 0)]
)
def test_default_prefill_and_root_only_keep_ordinary_shape(enabled, decode, rows, tokens):
    tree = NS(verification_eager_pad_to_full_tree=enabled, budget=16)
    assert prefix.eager_tree_verification_capacity(tree, 1024, rows, tokens, decode=decode) is None


@pytest.mark.parametrize("rows,tokens,limit", [(1, 18, 1024), (2, 18, 33), (-1, 1, 1024)])
def test_capacity_overflow_is_rejected(rows, tokens, limit):
    tree = NS(verification_eager_pad_to_full_tree=True, budget=16)
    with pytest.raises(ValueError, match="full-tree capacity"):
        prefix.eager_tree_verification_capacity(tree, limit, rows, tokens, decode=True)


def backend_config():
    return NS(
        model_config=NS(enforce_eager=True, hf_config=NS(model_type="qwen3_5")),
        speculative_config=NS(
            use_dspark=lambda: False,
            use_dflash=lambda: True,
            num_speculative_tokens=7,
            enforce_eager=True,
            draft_model_config=NS(hf_config=NS(architectures=["DFlash2DraftModel"])),
        ),
        scheduler_config=NS(max_num_seqs=3, max_num_batched_tokens=51),
    )


@pytest.mark.parametrize("invalid", [None, "target_graph", "draft_graph", "target_model", "draft_model", "capacity"])
def test_real_backend_validator_admits_only_the_explicit_supported_path(invalid):
    validate = method("vllm_ascend/ascend_config.py", "TreeSpecConfig", "_validate_backend", {})
    tree = NS(enabled=True, budget=16, verification_eager_pad_to_full_tree=True)
    cfg = backend_config()
    if invalid == "target_graph":
        cfg.model_config.enforce_eager = False
    elif invalid == "draft_graph":
        cfg.speculative_config.enforce_eager = False
    elif invalid == "target_model":
        cfg.model_config.hf_config.model_type = "qwen3"
    elif invalid == "draft_model":
        cfg.speculative_config.draft_model_config.hf_config.architectures = ["OtherDraft"]
    elif invalid == "capacity":
        cfg.scheduler_config.max_num_batched_tokens = 50
    if invalid is None:
        validate(tree, cfg)
    else:
        with pytest.raises(ValueError):
            validate(tree, cfg)


@pytest.mark.parametrize("requested,source", [(False, "scorer_logprob"), (True, "base_full_vocab")])
def test_padding_rejects_non_confidence_or_base_scoring(requested, source):
    validate = method("vllm_ascend/ascend_config.py", "TreeSpecConfig", "_validate", {})
    tree = NS(
        verification_eager_pad_to_full_tree=True,
        confidence_verification_requested=requested,
        verification_confidence_source=source,
    )
    with pytest.raises(ValueError, match="Scorer confidence"):
        validate(tree)


def test_graph_dispatch_selects_eager_capacity_and_retains_real_request_count(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.v2.spec_decode.tree.prefix_graph", prefix)
    dispatch = method(
        "vllm_ascend/worker/v2/aclgraph_utils.py",
        "ModelAclGraphManager",
        "dispatch",
        {"BatchExecutionDescriptor": NS, "CUDAGraphMode": NS(NONE="NONE")},
    )
    tree = NS(verification_eager_pad_to_full_tree=True, budget=16)
    manager = NS(
        vllm_config=NS(
            additional_config={"tree_spec_config": {"verification_eager_pad_to_full_tree": True}},
            scheduler_config=NS(max_num_batched_tokens=1024),
        ),
        model_runner=NS(ascend_config=NS(tree_spec_config=tree)),
        tree_prefix_decode=True,
    )
    result = dispatch(manager, 2, 18, None, 0, max_query_len=13)
    assert (result.cg_mode, result.num_reqs, result.num_tokens) == ("NONE", 2, 34)


def test_prepare_inputs_zeros_only_padding_and_retains_live_rows(monkeypatch):
    import sys

    import torch

    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.v2.spec_decode.tree.prefix_graph", prefix)
    prepare = method("vllm_ascend/worker/v2/model_runner.py", "NPUModelRunner", "prepare_inputs", {})
    buffers = NS(input_ids=torch.arange(34), positions=torch.arange(34), is_padding=torch.ones(34, dtype=torch.bool))
    batch = NS(num_tokens=18, num_tokens_after_padding=34, slot_positions=torch.arange(34))
    runner = NS(
        _prepare_inputs_impl=lambda *args: batch,
        input_buffers=buffers,
        vllm_config=NS(additional_config={}),
        ascend_config=NS(tree_spec_config=NS(verification_eager_pad_to_full_tree=True)),
    )
    assert prepare(runner, None, None, None, 0) is batch
    assert torch.equal(buffers.input_ids[:18], torch.arange(18))
    assert torch.equal(buffers.positions[:18], torch.arange(18))
    assert torch.equal(batch.slot_positions[:18], torch.arange(18))
    assert not buffers.is_padding[:18].any()
    assert buffers.is_padding[18:].all()
    assert torch.count_nonzero(buffers.input_ids[18:]) == 0
    assert torch.count_nonzero(buffers.positions[18:]) == 0
    assert torch.count_nonzero(batch.slot_positions[18:]) == 0
