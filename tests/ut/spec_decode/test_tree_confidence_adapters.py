# SPDX-License-Identifier: Apache-2.0
"""Mapped vocabulary and candidate normalization contracts with real builders."""

import os
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.tree.budget_policy import TreeScoreSummary
from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder, create_tree_builder
from vllm_ascend.worker.v2.spec_decode.tree.dflash2_builder import DFlash2BeamTreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout
from vllm_ascend.worker.v2.spec_decode.tree.scoring import (
    DFlash2Scorer,
    MarkovScorer,
    validate_confidence_vocabulary,
)


def output_buffers(rows, budget=6, depth=3):
    layout = TreeLayout(
        tokens=torch.empty(rows, budget, dtype=torch.long),
        depths=torch.empty(rows, budget, dtype=torch.long),
        parents=torch.empty(rows, budget, dtype=torch.long),
        num_nodes=torch.empty(rows, dtype=torch.int32),
        visibility=torch.empty(rows, budget, budget, dtype=torch.bool),
        first_child=torch.empty(rows, budget + 1, dtype=torch.int32),
        next_sibling=torch.empty(rows, budget + 1, dtype=torch.int32),
    )
    summary = TreeScoreSummary(
        torch.empty(rows, depth), torch.empty(rows, depth, dtype=torch.bool), torch.empty(rows, dtype=torch.bool)
    )
    return layout, summary


def markov_model(mapping=None):
    offsets = None if mapping is None else torch.tensor(mapping) - torch.arange(3)
    return NS(
        config=NS(vocab_size=8 if mapping is not None else 3, draft_vocab_size=3),
        logits_processor=NS(vocab_size=3, org_vocab_size=3),
        draft_id_to_target_id=offsets,
        map_draft_to_target=lambda ids: ids if offsets is None else ids + offsets[ids],
        markov_embed=lambda ids: torch.zeros(*ids.shape, 1),
        markov_bias=lambda hidden: torch.zeros(hidden.shape[0], 3),
    )


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("source", ["base_full_vocab", "scorer_logprob"])
@pytest.mark.parametrize("mapping", [None, [4, 1, 7]])
def test_markov_keeps_global_tree_ids_but_indexes_original_base_columns(rows, source, mapping):
    model = markov_model(mapping)
    scorer = MarkovScorer(model)
    assert validate_confidence_vocabulary(model, scorer, model.config.vocab_size, source) == 3
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 3.0, -2.0], [-1.0, 0.0, 1.0]]]).repeat(rows, 1, 1)
    logits += torch.arange(rows)[:, None, None] * 0.2
    output, summary = output_buffers(rows)
    builder = TreeBuilder(6, 2, scorer=scorer)
    roots = torch.zeros(rows, dtype=torch.long)
    tree = builder.build(
        logits,
        output,
        root_token_ids=roots,
        score_summary=summary,
        full_vocab_size=3,
        verification_confidence_source=source,
    )
    # With zero correction, enumerate the independent best path analytically.
    expected = logits.log_softmax(-1).amax(-1).cumsum(-1)
    torch.testing.assert_close(summary.max_path_logp, expected)
    assert summary.valid_depths.all() and not summary.invalid_scores.any()
    allowed = set(range(3) if mapping is None else mapping)
    assert set(tree.tokens.flatten().tolist()) <= allowed
    plain, _ = output_buffers(rows)
    TreeBuilder(6, 2, scorer=scorer).build(logits, plain, root_token_ids=roots)
    for field in ("tokens", "depths", "parents", "num_nodes", "visibility"):
        assert torch.equal(getattr(tree, field), getattr(plain, field))


@pytest.mark.parametrize("mapping", [[1, 1, 2], [0, 2, 8], [-1, 1, 3]])
def test_markov_rejects_duplicate_or_out_of_range_mapping_before_capture(mapping):
    model = markov_model(mapping)
    with pytest.raises(ValueError, match="injective in-range"):
        validate_confidence_vocabulary(model, MarkovScorer(model), 8, "base_full_vocab")


def test_markov_rejects_incomplete_probability_vocabulary():
    model = markov_model()
    model.logits_processor.org_vocab_size = 2
    with pytest.raises(ValueError, match="complete draft-vocabulary"):
        validate_confidence_vocabulary(model, MarkovScorer(model), 3, "scorer_logprob")


def dflash2_model():
    selector = NS(
        predecessor_codebook=torch.zeros(8, 2),
        successor_codebook=torch.zeros(8, 2),
        hidden_projection=torch.nn.Identity(),
        top_k=3,
    )
    return NS(config=NS(vocab_size=8), model=NS(candidate_selector=selector), compute_candidates=lambda hidden: None)


def test_dflash2_scores_normalize_over_candidate_lattice_and_keep_target_ids():
    model = dflash2_model()
    scorer = DFlash2Scorer(model)
    assert validate_confidence_vocabulary(model, scorer, 8, "scorer_logprob") == 8
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 3.0, -2.0], [-1.0, 0.0, 1.0]]]).repeat(2, 1, 1)
    ids = torch.tensor([[[6, 2, 7], [1, 4, 5], [7, 0, 3]]]).repeat(2, 1, 1)
    layout, summary = output_buffers(2)
    TreeBuilder(6, 2, scorer=scorer).build(
        logits,
        layout,
        root_token_ids=torch.tensor([0, 1]),
        draft_hidden=torch.zeros(2, 3, 2),
        draft_candidate_ids=ids,
        score_summary=summary,
        full_vocab_size=8,
        verification_confidence_source="scorer_logprob",
    )
    torch.testing.assert_close(summary.max_path_logp, logits.log_softmax(-1).amax(-1).cumsum(-1))
    assert summary.valid_depths.all() and not summary.invalid_scores.any()
    assert set(layout.tokens[:, :2].flatten().tolist()) == {2, 6}


def test_dflash2_does_not_claim_full_base_vocabulary_probabilities():
    model = dflash2_model()
    with pytest.raises(ValueError, match="full base logits are unavailable"):
        validate_confidence_vocabulary(model, DFlash2Scorer(model), 8, "base_full_vocab")


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("keep_origin", [False, True])
def test_dflash2_factory_beam_summarizes_pool_without_changing_tree(rows, keep_origin):
    # The runtime factory selects a separate beam builder with W^2 children
    # per depth. Exercising only TreeBuilder misses that real caller contract.
    model = dflash2_model()
    builder = create_tree_builder(6, 2, "dflash", draft_model=model, depth_penalty=0.7, keep_origin_chain=keep_origin)
    assert type(builder) is DFlash2BeamTreeBuilder
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 3.0, -2.0], [-1.0, 0.0, 1.0]]]).repeat(rows, 1, 1)
    ids = torch.tensor([[[6, 2, 7], [1, 4, 5], [7, 0, 3]]]).repeat(rows, 1, 1)
    layout, summary = output_buffers(rows)
    inputs = dict(root_token_ids=torch.arange(rows), draft_hidden=torch.zeros(rows, 3, 2), draft_candidate_ids=ids)
    builder.build(
        logits,
        layout,
        **inputs,
        score_summary=summary,
        full_vocab_size=8,
        verification_confidence_source="scorer_logprob",
    )
    # Zero correction gives an analytic path maximum, independent of pooling,
    # origin preservation and the final depth-penalized node selection.
    torch.testing.assert_close(summary.max_path_logp, logits.log_softmax(-1).amax(-1).cumsum(-1))
    assert summary.valid_depths.all() and not summary.invalid_scores.any()
    plain, _ = output_buffers(rows)
    builder.build(logits, plain, **inputs)
    for field in ("tokens", "depths", "parents", "num_nodes", "visibility", "first_child", "next_sibling"):
        assert torch.equal(getattr(layout, field), getattr(plain, field))


@pytest.mark.parametrize("bad", ["base", "ids", "vocabulary"])
def test_dflash2_factory_beam_rejects_unproven_confidence_inputs(bad):
    builder = create_tree_builder(6, 2, "dflash", draft_model=dflash2_model())
    layout, summary = output_buffers(1)
    inputs = dict(
        root_token_ids=torch.tensor([0]),
        draft_hidden=torch.zeros(1, 3, 2),
        draft_candidate_ids=torch.zeros(1, 3, 3, dtype=torch.long),
        score_summary=summary,
        full_vocab_size=8,
        verification_confidence_source="scorer_logprob",
    )
    if bad == "base":
        inputs["verification_confidence_source"] = "base_full_vocab"
    elif bad == "ids":
        inputs["draft_candidate_ids"] = torch.zeros(1, 3, 2, dtype=torch.long)
    else:
        inputs["full_vocab_size"] = 7
    with pytest.raises(ValueError, match="DFlash2 beam confidence"):
        builder.build(torch.zeros(1, 3, 3), layout, **inputs)


@pytest.mark.skipif(os.environ.get("ECHO_TEST_NPU") != "1", reason="explicit native validation only")
def test_dflash2_factory_beam_confidence_replays_on_device():
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    assert torch.npu.device_count() == 1
    torch.npu.set_device(0)
    init_device_properties_triton()
    model = dflash2_model()
    selector = model.model.candidate_selector
    selector.predecessor_codebook = selector.predecessor_codebook.to("npu")
    selector.successor_codebook = selector.successor_codebook.to("npu")
    builder = create_tree_builder(6, 2, "dflash", draft_model=model)
    layout, summary = output_buffers(2)
    layout = TreeLayout(**{name: tensor.to("npu") for name, tensor in vars(layout).items()})
    summary = TreeScoreSummary(*(tensor.to("npu") for tensor in summary))
    values = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 3.0, -2.0], [-1.0, 0.0, 1.0]]]).repeat(2, 1, 1)
    logits = values.to("npu")
    ids = torch.tensor([[[6, 2, 7], [1, 4, 5], [7, 0, 3]]]).repeat(2, 1, 1).to("npu")
    roots, hidden = torch.tensor([0, 1], device="npu"), torch.zeros(2, 3, 2, device="npu")

    def build():
        builder.build(
            logits,
            layout,
            root_token_ids=roots,
            draft_hidden=hidden,
            draft_candidate_ids=ids,
            score_summary=summary,
            full_vocab_size=8,
            verification_confidence_source="scorer_logprob",
        )

    build()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        build()
    for source in (values, values.flip(-1), values):
        logits.copy_(source.to("npu"))
        graph.replay()
        torch.npu.synchronize()
        expected = source.log_softmax(-1).amax(-1).cumsum(-1)
        torch.testing.assert_close(summary.max_path_logp.cpu(), expected)
        assert summary.valid_depths.all().cpu() and not summary.invalid_scores.any().cpu()
