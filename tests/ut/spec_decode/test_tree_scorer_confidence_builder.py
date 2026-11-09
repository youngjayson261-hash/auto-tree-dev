# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Scorer log-probability confidence through the complete tree builder."""

import dataclasses
import math

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.tree.base_confidence import prepare_scorer_confidence
from vllm_ascend.worker.v2.spec_decode.tree.budget_policy import (
    PolicyStatus,
    TreeLayoutError,
    TreeScoreSummary,
    select_tree_prefix_device,
    select_tree_prefix_reference,
)
from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder, TreeBuildState
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout
from vllm_ascend.worker.v2.spec_decode.tree.scoring import CandidateLayer, DominoScorer


class SmallCorrection:
    """Small model boundary; production Domino scoring and GRU updates run."""

    def __init__(self):
        self.gru_hidden_dim = 1
        self.gru_w_hh = torch.zeros(1)
        self.w_s = torch.ones(1, 1)
        self.middle = torch.nn.Identity()
        self.fc2_weight = torch.tensor([[-8.0], [5.0], [-1.7], [4.0], [2.5], [-0.5]])
        self.fc2_bias = torch.zeros(6)

    def project_z(self, hidden):
        return hidden[..., :1]

    def update_hidden(self, tokens, states):
        return states * 0.5 + (tokens.float()[:, None] + 1) * 0.1


class RecordingBuilder(TreeBuilder):
    """Independently accumulate scalar edge scores along actual selected IDs."""

    def _expand_layer(self, state, layer, child_depth, origin_candidate, **kwargs):
        if child_depth == 1:
            self.reference_paths = [{0: 0.0} for _ in range(state.tokens.shape[0])]
            self.reference_maxima = []
            self.edge_layers = []
        scores = layer.scores.expand(-1, state.width, -1).double().tolist()
        tokens = layer.token_ids.expand(-1, state.width, -1).tolist()
        parents = state.frontier.tolist()
        live = 1 if child_depth == 1 else state.width
        candidates = []
        for row, row_scores in enumerate(scores):
            candidates.append(
                [
                    self.reference_paths[row][parents[row][parent]] + score if parent < live else -math.inf
                    for parent, children in enumerate(row_scores)
                    for score in children
                ]
            )
        self.edge_layers.append((scores, tokens))
        self.reference_maxima.append([max(row) for row in candidates])
        frontier, selected = super()._expand_layer(state, layer, child_depth, origin_candidate, **kwargs)
        for row, indices in enumerate(selected.tolist()):
            for offset, index in enumerate(indices):
                self.reference_paths[row][state.num_nodes + 1 + offset] = candidates[row][index]
        return frontier, selected


def inputs():
    logits = torch.tensor(
        [
            [[3.0, 2.5, 2.0, 1.5, 0.0, -1.0], [2.0, 0.1, 1.1, 2.3, -1.0, -2.0], [0.2, 1.0, 2.0, 0.7, 0.0, -1.0]],
            [[0.5, 1.5, 2.0, 1.2, -1.0, 0.0], [1.0, 2.0, 0.1, 2.5, -2.0, 0.0], [1.7, 0.2, 2.2, 1.3, 0.3, -0.5]],
        ]
    )
    return logits, torch.tensor([5, 2]), torch.full((2, 3, 1), 0.8)


def output(rows, budget):
    return TreeLayout(
        tokens=torch.empty(rows, budget, dtype=torch.long),
        depths=torch.empty(rows, budget, dtype=torch.long),
        parents=torch.empty(rows, budget, dtype=torch.long),
        num_nodes=torch.empty(rows, dtype=torch.int32),
        visibility=torch.empty(rows, budget, budget, dtype=torch.bool),
        first_child=torch.empty(rows, budget + 1, dtype=torch.int32),
        next_sibling=torch.empty(rows, budget + 1, dtype=torch.int32),
    )


def summary(rows, depth=3):
    return TreeScoreSummary(
        torch.empty(rows, depth, dtype=torch.float32),
        torch.empty(rows, depth, dtype=torch.bool),
        torch.empty(rows, dtype=torch.bool),
    )


def assert_reference(builder, scores):
    expected = torch.tensor(builder.reference_maxima).T
    torch.testing.assert_close(scores.max_path_logp, expected)
    assert scores.valid_depths.all()
    assert not scores.invalid_scores.any()
    for row, paths in enumerate(builder.reference_paths):
        for node, value in paths.items():
            torch.testing.assert_close(builder._base_path_scores_buf[row, node], torch.tensor(value))
    assert builder._base_path_scores_buf.data_ptr() != builder._path_scores_buf.data_ptr()


@pytest.mark.parametrize("prefix", [0, 1, 3])
@pytest.mark.parametrize("origin,penalty,budget", [(False, 0.0, 5), (False, 1.5, 5), (True, 0.4, 3)])
def test_domino_scorer_confidence_matches_raw_paths_without_changing_tree_or_gru(prefix, origin, penalty, budget):
    logits, roots, hidden = inputs()
    builders, trees = [], []
    scores = summary(2)
    for enabled in (False, True):
        builder = RecordingBuilder(
            budget,
            2,
            DominoScorer(SmallCorrection(), prefix_len=prefix, candidate_size=4),
            keep_origin_chain=origin,
            depth_penalty=penalty,
        )
        tree = output(2, budget)
        kwargs = (
            dict(score_summary=scores, full_vocab_size=6, verification_confidence_source="scorer_logprob")
            if enabled
            else {}
        )
        builder.build(logits, tree, root_token_ids=roots, draft_hidden=hidden, **kwargs)
        builders.append(builder)
        trees.append(tree)
    for field in dataclasses.fields(TreeLayout):
        assert torch.equal(getattr(trees[0], field.name), getattr(trees[1], field.name)), field.name
    torch.testing.assert_close(builders[0]._path_scores_buf, builders[1]._path_scores_buf, rtol=0, atol=0)
    torch.testing.assert_close(builders[0].scorer._hidden, builders[1].scorer._hidden, rtol=0, atol=0)
    assert_reference(builders[1], scores)


def test_domino_retained_topk_is_not_renormalized_and_same_token_depends_on_parent():
    logits, roots, hidden = inputs()
    builder = RecordingBuilder(6, 2, DominoScorer(SmallCorrection(), candidate_size=4))
    scores = summary(2)
    builder.build(
        logits,
        output(2, 6),
        root_token_ids=roots,
        draft_hidden=hidden,
        score_summary=scores,
        full_vocab_size=6,
        verification_confidence_source="scorer_logprob",
    )
    assert_reference(builder, scores)
    # C=4 but only two edges are retained. Their mass must remain below one.
    first_edges = builder.edge_layers[0][0]
    assert any(sum(math.exp(value) for value in row[0]) < 0.9999 for row in first_edges)
    differences = []
    for edge_scores, token_ids in builder.edge_layers[1:]:
        for row_scores, row_ids in zip(edge_scores, token_ids):
            left = dict(zip(row_ids[0], row_scores[0]))
            right = dict(zip(row_ids[1], row_scores[1]))
            differences.extend(abs(left[token] - right[token]) for token in left.keys() & right.keys())
    assert differences and max(differences) > 0.01


def test_scorer_summary_selects_a_shorter_budget_matching_reference():
    logits, roots, hidden = inputs()
    builder = TreeBuilder(6, 2, DominoScorer(SmallCorrection(), candidate_size=4))
    scores = summary(2)
    tree = builder.build(
        logits,
        output(2, 6),
        root_token_ids=roots,
        draft_hidden=hidden,
        score_summary=scores,
        full_vocab_size=6,
        verification_confidence_source="scorer_logprob",
    )
    # Place the threshold strictly between row zero's first two depth scores.
    tau = math.exp(float(scores.max_path_logp[0, :2].mean()))
    packet = torch.empty(2, 2, dtype=torch.int32)
    select_tree_prefix_device(tree, scores, packet, tau=tau, bootstrap_mask=torch.zeros(2, dtype=torch.bool))
    for row in range(2):
        expected = select_tree_prefix_reference(
            depths=tree.depths[row].tolist(),
            parents=tree.parents[row].tolist(),
            max_path_logp=scores.max_path_logp[row].tolist(),
            valid_depths=scores.valid_depths[row].tolist(),
            invalid_scores=bool(scores.invalid_scores[row]),
            policy="confidence",
            tau=tau,
            bootstrap=False,
        )
        assert packet[row].tolist() == [expected[0], int(expected[1])]
    assert packet[0, 0] == 2
    assert packet[0, 0] < tree.num_nodes[0]
    assert packet[0, 1] == PolicyStatus.CONFIDENCE


def test_forced_origin_records_the_final_selected_path():
    builder = TreeBuilder(4, 2, keep_origin_chain=True)
    builder.prepare_for_capture(1, 4, 2, torch.device("cpu"), reserve_base_confidence=True)
    state = TreeBuildState(
        tokens=torch.tensor([[0, 1, -1, -1]]),
        depths=torch.tensor([[1, 1, 0, 0]]),
        parents=torch.zeros(1, 5, dtype=torch.long),
        path_scores=torch.tensor([[0.0, -0.1, -0.2, 0.0, 0.0]]),
        frontier=torch.tensor([[1, 2]]),
        width=2,
        num_nodes=2,
    )
    confidence = prepare_scorer_confidence(torch.empty(1, 5), summary(1, 2))
    confidence.path_scores[:, 1:3] = state.path_scores[:, 1:3]
    layer = CandidateLayer(
        scores=torch.tensor([[[-0.2, -2.0], [-0.2, -3.0]]]),
        token_ids=torch.tensor([[[0, 1], [2, 3]]]),
    )
    _, selected = builder._expand_layer(state, layer, 2, torch.tensor([3]), confidence=confidence)
    assert selected.tolist() == [[0, 3]]
    torch.testing.assert_close(confidence.path_scores[:, 3:5], state.path_scores[:, 3:5])
    torch.testing.assert_close(confidence.path_scores[:, 3:5], torch.tensor([[-0.3, -3.2]]))


def test_scorer_confidence_reuses_frozen_workspace_after_reorder_and_shrink():
    logits, roots, hidden = inputs()
    builder = RecordingBuilder(6, 2, DominoScorer(SmallCorrection(), prefix_len=1, candidate_size=4))
    builder.prepare_for_capture(2, 6, 3, torch.device("cpu"), reserve_base_confidence=True)
    pointers = tuple(tensor.data_ptr() for tensor in builder.workspace_tensors())
    for indices in ([0, 1], [1, 0], [1]):
        scores = summary(len(indices))
        builder.build(
            logits[indices],
            output(len(indices), 6),
            root_token_ids=roots[indices],
            draft_hidden=hidden[indices],
            score_summary=scores,
            full_vocab_size=6,
            verification_confidence_source="scorer_logprob",
        )
        assert tuple(tensor.data_ptr() for tensor in builder.workspace_tensors()) == pointers
        assert_reference(builder, scores)


def test_scorer_confidence_requires_reservation_and_supported_source():
    logits, roots, hidden = inputs()
    builder = TreeBuilder(6, 2, DominoScorer(SmallCorrection(), candidate_size=4))
    builder.prepare_for_capture(2, 6, 3, torch.device("cpu"))
    options = dict(score_summary=summary(2), full_vocab_size=6, verification_confidence_source="scorer_logprob")
    with pytest.raises(ValueError, match="not reserved before capture"):
        builder.build(logits, output(2, 6), root_token_ids=roots, draft_hidden=hidden, **options)
    options["verification_confidence_source"] = "priority"
    with pytest.raises(TreeLayoutError):
        builder.build(logits, output(2, 6), root_token_ids=roots, draft_hidden=hidden, **options)


def test_no_summary_keeps_the_existing_path_without_confidence_workspace():
    logits, roots, hidden = inputs()
    builder = TreeBuilder(6, 2, DominoScorer(SmallCorrection(), candidate_size=4))
    builder.build(
        logits,
        output(2, 6),
        root_token_ids=roots,
        draft_hidden=hidden,
        verification_confidence_source="scorer_logprob",
    )
    assert builder._base_path_scores_buf is None
