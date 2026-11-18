# SPDX-License-Identifier: Apache-2.0
"""DFlash2 beam construction with separate candidate pools and frontiers."""

import torch

from vllm_ascend.worker.v2.spec_decode.tree.base_confidence import prepare_base_confidence, prepare_scorer_confidence
from vllm_ascend.worker.v2.spec_decode.tree.budget_policy import TreeLayoutError, TreeScoreSummary
from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder, TreeBuildState
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, finalize_tree_layout_from_state
from vllm_ascend.worker.v2.spec_decode.tree.scoring import DFlash2Scorer


class DFlash2BeamTreeBuilder(TreeBuilder):
    """Keep every parent's top-k children before pruning the global budget.

    The reference DFlash2 beam stores all expanded children in its final
    candidate pool, while only the globally best top-k nodes form the next
    frontier. Keeping just the frontier at each depth loses useful siblings
    whenever the final tree budget exceeds the beam width.
    """

    def _frontier_width(self, vocab: int) -> int:
        return min(self.topk, vocab, self.scorer.top_k)

    def _pool_capacity(self, width: int, spec_num: int) -> int:
        return width + max(spec_num - 1, 0) * width * width

    def _expand_candidates(self, state, layer, child_depth, origin_frontier, confidence=None):
        num_reqs, width = state.tokens.shape[0], state.width
        num_parents = 1 if child_depth == 1 else width
        scores = layer.scores[:, :num_parents]
        tokens = layer.token_ids[:, :num_parents]
        confidence_candidates = None
        if confidence is not None:
            # Match the common builder's summary phase: all live candidate
            # edges, before the per-parent shortlist or frontier pruning.
            confidence_candidates = confidence.candidates(
                tokens,
                state.frontier[:, :num_parents],
                torch.ones((num_parents, scores.shape[-1]), dtype=torch.bool, device=scores.device),
                child_depth - 1,
                edge_scores=scores,
            )
        # Scorers normalize over the full selector lattice before this top-k.
        # Renormalizing the shortlist would change the reference path scores.
        top_scores, top_slots = scores.topk(width, dim=-1)
        if self.keep_origin_chain:
            origin_children = layer.origin_child[:, :num_parents]
            present = (top_slots == origin_children[..., None]).any(-1)
            top_slots[..., -1] = torch.where(present, top_slots[..., -1], origin_children)
            top_scores = scores.gather(-1, top_slots)
        top_tokens = tokens.gather(-1, top_slots)
        parent_ids = state.frontier[:, :num_parents]
        candidate_scores = state.path_scores.gather(1, parent_ids)[..., None] + top_scores
        num_added = num_parents * width
        start = state.num_nodes + 1
        end = start + num_added
        state.tokens[:, start - 1 : end - 1] = top_tokens.reshape(num_reqs, -1)
        state.depths[:, start - 1 : end - 1] = child_depth
        state.parents[:, start:end] = parent_ids.repeat_interleave(width, dim=-1)
        state.path_scores[:, start:end] = candidate_scores.reshape(num_reqs, -1)
        if confidence is not None:
            # The beam stores all P*W children, not just its next W frontier.
            # Use lattice indices to preserve confidence for every stored node.
            lattice_slots = (
                torch.arange(num_parents, device=scores.device)[None, :, None] * scores.shape[-1] + top_slots
            )
            confidence.record_selected(confidence_candidates, lattice_slots.reshape(num_reqs, -1), start)

        selected = candidate_scores.reshape(num_reqs, -1).topk(width, dim=-1).indices
        origin_pool = None
        if self.keep_origin_chain:
            child_slots = (top_slots == origin_children[..., None]).long().argmax(-1)
            origin_candidate = origin_frontier * width + child_slots.gather(1, origin_frontier[:, None])[:, 0]
            present = (selected == origin_candidate[:, None]).any(-1)
            selected[:, -1] = torch.where(present, selected[:, -1], origin_candidate)
            origin_frontier = (selected == origin_candidate[:, None]).long().argmax(-1)
            origin_pool = start + origin_candidate
        state.frontier = start + selected
        return num_added, origin_frontier, origin_pool

    def build(
        self,
        draft_logits,
        out: TreeLayout,
        *,
        root_token_ids,
        draft_hidden=None,
        draft_candidate_ids=None,
        score_summary: TreeScoreSummary | None = None,
        full_vocab_size: int | None = None,
        verification_confidence_source: str = "base_full_vocab",
        base_confidence_logits: torch.Tensor | None = None,
    ):
        num_reqs, depth, vocab = draft_logits.shape
        if score_summary is not None:
            if type(self.scorer) is not DFlash2Scorer:
                raise TreeLayoutError("DFlash2 confidence requires the exact admitted lattice scorer")
            target_vocab = self.scorer.predecessor_codebook.shape[0]
            if type(full_vocab_size) is not int or full_vocab_size != target_vocab:
                raise TreeLayoutError("DFlash2 confidence requires the proven target-vocabulary size")
            if verification_confidence_source not in ("base_full_vocab", "scorer_logprob"):
                raise TreeLayoutError("unsupported DFlash2 confidence source")
            if verification_confidence_source == "base_full_vocab" and (
                base_confidence_logits is None
                or base_confidence_logits.shape != (num_reqs, depth, target_vocab)
                or base_confidence_logits.device != draft_logits.device
            ):
                raise TreeLayoutError("DFlash2 base confidence requires independent complete base logits")
        width = self._frontier_width(vocab)
        if self.keep_origin_chain and self.budget < depth:
            raise ValueError("Keeping the complete origin chain requires budget >= depth")
        self._ensure_scratch(num_reqs, vocab, depth, draft_logits.device)
        capacity = self._pool_capacity(width, depth)
        state = TreeBuildState(
            self._tokens_buf[:num_reqs, :capacity],
            self._depths_buf[:num_reqs, :capacity],
            self._parents_buf[:num_reqs, : capacity + 1],
            self._path_scores_buf[:num_reqs, : capacity + 1],
            self._frontier_buf[:num_reqs, :width],
            width,
        )
        state.tokens.fill_(-1)
        state.depths.zero_()
        state.parents.zero_()
        state.path_scores.zero_()
        state.frontier.zero_()
        confidence = None
        if score_summary is not None:
            self._ensure_base_confidence_scratch()
            paths = self._base_path_scores_buf[:num_reqs, : capacity + 1]
            if verification_confidence_source == "base_full_vocab":
                confidence = prepare_base_confidence(base_confidence_logits, full_vocab_size, paths, score_summary)
            else:
                confidence = prepare_scorer_confidence(paths, score_summary)
        ctx = self.scorer.prepare(
            state,
            draft_logits,
            root_token_ids,
            draft_hidden,
            keep_origin_chain=self.keep_origin_chain,
            candidate_ids=draft_candidate_ids,
        )
        origin_frontier = (
            torch.zeros(num_reqs, dtype=torch.long, device=draft_logits.device) if self.keep_origin_chain else None
        )
        origin_pools = []
        for slot in range(depth):
            layer = self.scorer.score(state, ctx, slot, self.keep_origin_chain)
            num_added, origin_frontier, origin_pool = self._expand_candidates(
                state, layer, slot + 1, origin_frontier, confidence=confidence
            )
            if self.keep_origin_chain:
                origin_pools.append(origin_pool)
            self.scorer.update(state, ctx, num_added)
            state.num_nodes += num_added
        num_nodes = min(self.budget, state.num_nodes)
        priority = state.path_scores[:, 1 : state.num_nodes + 1]
        if self.depth_penalty:
            priority = priority - self.depth_penalty * state.depths[:, : state.num_nodes].float()
        if self.keep_origin_chain:
            priority = priority.clone()
            priority.scatter_(1, torch.stack(origin_pools, dim=1) - 1, float("inf"))
        self.num_selected_nodes = num_nodes
        return finalize_tree_layout_from_state(out, priority, state.tokens, state.depths, state.parents, num_nodes)
