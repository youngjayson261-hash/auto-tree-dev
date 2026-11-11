# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference contracts for ECHO tree verification budgets."""

from __future__ import annotations

import math
from ctypes import c_float
from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from torch import Tensor

    from .layout import TreeLayout


class PolicyStatus(IntEnum):
    FIXED = 0
    CONFIDENCE = 1
    BOOTSTRAP_FIXED = 2
    SCORE_FALLBACK = 3
    LAYOUT_ERROR = 4


class TreeLayoutError(ValueError):
    """The original selected tree or score-depth layout is inconsistent."""


class TreeScoreSummary(NamedTuple):
    """Device summary: FP32 path maxima, valid depth mask and bad-score rows."""

    max_path_logp: Tensor
    valid_depths: Tensor
    invalid_scores: Tensor


def validate_policy_config(policy: str, tau: float) -> None:
    """Validate the host configuration before publishing any tree budget."""
    if policy not in {"fixed", "confidence"}:
        raise ValueError("verification_policy must be fixed or confidence")
    if isinstance(tau, bool) or not isinstance(tau, (int, float)):
        raise ValueError("verification_threshold must be numeric")
    if not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("verification_threshold must be finite and in [0, 1]")


def validate_tree_counts(s: int, a: int, v: int, capacity: int) -> None:
    """Check full, published and consumed counts; the root is excluded."""
    if any(type(value) is not int for value in (s, a, v, capacity)):
        raise ValueError("tree counts must be host integers")
    if not 0 <= v <= a <= s <= capacity:
        raise ValueError("expected 0 <= V <= A <= S <= B")


def consecutive_kept_depth(max_path_logp, valid_depths, tau: float) -> int:
    """Keep consecutive passing depths using the device's FP32 boundary."""
    if len(max_path_logp) != len(valid_depths):
        raise TreeLayoutError("score summary lengths differ")
    validate_policy_config("confidence", tau)
    if tau == 0:
        raise ValueError("confidence threshold must be in (0, 1]")
    seen_gap = False
    for valid in valid_depths:
        if valid and seen_gap:
            raise TreeLayoutError("valid depths have a hole")
        seen_gap = seen_gap or not valid
    boundary = c_float(math.log(tau)).value
    kept = 0
    for value, valid in zip(max_path_logp, valid_depths):
        if not valid or c_float(value).value < boundary:
            break
        kept += 1
    return kept


def count_selected_prefix(depths, parents, kept_depth: int) -> int:
    """Count original selected nodes without reselection or renumbering."""
    if len(depths) != len(parents):
        raise TreeLayoutError("selected depth and parent lengths differ")
    if type(kept_depth) is not int or kept_depth < 0:
        raise TreeLayoutError("kept depth must be a nonnegative integer")
    previous_depth = 0
    for node_id, (depth, parent) in enumerate(zip(depths, parents), start=1):
        if type(depth) is not int or type(parent) is not int:
            raise TreeLayoutError("node depth and parent must be integers")
        if depth < 1 or depth < previous_depth or not 0 <= parent < node_id:
            raise TreeLayoutError("selected nodes are not an ancestor-closed depth prefix")
        expected_depth = 1 if parent == 0 else depths[parent - 1] + 1
        if depth != expected_depth:
            raise TreeLayoutError("parent depth does not match child depth")
        previous_depth = depth
    return sum(depth <= kept_depth for depth in depths)


def select_tree_prefix_reference(
    *,
    depths,
    parents,
    max_path_logp,
    valid_depths,
    invalid_scores,
    policy: str,
    tau: float,
    bootstrap: bool,
):
    """Return a non-root budget and status for one original selected tree.

    LAYOUT_ERROR is fatal for the caller; it is never a safe fallback budget.
    """
    validate_policy_config(policy, tau)
    stored = len(depths)
    try:
        count_selected_prefix(depths, parents, 0)
    except TreeLayoutError:
        return 0, PolicyStatus.LAYOUT_ERROR
    if policy == "fixed" or tau == 0:
        return stored, PolicyStatus.FIXED
    if bootstrap:
        return stored, PolicyStatus.BOOTSTRAP_FIXED
    if max_path_logp is None or valid_depths is None:
        return stored, PolicyStatus.SCORE_FALLBACK
    try:
        kept_depth = consecutive_kept_depth(max_path_logp, valid_depths, tau)
    except TreeLayoutError:
        return 0, PolicyStatus.LAYOUT_ERROR
    if stored and any(valid_depths) and max(depths) > sum(bool(valid) for valid in valid_depths):
        return 0, PolicyStatus.LAYOUT_ERROR
    bad_score = invalid_scores or any(
        valid and (math.isnan(value) or value == math.inf or value > 0)
        for value, valid in zip(max_path_logp, valid_depths)
    )
    if bad_score or (stored > 0 and not any(valid_depths)):
        return stored, PolicyStatus.SCORE_FALLBACK
    return count_selected_prefix(depths, parents, kept_depth), PolicyStatus.CONFIDENCE


def select_tree_prefix_batch_reference(rows, *, policy: str, tau: float):
    """Apply score fallback to every active confidence row in this batch."""
    decisions = [select_tree_prefix_reference(**row, policy=policy, tau=tau) for row in rows]
    score_fallback = any(status == PolicyStatus.SCORE_FALLBACK for _, status in decisions)
    if not score_fallback:
        return decisions
    return [
        (len(row["depths"]), PolicyStatus.SCORE_FALLBACK)
        if status in {PolicyStatus.CONFIDENCE, PolicyStatus.SCORE_FALLBACK}
        else (count, status)
        for row, (count, status) in zip(rows, decisions)
    ]


def admit_confidence_scorer(scorer_type, independent_type, full_vocab_proven: bool) -> bool:
    """Subclasses must establish their own probability semantics before admission."""
    return scorer_type is independent_type and full_vocab_proven


def admit_base_confidence_scorer(scorer_type, independent_type, domino_type, full_vocab_proven: bool) -> bool:
    """Admit scorers with an explicit full-vocabulary base-confidence path."""
    return full_vocab_proven is True and (scorer_type is independent_type or scorer_type is domino_type)


def admit_verification_confidence_scorer(
    scorer_type, independent_type, domino_type, full_vocab_proven: bool, source: str = "base_full_vocab"
) -> bool:
    """Admit exact adapters with proven logp semantics and global token IDs.

    In scorer_logprob mode the full-vocabulary proof establishes the current
    token-ID mapping boundary, not the normalization range of the edge scores.
    Independent uses its full distribution; Domino may use its Top-C proposal.
    Other scorers and subclasses must prove their own score and mapping contract.
    """
    return source in ("base_full_vocab", "scorer_logprob") and admit_base_confidence_scorer(
        scorer_type, independent_type, domino_type, full_vocab_proven
    )


def confidence_source_metadata(
    *, source, scorer_name, candidate_size, prefix_len, width, vocab_size, threshold
) -> dict:
    """Describe the loaded score source using host-only model/config metadata."""
    if source not in ("base_full_vocab", "scorer_logprob"):
        raise ValueError("verification_confidence_source must be base_full_vocab or scorer_logprob")
    domino = scorer_name == "domino"
    effective_candidates = None
    if domino:
        effective_candidates = width if candidate_size is None else max(width, min(candidate_size, vocab_size))
    return {
        "source": source,
        "scorer": scorer_name,
        "threshold": threshold,
        "score_semantics": "proposal_path_logprob",
        "candidate_size_configured": candidate_size if domino else None,
        "candidate_size_effective": effective_candidates,
        "candidate_size_rule": "width if candidate_size is None else max(width, min(candidate_size, vocab_size))",
        "frontier_width": width,
        "frontier_width_rule": "min(topk, vocab_size, budget)",
        "prefix_len": prefix_len if domino else 0,
        "prefix_rule": "depth_slot < prefix_len",
        "prefix_score_source": "base_full_vocab",
        "suffix_score_source": "domino_top_c" if source == "scorer_logprob" and domino else "base_full_vocab",
        "token_mapping_requirement": "full_vocab_identity",
    }


def validate_runtime_facts(
    *,
    async_scheduling,
    tp,
    pp,
    dp,
    pcp,
    dcp,
    finegrained_tp_sizes,
    max_concurrent_batches,
    bonus_count,
    enforce_eager,
    tree_graph_present,
    adaptive_manager_present,
    greedy,
    grammar,
    processors_supported,
    scorer_admitted,
    graph_admitted=False,
):
    """Reject unsupported R1 confidence configurations with concrete reasons."""
    checks = (
        (not async_scheduling, "async_scheduling"),
        (tp == 1 and pp == 1, "parallelism"),
        (type(dp) is int and dp == 1, "data_parallelism"),
        (all(type(size) is int and size == 1 for size in (pcp, dcp)), "context_parallelism"),
        (
            isinstance(finegrained_tp_sizes, tuple)
            and len(finegrained_tp_sizes) == 4
            and all(type(size) is int and size in (0, 1) for size in finegrained_tp_sizes),
            "finegrained_tp",
        ),
        (max_concurrent_batches == 1, "concurrent_batches"),
        (bonus_count == 1, "single_root"),
        ((enforce_eager and not tree_graph_present) or (graph_admitted is True and enforce_eager is False), "eager"),
        (not adaptive_manager_present, "adaptive_manager"),
        (greedy, "greedy"),
        (not grammar, "grammar"),
        (processors_supported, "processors"),
        (scorer_admitted, "scorer"),
    )
    failures = tuple(name for passed, name in checks if not passed)
    if failures:
        raise ValueError("confidence unsupported: " + ",".join(failures))


def select_tree_prefix_device(
    layout: TreeLayout,
    summary: TreeScoreSummary,
    packet_out: Tensor,
    *,
    tau: float,
    bootstrap_mask: Tensor,
) -> None:
    """Publish non-root A/status without reading device values on the host.

    ``packet_out`` is the active ``[R, 2]`` view of the caller's persistent
    int32 packet. Layout and D03 summary buffers may have additional capacity
    rows, which are ignored. Column views need not be contiguous.

    Layout errors publish A=0 and must stop the caller before verification.
    Only valid, non-bootstrap confidence rows can trigger batch score fallback.
    This selector does not alter the original tree, summary, or KV state.
    """
    import torch

    validate_policy_config("confidence", tau)
    if packet_out.ndim != 2 or packet_out.shape[1] != 2 or packet_out.dtype != torch.int32:
        raise ValueError("packet_out must be int32[R, 2]")
    r = packet_out.shape[0]
    if layout.depths.ndim != 2 or layout.parents.shape != layout.depths.shape:
        raise ValueError("depths and parents must share [capacity_R, B]")
    if layout.depths.shape[0] < r:
        raise ValueError("layout has insufficient request rows")
    if layout.num_nodes.ndim != 1 or layout.num_nodes.shape[0] < r:
        raise ValueError("num_nodes must provide one count per active request")
    integer_types = (torch.int32, torch.int64)
    layout_fields = (layout.depths, layout.parents, layout.num_nodes)
    if any(value.dtype not in integer_types for value in layout_fields):
        raise ValueError("layout counts and indices must be integral")
    device = packet_out.device
    if any(value.device != device for value in layout_fields):
        raise ValueError("layout and packet devices differ")
    if r == 0:
        return

    depths, parents = layout.depths[:r], layout.parents[:r]
    s = layout.num_nodes[:r]
    s32 = s.to(torch.int32)
    b = depths.shape[1]
    node_ids = torch.arange(1, b + 1, device=device, dtype=torch.int64).unsqueeze(0)
    valid_nodes = node_ids <= s.unsqueeze(1)
    parent_depths = torch.cat((torch.zeros((r, 1), dtype=depths.dtype, device=device), depths), dim=1)
    parent_depth = parent_depths.gather(1, parents.clamp(0, b).to(torch.int64))
    bad_nodes = ((depths < 1) | (parents < 0) | (parents >= node_ids) | (depths != parent_depth + 1)) & valid_nodes
    bad_layout = (s < 0) | (s > b) | bad_nodes.any(dim=1)
    if b > 1:
        bad_layout = bad_layout | ((depths[:, 1:] < depths[:, :-1]) & valid_nodes[:, 1:]).any(dim=1)

    if tau == 0:
        packet_out[:, 0].copy_(torch.where(bad_layout, 0, s32))
        status = torch.full((r,), int(PolicyStatus.FIXED), dtype=torch.int32, device=device)
        status.masked_fill_(bad_layout, int(PolicyStatus.LAYOUT_ERROR))
        packet_out[:, 1].copy_(status)
        return

    maxima, valid_all, invalid_all = summary
    if maxima.ndim != 2 or valid_all.shape != maxima.shape or maxima.shape[0] < r:
        raise ValueError("invalid score-summary shape")
    if maxima.dtype != torch.float32 or valid_all.dtype != torch.bool:
        raise ValueError("score-summary dtypes must be float32/bool")
    if invalid_all.ndim != 1 or invalid_all.shape[0] < r or invalid_all.dtype != torch.bool:
        raise ValueError("invalid_scores must be bool[capacity_R]")
    if bootstrap_mask.ndim != 1 or bootstrap_mask.shape[0] < r or bootstrap_mask.dtype != torch.bool:
        raise ValueError("bootstrap_mask must be bool[capacity_R]")
    if any(value.device != device for value in (maxima, valid_all, invalid_all, bootstrap_mask)):
        raise ValueError("summary/mask and packet devices differ")
    m, valid = maxima[:r], valid_all[:r]
    bootstrap = bootstrap_mask[:r]
    confidence = ~bootstrap
    dmax = m.shape[1]
    has_depths = valid.any(dim=1)
    bad_scores = invalid_all[:r] | ((s > 0) & ~has_depths)
    if dmax == 0:
        retained_depth = torch.zeros(r, dtype=torch.int32, device=device)
    else:
        depth_prefix = torch.cumprod(valid.to(torch.int32), dim=1, dtype=torch.int32)
        summary_hole = (valid & (depth_prefix == 0)).any(dim=1)
        depth_valid = valid.gather(1, (depths.clamp(1, dmax) - 1).to(torch.int64))
        insufficient_depths = (valid_nodes & ((depths > dmax) | ~depth_valid)).any(dim=1)
        bad_layout = bad_layout | (confidence & (summary_hole | (has_depths & insufficient_depths)))
        bad_values = torch.isnan(m) | torch.isposinf(m) | (m > 0)
        bad_scores = bad_scores | (bad_values & valid).any(dim=1)
        # Round log(tau), not tau: tiny positive host thresholds must not underflow.
        threshold = c_float(math.log(tau)).value
        passed = valid & (m >= threshold)
        retained_depth = torch.cumprod(passed.to(torch.int32), dim=1, dtype=torch.int32).sum(dim=1, dtype=torch.int32)

    a = (valid_nodes & (depths <= retained_depth.unsqueeze(1))).sum(dim=1, dtype=torch.int32)
    fallback_batch = (bad_scores & confidence & ~bad_layout).any()
    fallback_rows = confidence & ~bad_layout & fallback_batch
    a = torch.where(bootstrap | fallback_rows, s32, a)
    a = torch.where(bad_layout, 0, a)
    status = torch.full((r,), int(PolicyStatus.CONFIDENCE), dtype=torch.int32, device=device)
    status = torch.where(bootstrap, int(PolicyStatus.BOOTSTRAP_FIXED), status)
    status = torch.where(fallback_rows, int(PolicyStatus.SCORE_FALLBACK), status)
    status = torch.where(bad_layout, int(PolicyStatus.LAYOUT_ERROR), status)
    packet_out[:, 0].copy_(a)
    packet_out[:, 1].copy_(status)


def summarize_budget_counts(rows):
    """Sum non-root S/A/V/raw-accepted counts from one matched cohort."""
    totals = [0, 0, 0, 0]
    for s, a, v, accepted in rows:
        validate_tree_counts(s, a, v, s)
        if type(accepted) is not int or not 0 <= accepted <= v:
            raise ValueError("accepted must satisfy 0 <= accepted <= V")
        for index, value in enumerate((s, a, v, accepted)):
            totals[index] += value
    return dict(zip(("draft_non_root", "published_non_root", "verified_non_root", "accepted_non_root"), totals))
