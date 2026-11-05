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


def validate_runtime_facts(
    *,
    async_scheduling,
    tp,
    pp,
    max_concurrent_batches,
    bonus_count,
    enforce_eager,
    tree_graph_present,
    adaptive_manager_present,
    greedy,
    grammar,
    processors_supported,
    scorer_admitted,
):
    """Reject unsupported R1 confidence configurations with concrete reasons."""
    checks = (
        (not async_scheduling, "async_scheduling"),
        (tp == 1 and pp == 1, "parallelism"),
        (max_concurrent_batches == 1, "concurrent_batches"),
        (bonus_count == 1, "single_root"),
        (enforce_eager and not tree_graph_present, "eager"),
        (not adaptive_manager_present, "adaptive_manager"),
        (greedy, "greedy"),
        (not grammar, "grammar"),
        (processors_supported, "processors"),
        (scorer_admitted, "scorer"),
    )
    failures = tuple(name for passed, name in checks if not passed)
    if failures:
        raise ValueError("confidence unsupported: " + ",".join(failures))
