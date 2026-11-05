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

    max_path_logp: "Tensor"
    valid_depths: "Tensor"
    invalid_scores: "Tensor"


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
