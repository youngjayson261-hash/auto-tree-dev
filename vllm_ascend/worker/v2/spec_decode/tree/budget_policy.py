# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference contracts for ECHO tree verification budgets."""

from __future__ import annotations

import math
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
