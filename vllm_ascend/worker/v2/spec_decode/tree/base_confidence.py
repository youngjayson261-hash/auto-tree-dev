# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Confidence sources and shared path accumulation for tree candidates."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .budget_policy import TreeLayoutError, TreeScoreSummary


class ConfidenceContext(ABC):
    """Accumulate normalized edges independently of tree selection priorities."""

    path_scores: torch.Tensor
    summary: TreeScoreSummary

    @abstractmethod
    def edge_log_probs(self, token_ids: torch.Tensor, edge_scores: torch.Tensor | None, slot: int) -> torch.Tensor:
        """Return FP32 log probabilities with the same shape as token_ids."""
        raise NotImplementedError

    def candidates(
        self,
        token_ids: torch.Tensor,
        frontier: torch.Tensor,
        valid: torch.Tensor,
        slot: int,
        *,
        edge_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Summarize all live candidates before the builder selects children."""
        rows = self.path_scores.shape[0]
        depths = self.summary.max_path_logp.shape[1]
        device = self.path_scores.device
        if (
            type(slot) is not int
            or not 0 <= slot < depths
            or token_ids.ndim != 3
            or token_ids.shape[0] != rows
            or token_ids.dtype != torch.long
            or token_ids.device != device
        ):
            raise TreeLayoutError("invalid confidence candidate metadata")
        _, width, count = token_ids.shape
        if (
            width < 1
            or count < 1
            or frontier.shape != (rows, width)
            or frontier.dtype != torch.long
            or frontier.device != device
            or valid.shape != (width, count)
            or valid.dtype != torch.bool
            or valid.device != device
        ):
            raise TreeLayoutError("confidence frontier/mask mismatch")
        edge = self.edge_log_probs(token_ids, edge_scores, slot)
        paths = self.path_scores.gather(1, frontier).unsqueeze(-1) + edge
        flat = paths.masked_fill(~valid.unsqueeze(0), -torch.inf).reshape(rows, width * count)
        self.summary.max_path_logp[:, slot].copy_(flat.amax(dim=-1))
        self.summary.valid_depths[:, slot].fill_(True)
        # A positive edge is invalid even when a negative parent hides it in
        # the accumulated path. Masked padding must not invalidate a request.
        invalid = torch.isnan(edge) | torch.isposinf(edge) | (edge > 0)
        invalid.logical_or_(torch.isnan(paths) | torch.isposinf(paths) | (paths > 0))
        invalid.logical_and_(valid.unsqueeze(0))
        self.summary.invalid_scores.logical_or_(invalid.reshape(rows, width * count).any(dim=-1))
        return flat

    def record_selected(self, candidates: torch.Tensor, selected: torch.Tensor, start: int) -> None:
        """Follow the builder's selected parents, including forced origin nodes."""
        rows = self.path_scores.shape[0]
        device = self.path_scores.device
        if (
            candidates.ndim != 2
            or candidates.shape[0] != rows
            or candidates.dtype != torch.float32
            or candidates.device != device
            or selected.ndim != 2
            or selected.shape[0] != rows
            or selected.dtype != torch.long
            or selected.device != device
            or type(start) is not int
            or start < 1
            or start + selected.shape[1] > self.path_scores.shape[1]
        ):
            raise TreeLayoutError("invalid confidence selected-node metadata")
        self.path_scores[:, start : start + selected.shape[1]].copy_(candidates.gather(1, selected))


@dataclass
class BaseConfidenceContext(ConfidenceContext):
    # Retain the original field order for callers constructing this context.
    logits: torch.Tensor
    log_normalizers: torch.Tensor
    path_scores: torch.Tensor
    summary: TreeScoreSummary

    def edge_log_probs(self, token_ids: torch.Tensor, edge_scores: torch.Tensor | None, slot: int) -> torch.Tensor:
        rows, width, count = token_ids.shape
        edge = self.logits[:, slot].float().gather(1, token_ids.reshape(rows, width * count))
        return edge.reshape(rows, width, count) - self.log_normalizers[:, slot, None, None]


@dataclass
class ScorerConfidenceContext(ConfidenceContext):
    """Use the scorer's normalized, parent-specific candidate log probabilities."""

    path_scores: torch.Tensor
    summary: TreeScoreSummary

    def edge_log_probs(self, token_ids: torch.Tensor, edge_scores: torch.Tensor | None, slot: int) -> torch.Tensor:
        if (
            not isinstance(edge_scores, torch.Tensor)
            or edge_scores.shape != token_ids.shape
            or not edge_scores.is_floating_point()
            or edge_scores.device != self.path_scores.device
        ):
            raise TreeLayoutError("scorer-confidence edge shape/dtype/device mismatch")
        return edge_scores.float()


def _reset_confidence_buffers(
    path_scores: torch.Tensor,
    summary: TreeScoreSummary,
    rows: int,
    depths: int,
    device: torch.device,
) -> None:
    """Validate metadata and reset the caller-owned active buffer views."""
    expected = (
        (summary.max_path_logp, (rows, depths), torch.float32),
        (summary.valid_depths, (rows, depths), torch.bool),
        (summary.invalid_scores, (rows,), torch.bool),
    )
    if any(t.shape != shape or t.dtype != dtype or t.device != device for t, shape, dtype in expected):
        raise TreeLayoutError("confidence summary shape/dtype/device mismatch")
    if (
        path_scores.ndim != 2
        or path_scores.shape[0] != rows
        or path_scores.shape[1] < 1
        or path_scores.dtype != torch.float32
        or path_scores.device != device
    ):
        raise TreeLayoutError("confidence path pool shape/dtype/device mismatch")
    summary.max_path_logp.fill_(-torch.inf)
    summary.valid_depths.zero_()
    summary.invalid_scores.zero_()
    path_scores.zero_()


def prepare_scorer_confidence(path_scores: torch.Tensor, summary: TreeScoreSummary) -> ScorerConfidenceContext:
    """Reset active path/summary views without normalizing or reading logits."""
    if path_scores.ndim != 2 or summary.max_path_logp.ndim != 2:
        raise TreeLayoutError("scorer-confidence path/summary dimensions mismatch")
    _reset_confidence_buffers(
        path_scores, summary, path_scores.shape[0], summary.max_path_logp.shape[1], path_scores.device
    )
    return ScorerConfidenceContext(path_scores, summary)


def prepare_base_confidence(
    logits: torch.Tensor,
    full_vocab_size: int,
    path_scores: torch.Tensor,
    summary: TreeScoreSummary,
) -> BaseConfidenceContext:
    """Reset caller-owned buffers and normalize the explicitly proven vocabulary."""
    if (
        logits.ndim != 3
        or not logits.is_floating_point()
        or type(full_vocab_size) is not int
        or full_vocab_size < 1
        or logits.shape[-1] != full_vocab_size
    ):
        raise TreeLayoutError("base confidence requires proven full-vocabulary logits")
    rows, depths, _ = logits.shape
    _reset_confidence_buffers(path_scores, summary, rows, depths, logits.device)
    log_normalizers = torch.logsumexp(logits.float(), dim=-1)
    summary.invalid_scores.copy_((~torch.isfinite(log_normalizers)).any(dim=-1))
    return BaseConfidenceContext(logits, log_normalizers, path_scores, summary)
