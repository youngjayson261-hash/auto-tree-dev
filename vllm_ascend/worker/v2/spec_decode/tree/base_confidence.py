# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Full-vocabulary confidence for the candidates expanded by a tree builder."""

from dataclasses import dataclass

import torch

from .budget_policy import TreeLayoutError, TreeScoreSummary


@dataclass
class BaseConfidenceContext:
    logits: torch.Tensor
    log_normalizers: torch.Tensor
    path_scores: torch.Tensor
    summary: TreeScoreSummary

    def candidates(
        self,
        token_ids: torch.Tensor,
        frontier: torch.Tensor,
        valid: torch.Tensor,
        slot: int,
    ) -> torch.Tensor:
        """Summarize all live candidates before the builder selects children."""
        rows, depths, _ = self.logits.shape
        if (
            type(slot) is not int
            or not 0 <= slot < depths
            or token_ids.ndim != 3
            or token_ids.shape[0] != rows
            or token_ids.dtype != torch.long
            or token_ids.device != self.logits.device
        ):
            raise TreeLayoutError("invalid base-confidence candidate metadata")
        _, width, count = token_ids.shape
        if (
            width < 1
            or count < 1
            or frontier.shape != (rows, width)
            or frontier.dtype != torch.long
            or frontier.device != self.logits.device
            or valid.shape != (width, count)
            or valid.dtype != torch.bool
            or valid.device != self.logits.device
        ):
            raise TreeLayoutError("base-confidence frontier/mask mismatch")
        edge = self.logits[:, slot].float().gather(1, token_ids.reshape(rows, width * count))
        edge = edge.reshape(rows, width, count) - self.log_normalizers[:, slot, None, None]
        paths = self.path_scores.gather(1, frontier).unsqueeze(-1) + edge
        flat = paths.masked_fill(~valid.unsqueeze(0), -torch.inf).reshape(rows, width * count)
        self.summary.max_path_logp[:, slot].copy_(flat.amax(dim=-1))
        self.summary.valid_depths[:, slot].fill_(True)
        self.summary.invalid_scores.logical_or_((torch.isnan(flat) | torch.isposinf(flat) | (flat > 0)).any(dim=-1))
        return flat

    def record_selected(self, candidates: torch.Tensor, selected: torch.Tensor, start: int) -> None:
        """Follow the builder's selected parents, including forced origin nodes."""
        rows = self.logits.shape[0]
        if (
            candidates.ndim != 2
            or candidates.shape[0] != rows
            or candidates.dtype != torch.float32
            or candidates.device != self.logits.device
            or selected.ndim != 2
            or selected.shape[0] != rows
            or selected.dtype != torch.long
            or selected.device != self.logits.device
            or type(start) is not int
            or start < 1
            or start + selected.shape[1] > self.path_scores.shape[1]
        ):
            raise TreeLayoutError("invalid base-confidence selected-node metadata")
        self.path_scores[:, start : start + selected.shape[1]].copy_(candidates.gather(1, selected))


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
    expected = (
        (summary.max_path_logp, (rows, depths), torch.float32),
        (summary.valid_depths, (rows, depths), torch.bool),
        (summary.invalid_scores, (rows,), torch.bool),
    )
    if any(t.shape != shape or t.dtype != dtype or t.device != logits.device for t, shape, dtype in expected):
        raise TreeLayoutError("base-confidence summary shape/dtype/device mismatch")
    if (
        path_scores.ndim != 2
        or path_scores.shape[0] != rows
        or path_scores.shape[1] < 1
        or path_scores.dtype != torch.float32
        or path_scores.device != logits.device
    ):
        raise TreeLayoutError("base-confidence path pool shape/dtype/device mismatch")
    log_normalizers = torch.logsumexp(logits.float(), dim=-1)
    summary.max_path_logp.fill_(-torch.inf)
    summary.valid_depths.zero_()
    summary.invalid_scores.copy_((~torch.isfinite(log_normalizers)).any(dim=-1))
    path_scores.zero_()
    return BaseConfidenceContext(logits, log_normalizers, path_scores, summary)
