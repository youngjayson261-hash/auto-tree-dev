# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Passive host budget counts; never read a device value for logging.

The module logger's DEBUG level controls collection. Counts from production,
delivery and consumption deliberately have different cohorts. No inference
decision or publication identity is controlled by this diagnostic metadata.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from functools import wraps

from vllm.logger import init_logger

from .budget_policy import summarize_budget_counts

logger = init_logger("vllm." + __name__)


def _diagnostic(method):
    """Isolate diagnostic code only; native inference calls never enter here."""

    @wraps(method)
    def observed(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as error:
            self.note_error(error)
            return None

    return observed


@dataclass
class _DeliveredBudget:
    s: int
    a: int
    status: int
    consumed: bool = False


class TreeBudgetStats:
    """Internal counts, with no public metrics API or per-step tensor retention."""

    def __init__(self):
        self.produced_non_root = 0
        self.produced_requests = 0
        self.delivered_s_non_root = 0
        self.published_non_root = 0
        self.published_requests = 0
        self.verified_non_root = 0
        self.accepted_non_root = 0
        self.verification_requests = 0
        self.matched_counts = summarize_budget_counts([])
        self.matched_requests = 0
        self.unmatched_verifications = 0
        self.invalid_metric_rows = 0
        self.discarded_publications = 0
        self.discarded_s_non_root = 0
        self.discarded_a_non_root = 0
        self.score_fallback_batches = 0
        self.status_requests = Counter()
        self._delivered = {}
        self.diagnostic_unknown = False
        self.diagnostic_errors = 0
        self.last_diagnostic_error_type = None

    def note_error(self, error):
        # Scalar-only and sticky. Do not use the possibly failing logger, tensor
        # repr, exception formatting, or reset partially updated counts to zero.
        self.diagnostic_unknown = True
        self.diagnostic_errors += 1
        self.last_diagnostic_error_type = type(error).__name__

    @property
    def enabled(self):
        try:
            return logger.isEnabledFor(logging.DEBUG)
        except Exception as error:
            self.note_error(error)
            return False

    @_diagnostic
    def record_produced(self, s, num_reqs):
        if not self.enabled:
            return
        if type(s) is not int or type(num_reqs) is not int or min(s, num_reqs) < 0:
            self.invalid_metric_rows += 1
            return
        self.produced_non_root += s * num_reqs
        self.produced_requests += num_reqs

    @_diagnostic
    def record_published(self, s, rows):
        """Rows are (req_id,A,status), after the existing handler host delivery."""
        if not self.enabled:
            return
        rows = tuple(rows)
        req_ids = [row[0] for row in rows]
        if len(req_ids) != len(set(req_ids)):
            self.invalid_metric_rows += 1
            return
        fallback = False
        for req_id, a, status in rows:
            try:
                summarize_budget_counts([(s, a, 0, 0)])
            except (TypeError, ValueError):
                self.invalid_metric_rows += 1
                continue
            self.discard(req_id)
            self._delivered[req_id] = _DeliveredBudget(s, a, status)
            self.delivered_s_non_root += s
            self.published_non_root += a
            self.published_requests += 1
            self.status_requests[status] += 1
            fallback = fallback or status == 3
        self.score_fallback_batches += int(fallback)

    @_diagnostic
    def snapshot_consumption(self, req_ids, verified):
        """Capture immutable scalar sources before the next propose can replace them."""
        if not self.enabled:
            return ()
        if len(req_ids) != len(verified) or len(req_ids) != len(set(req_ids)):
            self.invalid_metric_rows += 1
            return ()
        return tuple((req_id, self._delivered.get(req_id), v) for req_id, v in zip(req_ids, verified))

    @_diagnostic
    def record_verified(self, cohort, req_ids, sampled_token_ids):
        """The native AsyncOutput has already synchronized and trimmed host IDs.

        Accepted counts describe raw sampler paths (one bonus/root excluded),
        before Scheduler/frontend terminal clipping. This is not API token use.
        """
        if not self.enabled:
            return
        source = {req_id: (published, v) for req_id, published, v in cohort}
        if (
            len(source) != len(cohort)
            or len(req_ids) != len(sampled_token_ids)
            or len(req_ids) != len(set(req_ids))
            or any(req_id not in source for req_id in req_ids)
        ):
            self.invalid_metric_rows += 1
            return
        for req_id, token_ids in zip(req_ids, sampled_token_ids):
            published, v = source.get(req_id, (None, None))
            accepted = max(0, len(token_ids) - 1)
            if type(v) is not int or not 0 <= accepted <= v:
                self.invalid_metric_rows += 1
                continue
            if published is not None:
                if published.consumed:
                    self.invalid_metric_rows += 1
                    continue
                try:
                    counts = summarize_budget_counts([(published.s, published.a, v, accepted)])
                except (TypeError, ValueError):
                    self.invalid_metric_rows += 1
                    continue
            self.verified_non_root += v
            self.accepted_non_root += accepted
            self.verification_requests += 1
            if published is None:
                self.unmatched_verifications += 1
                continue
            for key, value in counts.items():
                self.matched_counts[key] += value
            self.matched_requests += 1
            published.consumed = True
            if self._delivered.get(req_id) is published:
                self._delivered.pop(req_id)
        logger.debug("Tree verification budget counts %s", self.snapshot())

    @_diagnostic
    def discard(self, req_id):
        """Release only diagnostic metadata; never touch inference/request state."""
        previous = self._delivered.pop(req_id, None)
        if self.enabled and previous is not None and not previous.consumed:
            self.discarded_publications += 1
            self.discarded_s_non_root += previous.s
            self.discarded_a_non_root += previous.a

    @_diagnostic
    def capture_consumption(self, execute_model_state):
        if not self.enabled or execute_model_state is None:
            return None
        batch = execute_model_state.input_batch
        verified = getattr(batch, "tree_verified_counts_cpu", None)
        if verified is None:
            cu = batch.cu_num_logits_np
            verified = [max(0, int(cu[i + 1] - cu[i]) - 1) for i in range(batch.num_reqs)]
        return self.snapshot_consumption(batch.req_ids, verified)

    @_diagnostic
    def record_host_output(self, cohort, output):
        if self.enabled:
            self.record_verified(cohort, output.req_ids, output.sampled_token_ids)

    @_diagnostic
    def record_builder_produced(self, builder, num_reqs):
        if self.enabled:
            self.record_produced(builder.num_selected_nodes, num_reqs)

    @_diagnostic
    def record_fixed_delivery(self, builder, result, status):
        if not self.enabled or result is None:
            return
        if len(result.req_ids) != len(result.draft_token_ids):
            self.invalid_metric_rows += 1
            return
        self.record_published(
            getattr(builder, "num_selected_nodes", None),
            [(req, len(tokens), status) for req, tokens in zip(result.req_ids, result.draft_token_ids)],
        )

    @_diagnostic
    def record_budget_delivery(self, builder, req_ids, rows, accepted):
        if not self.enabled:
            return
        if len(req_ids) != len(rows):
            self.invalid_metric_rows += 1
            return
        accepted_ids = {req for req, _, _ in accepted}
        self.record_published(
            getattr(builder, "num_selected_nodes", None),
            [(req, budget, status) for req, (budget, status) in zip(req_ids, rows) if req in accepted_ids],
        )

    def snapshot(self):
        return {
            "diagnostic_unknown": self.diagnostic_unknown,
            "diagnostic_errors": self.diagnostic_errors,
            "last_diagnostic_error_type": self.last_diagnostic_error_type,
            "metrics_comparable": not self.diagnostic_unknown and self.invalid_metric_rows == 0,
            "produced_tree_non_root": self.produced_non_root,
            "produced_requests": self.produced_requests,
            "delivered_s_non_root": self.delivered_s_non_root,
            "published_non_root": self.published_non_root,
            "published_requests": self.published_requests,
            "verified_non_root": self.verified_non_root,
            "raw_sampler_accepted_non_root": self.accepted_non_root,
            "verification_requests": self.verification_requests,
            "matched_consumptions": dict(self.matched_counts),
            "matched_requests": self.matched_requests,
            "unmatched_verifications": self.unmatched_verifications,
            "invalid_metric_rows": self.invalid_metric_rows,
            "unconsumed_deliveries": len(self._delivered),
            "unconsumed_s_non_root": sum(row.s for row in self._delivered.values()),
            "unconsumed_a_non_root": sum(row.a for row in self._delivered.values()),
            "discarded_publications": self.discarded_publications,
            "discarded_s_non_root": self.discarded_s_non_root,
            "discarded_a_non_root": self.discarded_a_non_root,
            "status_requests": dict(self.status_requests),
            "score_fallback_batches": self.score_fallback_batches,
        }
