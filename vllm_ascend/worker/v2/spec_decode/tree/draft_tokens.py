# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish synchronous tree proposals independently of draft-chain depth."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, NamedTuple

import torch
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokenIds, DraftTokensHandler

from vllm_ascend.worker.v2.spec_decode.tree.budget_policy import PolicyStatus, validate_policy_config

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from vllm_ascend.worker.v2.spec_decode.tree.speculator import AscendTreeSpeculator
    from vllm_ascend.worker.v2.states import AscendRequestState


class _PendingBudget(NamedTuple):
    req_ids: tuple[str, ...]
    slots: tuple[int, ...]
    packet: torch.Tensor


class TreeDraftTokensHandler(DraftTokensHandler):
    """Reuse upstream publication and its single structured-output copy.

    Request-state tokens remain on device. The builder's selected-node count
    is uniform across requests and comes from host shapes, including during
    graph capture; it never reads the device ``tree.num_nodes`` buffer.

    The default path enqueues the packet D2H copy at seal time (right after
    production is published) and synchronizes only when the scheduler actually
    consumes the width. The copy therefore overlaps the host's output-handling
    window, and the consume-side synchronize returns immediately. The consumed
    packet is the current step's own production, so the tree/width pairing is
    identical to a fully synchronous readback.
    """

    def __init__(
        self,
        device: torch.device,
        req_states: AscendRequestState,
        speculator: AscendTreeSpeculator,
        *,
        verification_policy: str = "fixed",
        threshold: float = 0.0,
        budget_stats=None,
        budget_placeholder_schedule: bool = False,
    ) -> None:
        validate_policy_config(verification_policy, threshold)
        super().__init__(device)
        self.req_states = req_states
        self.speculator = speculator
        self._budget_enabled = verification_policy == "confidence" and threshold > 0
        self._budget_stats = budget_stats
        self._pending_budget: _PendingBudget | None = None
        self._budget_sealed = False
        self._placeholder_mode = budget_placeholder_schedule is True
        self._placeholder_outstanding: _PendingBudget | None = None
        # Per-step true-width trace (env-gated): every delivery/patch appends
        # one jsonl row per request so true acceptance rates can be computed
        # offline without a profiler.
        self._width_trace_path = os.environ.get("VLLM_ASCEND_BUDGET_WIDTH_TRACE") or None
        self._width_trace_file = None
        self._width_trace_step = 0
        # Per-step overhead timing trace (env-gated): each placeholder resolve
        # flushes one jsonl row with the cost of every lifecycle segment
        # (seal/publish/patch-wait/patch-work/trace/handoff/admission).
        self._timing_trace_path = os.environ.get("VLLM_ASCEND_BUDGET_TIMING_TRACE") or None
        self._timing_trace_file = None
        self._timing_step = 0
        self._timing_pending: dict = {}
        # Stats-correction handoff for placeholder mode (env-gated): the worker
        # drops this step's true widths here so the core-side scheduler stats
        # patch can count real widths instead of placeholder widths.
        self._width_patch_dir = os.environ.get("VLLM_ASCEND_TREE_BUDGET_PATCH_DIR") or None
        if self._budget_enabled:
            self._budget_cpu = torch.empty((req_states.max_num_reqs, 2), dtype=torch.int32, pin_memory=True)
            self._budget_produced = torch.npu.Event()

    def set_draft_tokens(self, input_batch: InputBatch, draft_tokens: torch.Tensor) -> None:
        if self._budget_enabled:
            if input_batch.has_structured_output_reqs:
                raise ValueError("confidence does not support structured output")
            if self._pending_budget is not None:
                raise RuntimeError("confidence budget is still pending")
            req_ids = tuple(input_batch.req_ids)
            slots = tuple(int(idx) for idx in input_batch.idx_mapping_np)
            packet = self.speculator.tree_budget_packet[: len(req_ids)]
            if packet.shape != (len(req_ids), 2) or packet.dtype != torch.int32:
                raise ValueError("invalid confidence budget packet")
            if len(slots) != len(req_ids) or len(set(req_ids)) != len(req_ids) or len(set(slots)) != len(slots):
                raise ValueError("invalid confidence request mapping")
            self._pending_budget = _PendingBudget(req_ids, slots, packet)
            self._budget_sealed = False
            return
        if draft_tokens.shape[1] == 0:
            # Preserve the upstream skip_drafts contract instead of publishing
            # the previous tree's selected nodes at the output-length boundary.
            super().set_draft_tokens(input_batch, draft_tokens)
            return
        # MRV2 passes only K columns after #57053. The completed tree proposal
        # is already in request state, where storage retains its B-column width.
        num_nodes = self.speculator.tree_shape.validate_selected_nodes(self.speculator.tree_builder.num_selected_nodes)
        tokens = self.req_states.draft_tokens[input_batch.idx_mapping, :num_nodes]
        super().set_draft_tokens(input_batch, tokens)

    def seal_budget_production(self) -> None:
        if not self._budget_enabled or self._pending_budget is None:
            return
        if self._budget_sealed:
            raise RuntimeError("confidence budget already sealed")
        t0 = time.perf_counter() if self._timing_trace_path else None
        self._budget_produced.record()
        self._seal_budget_copy()
        self._budget_sealed = True
        if t0 is not None:
            self._timing_pending["seal_ms"] = (time.perf_counter() - t0) * 1e3

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if not self._budget_enabled:
            result = super().get_draft_tokens()
            if self._budget_stats is not None and self._budget_stats.enabled and result is not None:
                self._budget_stats.record_fixed_delivery(self.speculator.tree_builder, result, int(PolicyStatus.FIXED))
            return result
        pending = self._pending_budget
        if pending is None:
            return None
        if not self._budget_sealed:
            raise RuntimeError("confidence budget production is not sealed")
        if self._placeholder_mode:
            # Publish the full tree width without reading the device packet:
            # the engine and scheduler never block on the current step's
            # production. ``resolve_placeholder_budget`` patches the true width
            # into the scheduler output right before the step executes, so the
            # executed shapes stay bit-identical to the synchronous path. The
            # packet D2H enqueued at seal time keeps overlapping underneath.
            if self._placeholder_outstanding is not None:
                raise RuntimeError("confidence placeholder publication already outstanding")
            t0 = time.perf_counter() if self._timing_trace_path else None
            self._placeholder_outstanding = pending
            full = self.speculator.budget
            result = DraftTokenIds(list(pending.req_ids), [[-1] * full for _ in pending.req_ids])
            if t0 is not None:
                self._timing_pending["publish_ms"] = (time.perf_counter() - t0) * 1e3
            return result
        try:
            rows = self._read_budget_rows(pending)
            accepted = []
            for req_id, slot, (budget, status) in zip(pending.req_ids, pending.slots, rows):
                status = PolicyStatus(status)
                if status == PolicyStatus.LAYOUT_ERROR or not 0 <= budget <= self.speculator.budget:
                    raise ValueError("invalid confidence budget or tree layout")
                current_slot = self.req_states.req_id_to_index.get(req_id)
                if current_slot is None:
                    continue
                if current_slot != slot:
                    raise RuntimeError("confidence request slot changed before publication")
                accepted.append((req_id, slot, budget))
            for _, slot, budget in accepted:
                self.req_states.tree_next_budget_cpu[slot] = budget
            if self._budget_stats is not None and self._budget_stats.enabled:
                self._budget_stats.record_budget_delivery(self.speculator.tree_builder, pending.req_ids, rows, accepted)
            self._record_width_rows(pending, rows, accepted, "sync")
            return DraftTokenIds([req_id for req_id, _, _ in accepted], [[-1] * budget for _, _, budget in accepted])
        finally:
            self._pending_budget = None
            self._budget_sealed = False

    def resolve_placeholder_budget(self, scheduler_output) -> float:
        """Patch the true confidence width into a placeholder scheduler output.

        The model runner calls this right before ``execute_model`` consumes
        the output, so batch gathering, graph dispatch and input preparation
        all observe the same per-request widths as a synchronous readback —
        the placeholder value is a start permit only and never reaches
        execution. Returns the patch wait in seconds (telemetry). No-op when
        no placeholder publication is outstanding.
        """
        outstanding = self._placeholder_outstanding
        if outstanding is None:
            return 0.0
        self._placeholder_outstanding = None
        # The packet copy was enqueued a full engine handoff ago (seal of the
        # producing step); this synchronize returns immediately in steady
        # state. LAYOUT_ERROR still raises before launch, exactly like the
        # synchronous path.
        wait_start = time.perf_counter()
        rows = self._read_budget_rows(outstanding)
        wait_s = time.perf_counter() - wait_start
        t0 = time.perf_counter() if self._timing_trace_path else None
        try:
            full = self.speculator.budget
            spec_tokens = scheduler_output.scheduled_spec_decode_tokens
            num_scheduled = scheduler_output.num_scheduled_tokens
            accepted = []
            for req_id, slot, (budget, status) in zip(outstanding.req_ids, outstanding.slots, rows):
                status = PolicyStatus(status)
                if status == PolicyStatus.LAYOUT_ERROR or not 0 <= budget <= full:
                    raise ValueError("invalid confidence budget or tree layout")
                current_slot = self.req_states.req_id_to_index.get(req_id)
                if current_slot is None:
                    continue
                if current_slot != slot:
                    raise RuntimeError("confidence request slot changed before publication")
                accepted.append((req_id, slot, budget))
            total_delta = 0
            for req_id, slot, budget in accepted:
                self.req_states.tree_next_budget_cpu[slot] = budget
                placeholder = spec_tokens.get(req_id)
                if placeholder is None:
                    # The draft is no longer scheduled (e.g. preempted between
                    # publication and execution); the width is simply unused.
                    continue
                if len(placeholder) != full or any(token != -1 for token in placeholder):
                    raise RuntimeError("confidence placeholder publication corrupted before patch")
                if budget == full:
                    continue
                del placeholder[budget:]
                delta = full - budget
                if num_scheduled.get(req_id) != 1 + full:
                    raise RuntimeError("confidence placeholder accounting drifted before patch")
                num_scheduled[req_id] = 1 + budget
                total_delta += delta
            scheduler_output.total_num_scheduled_tokens -= total_delta
            if self._budget_stats is not None and self._budget_stats.enabled:
                self._budget_stats.record_budget_delivery(
                    self.speculator.tree_builder, outstanding.req_ids, rows, accepted
                )
            if t0 is not None:
                self._timing_pending["patch_work_ms"] = (time.perf_counter() - t0) * 1e3
                t0 = time.perf_counter()
            self._record_width_rows(outstanding, rows, accepted, "placeholder")
            if t0 is not None:
                self._timing_pending["trace_ms"] = (time.perf_counter() - t0) * 1e3
                t0 = time.perf_counter()
            self._publish_width_patch(accepted)
            if t0 is not None:
                self._timing_pending["handoff_ms"] = (time.perf_counter() - t0) * 1e3
            self._flush_timing_row(outstanding, accepted, "placeholder", wait_s)
        finally:
            self._pending_budget = None
            self._budget_sealed = False
        return wait_s

    @staticmethod
    def _trace_rank_zero() -> bool:
        """All TP ranks run the handler with identical rows; only rank 0 writes."""
        try:
            from vllm.distributed import get_tp_group

            return get_tp_group().rank == 0
        except Exception:
            return True

    def _record_width_rows(self, pending: _PendingBudget, rows: list, accepted: list, mode: str) -> None:
        """Append this step's true pruned widths to the trace jsonl (env-gated)."""
        if not self._width_trace_path or not self._trace_rank_zero():
            return
        try:
            if self._width_trace_file is None:
                self._width_trace_file = open(self._width_trace_path, "a", encoding="utf-8")
            status_by_req = {req: int(status) for req, (_, status) in zip(pending.req_ids, rows)}
            step = self._width_trace_step
            self._width_trace_step += 1
            for req_id, _, budget in accepted:
                status = status_by_req.get(req_id, -1)
                try:
                    status_name = PolicyStatus(status).name
                except ValueError:
                    status_name = "UNKNOWN"
                row = dict(ts=time.time(), step=step, req_id=req_id, budget=budget,
                           status=status, status_name=status_name, mode=mode)
                self._width_trace_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._width_trace_file.flush()
        except Exception:
            # Telemetry must never break serving.
            pass

    def _publish_width_patch(self, accepted: list) -> None:
        """Hand this step's true widths to the core-side scheduler stats patch.

        Placeholder mode publishes full-width tokens to the core, so the core's
        spec-decode stats would count placeholder widths. The true widths are
        dropped here (atomic rename) right before execution; the core consumes
        them during this step's output processing. Sync scheduling guarantees
        the ordering; functional accounting is intentionally left on the
        placeholder length (schedule-advance and rejection-rollback cancel out),
        only stats are corrected.
        """
        if not self._width_patch_dir or not self._trace_rank_zero():
            return
        try:
            payload = json.dumps({"widths": {req: budget for req, _, budget in accepted}},
                                 separators=(",", ":"))
            tmp = os.path.join(self._width_patch_dir, ".pending_widths.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, os.path.join(self._width_patch_dir, "pending_widths.json"))
        except Exception:
            # Stats correction must never break serving.
            pass

    def _flush_timing_row(self, pending: _PendingBudget, accepted: list, mode: str, wait_s: float) -> None:
        """Append one per-step overhead row (env-gated); never breaks serving."""
        if not self._timing_trace_path or not self._trace_rank_zero():
            return
        try:
            if self._timing_trace_file is None:
                self._timing_trace_file = open(self._timing_trace_path, "a", encoding="utf-8")
            row = dict(ts=time.time(), step=self._timing_step, mode=mode,
                       req_ids=list(pending.req_ids),
                       budgets=[budget for _, _, budget in accepted],
                       patch_wait_ms=wait_s * 1e3,
                       **self._timing_pending)
            self._timing_pending = {}
            self._timing_step += 1
            self._timing_trace_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._timing_trace_file.flush()
        except Exception:
            pass

    def _seal_budget_copy(self) -> None:
        """Enqueue this step's packet D2H on the copy stream without blocking.

        The packet is final once the production event completes on device, so
        the copy stream fetches it while the host finishes output handling;
        the consume-side synchronize in ``get_draft_tokens`` then finds the
        data already in the pinned buffer. The host synchronizes before the
        next step is scheduled, which keeps the persistent device packet safe
        from being overwritten mid-copy without a packet-free event.
        """
        pending = self._pending_budget
        assert pending is not None
        count = len(pending.req_ids)
        self.copy_stream.wait_event(self._budget_produced)
        with torch.npu.stream(self.copy_stream):
            self._budget_cpu[:count].copy_(pending.packet, non_blocking=True)
            pending.packet.record_stream(self.copy_stream)
            self.copy_event.record(self.copy_stream)

    def _read_budget_rows(self, pending: _PendingBudget) -> list:
        count = len(pending.req_ids)
        # The copy was enqueued at seal time; the output-handling window
        # gives it a full step's tail to complete, so this returns fast.
        self.copy_event.synchronize()
        return self._budget_cpu[:count].tolist()
