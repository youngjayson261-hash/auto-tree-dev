# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish synchronous tree proposals independently of draft-chain depth."""

from __future__ import annotations

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
    """

    def __init__(
        self,
        device: torch.device,
        req_states: AscendRequestState,
        speculator: AscendTreeSpeculator,
        *,
        verification_policy: str = "fixed",
        threshold: float = 0.0,
    ) -> None:
        validate_policy_config(verification_policy, threshold)
        super().__init__(device)
        self.req_states = req_states
        self.speculator = speculator
        self._budget_enabled = verification_policy == "confidence" and threshold > 0
        self._pending_budget: _PendingBudget | None = None
        self._budget_sealed = False
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
        self._budget_produced.record()
        self._budget_sealed = True

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if not self._budget_enabled:
            return super().get_draft_tokens()
        pending = self._pending_budget
        if pending is None:
            return None
        if not self._budget_sealed:
            raise RuntimeError("confidence budget production is not sealed")
        try:
            count = len(pending.req_ids)
            self.copy_stream.wait_event(self._budget_produced)
            with torch.npu.stream(self.copy_stream):
                self._budget_cpu[:count].copy_(pending.packet, non_blocking=True)
                pending.packet.record_stream(self.copy_stream)
                self.copy_event.record(self.copy_stream)
            self.copy_event.synchronize()
            rows = self._budget_cpu[:count].tolist()
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
            return DraftTokenIds([req_id for req_id, _, _ in accepted], [[-1] * budget for _, _, budget in accepted])
        finally:
            self._pending_budget = None
            self._budget_sealed = False
