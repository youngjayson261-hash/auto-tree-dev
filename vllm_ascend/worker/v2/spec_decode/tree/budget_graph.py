# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup-only capture of the reference budget selector, with bound buffers."""

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from .budget_policy import TreeScoreSummary, select_tree_prefix_device, validate_policy_config


@dataclass(frozen=True)
class SelectorBinding:
    layout: Any
    summary: TreeScoreSummary
    bootstrap_mask: torch.Tensor
    packet: torch.Tensor
    tau: float

    @property
    def num_reqs(self) -> int:
        return self.packet.shape[0]

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.layout.depths,
            self.layout.parents,
            self.layout.num_nodes,
            *self.summary,
            self.bootstrap_mask,
            self.packet,
        )

    def signature(self) -> tuple:
        return (
            self.num_reqs,
            self.tau,
            tuple((t.data_ptr(), tuple(t.shape), t.stride(), t.dtype, t.device) for t in self.tensors()),
        )


@dataclass
class _SelectorGraphEntry:
    binding: SelectorBinding
    signature: tuple
    state: str = "EMPTY"
    reason: str = "not prepared"
    graph: Any = None


class _NpuGraphBackend:
    """No device initialization at import; used only during explicit preparation."""

    def supports(self, device: torch.device) -> bool:
        api = getattr(torch, "npu", None)
        return device.type in ("npu", "privateuseone") and all(
            hasattr(api, name)
            for name in (
                "NPUGraph",
                "graph",
                "is_current_stream_capturing",
                "synchronize",
                "memory_allocated",
                "memory_reserved",
            )
        )

    def is_capturing(self) -> bool:
        return torch.npu.is_current_stream_capturing()

    def new_graph(self):
        return torch.npu.NPUGraph()

    def capture(self, graph, *, pool):
        return torch.npu.graph(graph, pool=pool)

    def synchronize(self, device):
        torch.npu.synchronize(device)

    def memory(self, device):
        return torch.npu.memory_allocated(device), torch.npu.memory_reserved(device)


class TreeBudgetSelectorGraphRunner:
    """One entry per prepared R. Calls on the request path never capture/sync.

    The owner must serialize preparation with all requests and retain the
    existing producer/consumer stream ordering. An entry retains all buffers;
    its graph caches operations, never score values or budget results.
    """

    def __init__(self, *, backend=None):
        self._backend = _NpuGraphBackend() if backend is None else backend
        self._entries: dict[int, _SelectorGraphEntry] = {}
        self._replays = 0
        self._misses: Counter = Counter()
        self._startup: list[dict] = []

    @staticmethod
    def _run(binding: SelectorBinding) -> None:
        select_tree_prefix_device(
            binding.layout,
            binding.summary,
            binding.packet,
            tau=binding.tau,
            bootstrap_mask=binding.bootstrap_mask,
        )

    def prepare(self, binding: SelectorBinding, *, pool) -> bool:
        """Prepare only at startup or a caller-guaranteed quiescent point.

        Only an explicit unsupported operation may recover after capture exit,
        packet restoration and synchronization. Other errors remain fatal.
        """
        validate_policy_config("confidence", binding.tau)
        rows = binding.num_reqs
        if rows == 0 or binding.tau == 0:
            return False
        signature = binding.signature()
        if not self._backend.supports(binding.packet.device):
            self._entries[rows] = _SelectorGraphEntry(binding, signature, "UNAVAILABLE", "backend unavailable")
            return False
        if self._backend.is_capturing():
            raise RuntimeError("budget selector preparation cannot nest inside capture")
        previous = self._entries.get(rows)
        if previous is not None and previous.state == "READY" and previous.signature == signature:
            return True
        entry = _SelectorGraphEntry(binding, signature)
        self._entries[rows] = entry
        started = time.perf_counter()
        memory_before = self._backend.memory(binding.packet.device)
        with torch.inference_mode():
            saved_packet = binding.packet.clone()
            try:
                self._run(binding)
                self._backend.synchronize(binding.packet.device)
                entry.graph = self._backend.new_graph()
                with self._backend.capture(entry.graph, pool=pool):
                    self._run(binding)
            except NotImplementedError:
                entry.state, entry.reason = "UNAVAILABLE", "capture unsupported"
            except BaseException:
                entry.state, entry.reason = "STALE", "preparation failed"
                raise
            finally:
                # If capture did not exit, recovery operations are unsafe.
                if self._backend.is_capturing():
                    entry.state, entry.reason = "STALE", "capture did not exit"
                    raise RuntimeError("budget selector capture did not exit safely")
                binding.packet.copy_(saved_packet)
                self._backend.synchronize(binding.packet.device)
        memory_after = self._backend.memory(binding.packet.device)
        self._startup.append(
            dict(
                num_reqs=rows,
                seconds=time.perf_counter() - started,
                allocated_delta_bytes=memory_after[0] - memory_before[0],
                reserved_delta_bytes=memory_after[1] - memory_before[1],
            )
        )
        if entry.state == "UNAVAILABLE":
            return False
        entry.state, entry.reason = "READY", "prepared"
        return True

    def try_replay(self, binding: SelectorBinding) -> bool:
        rows = binding.num_reqs
        if rows == 0 or binding.tau == 0:
            self._misses["inactive policy or empty batch"] += 1
            return False
        entry = self._entries.get(rows)
        if entry is None:
            self._misses["unprepared rows"] += 1
            return False
        if entry.state != "READY":
            self._misses[entry.reason] += 1
            return False
        if entry.signature != binding.signature():
            self.invalidate(reason="binding changed", num_reqs=rows)
            self._misses["binding changed"] += 1
            return False
        # Never retry eagerly after a replay starts: it can have partial writes.
        entry.graph.replay()
        self._replays += 1
        return True

    def invalidate(self, *, reason: str, num_reqs: int | None = None) -> None:
        entries = self._entries.values() if num_reqs is None else (self._entries.get(num_reqs),)
        for entry in entries:
            if entry is not None:
                entry.state, entry.reason = "STALE", reason

    def diagnostics(self) -> dict:
        """CPU-only snapshot; memory deltas of a shared pool are not additive."""
        return dict(
            entries={r: dict(state=e.state, reason=e.reason) for r, e in self._entries.items()},
            prepared_rows=sorted(r for r, e in self._entries.items() if e.state == "READY"),
            replays=self._replays,
            eager_misses=dict(self._misses),
            startup=[dict(record) for record in self._startup],
        )
