# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ascend-local adaptation of an already existing host output handoff."""

from vllm.v1.outputs import AsyncModelRunnerOutput


class TreeBudgetOutput(AsyncModelRunnerOutput):
    """Delegate native synchronization and copies, then count returned host IDs."""

    def __init__(self, original, stats, cohort):
        self._original = original
        self._stats = stats
        self._cohort = cohort
        self._reported = False

    def get_output(self):
        output = self._original.get_output()
        if not self._reported:
            self._reported = True
            self._stats.record_host_output(self._cohort, output)
        return output
