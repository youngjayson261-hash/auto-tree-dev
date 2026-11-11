# SPDX-License-Identifier: Apache-2.0
"""Bounded graph gears for the single-request Qwen3 Domino tree path."""

from contextlib import contextmanager
from contextvars import ContextVar

_TARGET_CAPTURE = ContextVar("tree_prefix_target_capture", default=False)


def prefix_graph_configured(config) -> bool:
    tree = (getattr(config, "additional_config", None) or {}).get("tree_spec_config") or {}
    budget = tree.get("budget")
    if (
        tree.get("enabled") is not True
        or type(budget) is not int
        or budget < 1
        or tree.get("verification_confidence_source") != "scorer_logprob"
    ):
        return False
    scheduler, parallel, compilation = config.scheduler_config, config.parallel_config, config.compilation_config
    spec = config.speculative_config
    if (
        scheduler.max_num_seqs != 1
        or scheduler.async_scheduling
        or config.model_config.enforce_eager
        or spec is None
        or spec.enforce_eager
        or getattr(compilation.cudagraph_mode, "name", None) != "FULL"
        or getattr(config, "max_concurrent_batches", None) != 1
        or getattr(config, "lora_config", None) is not None
        or getattr(config, "kv_transfer_config", None) is not None
    ):
        return False
    if any(
        getattr(parallel, key, None) != 1
        for key in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    ):
        return False
    sizes = compilation.cudagraph_capture_sizes
    if (
        not sizes
        or len(sizes) > 9
        or any(type(n) is not int or not 1 <= n <= budget + 1 for n in sizes)
        or len(set(sizes)) != len(sizes)
        or 1 not in sizes
        or budget + 1 not in sizes
    ):
        return False
    if not spec.use_dflash() or getattr(spec, "enable_adaptive_verification", False):
        return False
    hf = spec.draft_model_config.hf_config
    draft = getattr(hf, "dflash_config", None)
    draft = draft if isinstance(draft, dict) else vars(draft) if draft is not None else {}
    return (
        getattr(config.model_config.hf_config, "model_type", None) == "qwen3"
        and draft.get("projector_type") == "domino"
        and draft.get("shift_label") is True
    )


def prefix_shape(cumulative_query_lens, num_decodes, capacity):
    if (
        num_decodes != 1
        or len(cumulative_query_lens) != 1
        or type(cumulative_query_lens[0]) is not int
        or not 1 <= cumulative_query_lens[0] <= capacity
    ):
        raise ValueError("prefix FULL graph requires one real query within physical graph capacity")
    return 1, cumulative_query_lens[0]


@contextmanager
def target_prefix_capture(enabled):
    token = _TARGET_CAPTURE.set(enabled is True)
    try:
        yield
    finally:
        _TARGET_CAPTURE.reset(token)


def prefix_capture_active():
    return _TARGET_CAPTURE.get()
