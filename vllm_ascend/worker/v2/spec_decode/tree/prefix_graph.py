# SPDX-License-Identifier: Apache-2.0
"""Bounded Qwen3 tree graphs, keyed by real request count and query capacity."""

from contextlib import contextmanager
from contextvars import ContextVar

_TARGET_CAPTURE = ContextVar("tree_prefix_target_capture", default=False)


def eager_tree_verification_capacity(tree_config, max_tokens, num_reqs, num_tokens, *, decode):
    """Return the explicit eager capacity without adding requests or candidates.

    Hybrid GDN cannot use batch-invariant mode. An opt-in full-tree physical
    shape avoids changing matrix kernels as confidence trims the live prefix.
    Prefill and root-only batches retain their ordinary execution shape.
    """
    if not tree_config.verification_eager_pad_to_full_tree or not decode or num_tokens <= num_reqs:
        return None
    capacity = num_reqs * (tree_config.budget + 1)
    if num_reqs < 1 or num_tokens > capacity or capacity > max_tokens:
        raise ValueError("eager tree verification exceeds the admitted full-tree capacity")
    return capacity


def prefix_graph_configured(config) -> bool:
    tree = (getattr(config, "additional_config", None) or {}).get("tree_spec_config") or {}
    budget = tree.get("budget")
    if (
        tree.get("enabled") is not True
        or type(budget) is not int
        or budget < 1
        or tree.get("verification_confidence_source", "base_full_vocab") not in ("base_full_vocab", "scorer_logprob")
    ):
        return False
    scheduler, parallel, compilation = config.scheduler_config, config.parallel_config, config.compilation_config
    spec = config.speculative_config
    if (
        type(scheduler.max_num_seqs) is not int
        or scheduler.max_num_seqs < 1
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
    tp = getattr(parallel, "tensor_parallel_size", None)
    if (
        type(tp) is not int
        or tp < 1
        or any(
            type(getattr(parallel, key, None)) is not int or getattr(parallel, key, None) != 1
            for key in (
                "pipeline_parallel_size",
                "data_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        )
    ):
        return False
    sizes = compilation.cudagraph_capture_sizes
    if not sizes or any(type(n) is not int or not 1 <= n <= budget + 1 for n in sizes) or len(set(sizes)) != len(sizes):
        return False
    if getattr(spec, "enable_adaptive_verification", False):
        return False
    hf = spec.draft_model_config.hf_config
    draft = getattr(hf, "dflash_config", None)
    draft = draft if isinstance(draft, dict) else vars(draft) if draft is not None else {}
    if getattr(config.model_config.hf_config, "model_type", None) != "qwen3":
        return False
    # Endpoint gears are added below. Every real request must fit a full-tree
    # gear without extending the runner's existing token buffers.
    max_tokens = getattr(scheduler, "max_num_batched_tokens", None)
    if type(max_tokens) is not int or max_tokens < scheduler.max_num_seqs * (budget + 1):
        return False
    if callable(getattr(spec, "use_dspark", None)) and spec.use_dspark():
        return "Qwen3DSparkModel" in (getattr(hf, "architectures", None) or [])
    if not spec.use_dflash():
        return False
    if "DFlash2DraftModel" in (getattr(hf, "architectures", None) or []):
        return tree.get("verification_confidence_source") == "scorer_logprob"
    if draft.get("projector_type") == "domino":
        # Retain DOMINO's Scorer-only FULL boundary: Base eager was validated,
        # but Base FULL model comparisons did not establish AR agreement.
        return draft.get("shift_label") is True and tree.get("verification_confidence_source") == "scorer_logprob"
    return draft.get("projector_type") in (None, "linear")


def prefix_capture_sizes(config):
    """Per-request capacities; always cover root-only and the complete tree."""
    tree = config.additional_config["tree_spec_config"]
    return sorted({1, tree["budget"] + 1, *config.compilation_config.cudagraph_capture_sizes})


def prefix_capture_shapes(config):
    """Capture each real BS separately; never fabricate padding requests."""
    return [
        (rows, capacity)
        for rows in range(1, config.scheduler_config.max_num_seqs + 1)
        for capacity in prefix_capture_sizes(config)
    ]


def prefix_query_lens(cumulative_query_lens, num_decodes, capacity):
    if type(num_decodes) is not int or num_decodes < 1 or len(cumulative_query_lens) != num_decodes:
        raise ValueError("prefix FULL graph requires exactly the real decode requests")
    previous = 0
    result = []
    for end in cumulative_query_lens:
        if type(end) is not int or not 1 <= end - previous <= capacity:
            raise ValueError("prefix FULL query exceeds its per-request physical graph capacity")
        result.append(end - previous)
        previous = end
    return result


def prefix_shape(cumulative_query_lens, num_decodes, capacity):
    lengths = prefix_query_lens(cumulative_query_lens, num_decodes, capacity)
    return num_decodes, max(lengths)


@contextmanager
def target_prefix_capture(enabled):
    token = _TARGET_CAPTURE.set(enabled is True)
    try:
        yield
    finally:
        _TARGET_CAPTURE.reset(token)


def prefix_capture_active():
    return _TARGET_CAPTURE.get()
