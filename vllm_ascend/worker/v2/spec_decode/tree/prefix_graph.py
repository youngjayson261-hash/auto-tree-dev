# SPDX-License-Identifier: Apache-2.0
"""Static admission for bounded single-request tree FULL graph gears."""

from contextlib import contextmanager
from contextvars import ContextVar

_TARGET_CAPTURE = ContextVar("tree_prefix_target_capture", default=False)


def qwen35_dflash2_prefix_configured(config) -> bool:
    """Recognize a compatible hybrid candidate, not a captured runtime graph.

    Query dispatch must still prove the current graph descriptor, buffer
    capacity and recurrent-state layout. No user capability flag is evidence.
    """
    model_config = getattr(config, "model_config", None)
    target = getattr(model_config, "hf_config", None)
    text = getattr(model_config, "hf_text_config", None)
    spec = getattr(config, "speculative_config", None)
    hf = getattr(getattr(spec, "draft_model_config", None), "hf_config", None)
    draft_model_type = getattr(hf, "model_type", None)
    if draft_model_type == "eagle":
        # vLLM wraps DFlash in EAGLEConfig while retaining the original in
        # .model. Validate that origin, then validate the effective metadata
        # below; this static candidate still needs the loaded runtime proof.
        original = getattr(hf, "model", None)
        if getattr(original, "architectures", None) != ["DFlash2DraftModel"]:
            return False
        draft_model_type = getattr(original, "model_type", None)
    if (
        getattr(model_config, "is_hybrid", None) is not True
        or getattr(target, "model_type", None) != "qwen3_5"
        or getattr(target, "architectures", None) != ["Qwen3_5ForConditionalGeneration"]
        or getattr(text, "model_type", None) != "qwen3_5_text"
        or draft_model_type != "qwen3"
        or getattr(hf, "architectures", None) != ["DFlash2DraftModel"]
        or getattr(hf, "is_causal", None) is not False
    ):
        return False
    target_layers = getattr(text, "num_hidden_layers", None)
    draft_layers = getattr(hf, "num_hidden_layers", None)
    if any(
        type(value) is not int or value < 1
        for value in (
            target_layers,
            draft_layers,
            getattr(text, "hidden_size", None),
            getattr(text, "vocab_size", None),
        )
    ):
        return False
    if any(
        type(getattr(hf, draft_name, None)) is not int
        or getattr(hf, draft_name) != getattr(text, target_name)
        for draft_name, target_name in (
            ("num_target_layers", "num_hidden_layers"),
            ("hidden_size", "hidden_size"),
            ("vocab_size", "vocab_size"),
        )
    ):
        return False
    layer_types = getattr(text, "layer_types", None)
    if (
        not isinstance(layer_types, (list, tuple))
        or len(layer_types) != target_layers
        or any(type(layer) is not str for layer in layer_types)
        or set(layer_types) != {"linear_attention", "full_attention"}
        or getattr(hf, "layer_types", None) != ["sliding_attention"] * draft_layers
    ):
        return False
    draft = getattr(hf, "dflash_config", None)
    draft = draft if isinstance(draft, dict) else getattr(draft, "__dict__", {})
    target_ids = draft.get("target_layer_ids")
    return (
        type(draft.get("block_size")) is int
        and draft["block_size"] == 8
        and type(draft.get("selector_top_k")) is int
        and 1 <= draft["selector_top_k"] <= text.vocab_size
        and type(draft.get("selector_rank")) is int
        and draft["selector_rank"] > 0
        and isinstance(target_ids, (list, tuple))
        and len(target_ids) == draft_layers
        and all(type(layer) is int and 0 <= layer < target_layers for layer in target_ids)
        and len(set(target_ids)) == len(target_ids)
    )


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
    hybrid_candidate = qwen35_dflash2_prefix_configured(config)
    # GDN supports uniform decode graphs while prefill remains eager. This
    # candidate still requires the captured descriptor and recurrent-state proof.
    graph_modes = ("FULL", "FULL_DECODE_ONLY") if hybrid_candidate else ("FULL",)
    if (
        scheduler.max_num_seqs != 1
        or scheduler.async_scheduling
        or config.model_config.enforce_eager
        or spec is None
        or spec.enforce_eager
        or getattr(compilation.cudagraph_mode, "name", None) not in graph_modes
        or getattr(config, "max_concurrent_batches", None) != 1
        or getattr(config, "lora_config", None) is not None
        or getattr(config, "kv_transfer_config", None) is not None
    ):
        return False
    tp = getattr(parallel, "tensor_parallel_size", None)
    if type(tp) is not int or tp not in ((1, 2) if hybrid_candidate else (1,)):
        return False
    if any(
        type(getattr(parallel, key, None)) is not int
        or getattr(parallel, key) != 1
        for key in (
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
    # Hybrid configuration is only a candidate. Runtime query transport must
    # additionally check the current captured descriptor and state layout.
    if hybrid_candidate:
        return True
    # Preserve the established dense Qwen3 + DOMINO path; a hybrid or DFlash2
    # model cannot masquerade as that implementation with a projector flag.
    if getattr(config.model_config, "is_hybrid", False) is not False:
        return False
    if "DFlash2DraftModel" in (getattr(hf, "architectures", None) or ()):
        return False
    return (
        getattr(config.model_config.hf_config, "model_type", None) == "qwen3"
        and draft.get("projector_type") == "domino"
        and draft.get("shift_label") is True
    )


def prefix_graph_runtime_proven(config, speculator, tp_group=None) -> bool:
    """Require a live loaded-object proof for the hybrid candidate only.

    Descriptor/capacity/state validation is still required at graph dispatch.
    The established dense DOMINO path keeps its existing admission contract.
    """
    if not prefix_graph_configured(config):
        return False
    if getattr(config.model_config.hf_config, "model_type", None) != "qwen3_5":
        return True
    from vllm_ascend.worker.v2.spec_decode.tree.confidence_admission import valid_prefix_graph_proof

    proof = getattr(speculator, "_tree_dflash2_prefix_proof", None)
    if not valid_prefix_graph_proof(proof, config.parallel_config.tensor_parallel_size, tp_group):
        return False
    hf = config.speculative_config.draft_model_config.hf_config
    draft = getattr(hf, "dflash_config", None)
    draft = draft if isinstance(draft, dict) else getattr(draft, "__dict__", {})
    return (
        proof.vocab_size == config.model_config.hf_text_config.vocab_size
        and proof.normalization_size == draft.get("selector_top_k")
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
