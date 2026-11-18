# SPDX-License-Identifier: Apache-2.0
"""Host-only proofs for the loaded DFlash2 confidence score interface."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DFlash2ConfidenceProof:
    vocab_size: int
    score_domain: str
    normalization_size: int
    tp_size: int = 1
    tp_rank: int = 0
    tp_group: object = field(default=None, repr=False, compare=False)
    purpose: str = "confidence"


def valid_tp2_confidence_proof(proof) -> bool:
    """Accept only an internal proof still bound to its live default TP group."""
    return (
        type(proof) is DFlash2ConfidenceProof
        and proof.purpose == "confidence"
        and type(proof.tp_size) is int
        and proof.tp_size == 2
        and type(proof.tp_rank) is int
        and proof.tp_rank in (0, 1)
        and type(getattr(proof.tp_group, "world_size", None)) is int
        and proof.tp_group.world_size == 2
        and type(getattr(proof.tp_group, "rank_in_group", None)) is int
        and proof.tp_group.rank_in_group == proof.tp_rank
    )


def valid_prefix_graph_proof(proof, tp_size, tp_group) -> bool:
    """Keep hybrid prefix replay bound to the actual loaded default TP group."""
    return (
        type(tp_size) is int
        and tp_size in (1, 2)
        and type(proof) is DFlash2ConfidenceProof
        and proof.purpose == "prefix_full"
        and proof.score_domain == "selector_lattice"
        and type(proof.tp_size) is int
        and proof.tp_size == tp_size
        and proof.tp_group is tp_group
        and type(proof.tp_rank) is int
        and 0 <= proof.tp_rank < tp_size
        and type(getattr(tp_group, "world_size", None)) is int
        and tp_group.world_size == tp_size
        and type(getattr(tp_group, "rank_in_group", None)) is int
        and tp_group.rank_in_group == proof.tp_rank
    )


def _parallel_layout_supported(head, processor, tp_group, tp_size, vocab_size, top_k, unquantized_head_type):
    rank = getattr(tp_group, "rank_in_group", None)
    if (
        type(getattr(tp_group, "world_size", None)) is not int
        or tp_group.world_size != tp_size
        or type(rank) is not int
        or not 0 <= rank < tp_size
        or type(getattr(head, "tp_size", None)) is not int
        or head.tp_size != tp_size
        or type(getattr(head, "tp_rank", None)) is not int
        or head.tp_rank != rank
        or getattr(head, "parallel_group", None) is not None
    ):
        return False
    shard = getattr(head, "shard_indices", None)
    if tp_size == 1:
        return (
            type(getattr(shard, "org_vocab_start_index", None)) is int
            and shard.org_vocab_start_index == 0
            and type(getattr(shard, "org_vocab_end_index", None)) is int
            and shard.org_vocab_end_index == vocab_size
        )
    partition = getattr(head, "num_embeddings_per_partition", None)
    if (
        type(partition) is not int
        or partition < 1
        or getattr(head, "disable_tp", None) is not False
        or not isinstance(unquantized_head_type, type)
        or not isinstance(getattr(head, "quant_method", None), unquantized_head_type)
        or getattr(processor, "use_all_gather", None) is not True
        or getattr(processor, "logits_as_input", None) is not False
        or not callable(getattr(processor, "_get_logits", None))
    ):
        return False
    expected = {
        "num_added_embeddings": 0,
        "num_embeddings": vocab_size,
        "org_vocab_size": vocab_size,
        "org_vocab_size_padded": partition * tp_size,
        "num_embeddings_padded": partition * tp_size,
    }
    if any(type(getattr(head, name, None)) is not int or getattr(head, name) != value for name, value in expected.items()):
        return False
    padded_start, padded_end = rank * partition, (rank + 1) * partition
    start, end = min(padded_start, vocab_size), min(padded_end, vocab_size)
    expected_shard = {
        "padded_org_vocab_start_index": padded_start,
        "padded_org_vocab_end_index": padded_end,
        "org_vocab_start_index": start,
        "org_vocab_end_index": end,
        "num_org_vocab_padding": partition - (end - start),
    }
    if any(
        type(getattr(shard, name, None)) is not int or getattr(shard, name) != value
        for name, value in expected_shard.items()
    ):
        return False
    shape = getattr(getattr(head, "weight", None), "shape", ())
    return (
        len(shape) == 2
        and type(shape[0]) is int
        and shape[0] == partition
        and vocab_size <= partition * tp_size
        and all(
            min((r + 1) * partition, vocab_size) - min(r * partition, vocab_size) >= top_k
            for r in range(tp_size)
        )
    )


def prove_dflash2_confidence(
    model,
    scorer,
    builder,
    *,
    model_type,
    scorer_type,
    builder_type,
    target_vocab_size,
    confidence_source,
    tensor_parallel_size,
    tp_group=None,
    unquantized_head_type=None,
    purpose="confidence",
):
    """Return positive loaded-object evidence, never an architecture-only flag.

    This checks token-ID identity and the selector's candidate domain. It does
    not call an NPU tensor, read a tensor value or prove hybrid graph support.
    The builder separately validates the actual [R,D,V] full-logit payload.
    """
    if (
        type(model) is not model_type
        or type(scorer) is not scorer_type
        or type(builder) is not builder_type
        or getattr(builder, "scorer", None) is not scorer
        or type(target_vocab_size) is not int
        or target_vocab_size < 1
        or type(tensor_parallel_size) is not int
        or tensor_parallel_size not in (1, 2)
        or confidence_source not in ("base_full_vocab", "scorer_logprob")
        or purpose not in ("confidence", "prefix_full")
        or (purpose == "prefix_full" and confidence_source != "scorer_logprob")
    ):
        return None
    config = getattr(model, "config", None)
    processor = getattr(model, "logits_processor", None)
    head = getattr(model, "lm_head", None)
    if (
        getattr(config, "architectures", None) != ["DFlash2DraftModel"]
        or any(
            type(value) is not int or value != target_vocab_size
            for value in (
                getattr(config, "vocab_size", None),
                getattr(config, "draft_vocab_size", target_vocab_size),
                getattr(processor, "vocab_size", None),
                getattr(processor, "org_vocab_size", None),
            )
        )
        or getattr(model, "draft_id_to_target_id", None) is not None
    ):
        return None
    selector = getattr(getattr(model, "model", None), "candidate_selector", None)
    top_k = getattr(selector, "top_k", None)
    if type(top_k) is not int or not 1 <= top_k <= target_vocab_size:
        return None
    if not _parallel_layout_supported(
        head, processor, tp_group, tensor_parallel_size, target_vocab_size, top_k, unquantized_head_type
    ):
        return None
    shapes = []
    for name in ("predecessor_codebook", "successor_codebook"):
        codebook = getattr(selector, name, None)
        shape = getattr(codebook, "shape", ())
        if (
            getattr(scorer, name, None) is not codebook
            or len(shape) != 2
            or type(shape[0]) is not int
            or shape[0] != target_vocab_size
            or type(shape[1]) is not int
            or shape[1] < 1
        ):
            return None
        shapes.append(tuple(shape))
    if (
        shapes[0] != shapes[1]
        or type(getattr(scorer, "top_k", None)) is not int
        or getattr(scorer, "top_k", None) != top_k
        or getattr(scorer, "hidden_projection", None) is not getattr(selector, "hidden_projection", None)
        or not callable(getattr(selector, "hidden_projection", None))
        or not callable(getattr(model_type, "compute_candidates", None))
        or not callable(getattr(scorer_type, "confidence_metadata", None))
    ):
        return None
    metadata = scorer.confidence_metadata()
    if (
        not isinstance(metadata, dict)
        or metadata.get("normalization_domain") != "selector_lattice"
        or metadata.get("token_id_domain") != "target_vocab"
        or type(metadata.get("normalization_size")) is not int
        or metadata.get("normalization_size") != top_k
        or type(metadata.get("target_vocab_size")) is not int
        or metadata.get("target_vocab_size") != target_vocab_size
    ):
        return None
    if confidence_source == "scorer_logprob":
        return DFlash2ConfidenceProof(
            target_vocab_size, "selector_lattice", top_k, tensor_parallel_size, tp_group.rank_in_group, tp_group, purpose
        )
    if not callable(getattr(model_type, "compute_candidates_with_base_logits", None)):
        return None
    if tensor_parallel_size == 2 and vars(model_type).get("base_logits_parallel_contract") != "raw_get_logits_rankcat_v1":
        return None
    return DFlash2ConfidenceProof(
        target_vocab_size, "base_full_vocab", target_vocab_size, tensor_parallel_size, tp_group.rank_in_group, tp_group, purpose
    )
