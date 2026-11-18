import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.utils import maybe_prefix

from vllm_ascend.models.qwen3_dspark import align_draft_weights
from vllm_ascend.utils import get_rotation_path


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(f"conv_group_size={group_size} must divide hidden_size={hidden_size}.")
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(self, hidden_states: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(**conv_args, prefix=maybe_prefix(prefix, "attention_conv"))
        self.mlp_conv = DFlashGroupedConv(**conv_args, prefix=maybe_prefix(prefix, "mlp_conv"))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum("blpr,blcr->blpc", predecessors * hidden[:, :, None], successors)


@support_torch_compile
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False)
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False)
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states is [num_reqs, num_steps, H]. NPU unquantized_gemm's Dynamo
        # fake is 2D-only, so flatten to [tokens, H], project, then restore (B, L).
        # torch._check ties independently marked dynamic batch dims so
        # predecessors [B, L, K, R] can broadcast with projected hidden.
        torch._check(hidden_states.shape[0] == candidate_ids.shape[0])
        torch._check(hidden_states.shape[1] == candidate_ids.shape[1])
        hidden = self.hidden_projection(hidden_states.flatten(0, 1))
        hidden = hidden.view(*hidden_states.shape[:-1], -1)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    # vLLM #52816 switched the parent constructor from its module global to
    # this class factory. Declare the main-lane factory so DFlash2 decoder
    # layers are built.
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )

        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(draft_config.get("input_embedding_scale", 1.0))
        # Draft load uses set_model_tag("eagle_head"); without a distinct tag the
        # selector's @support_torch_compile backend shares that npugraph cache and
        # unpacks the draft graph (67 values) with the selector's 9 inputs.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    # vLLM #52816 likewise routes the draft model through this class factory.
    model_cls = DFlash2Qwen3Model

    # Share the target LM head so compute_candidates can top-k the full vocab.
    has_own_lm_head = False
    base_logits_parallel_contract = "raw_get_logits_rankcat_v1"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        draft_config = self.config.dflash_config
        self.output_multiplier = float(draft_config.get("output_multiplier", 1.0))
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.final_logit_softcapping = softcap if softcap > 0 else None

    def post_process(self, vllm_config: VllmConfig) -> None:
        if get_rotation_path(vllm_config) is None:
            return
        if not hasattr(self.model, "fc"):
            raise ValueError("DFlash2 with a QuaRot target requires auxiliary hidden states.")
        align_draft_weights(self, self.model.fc, vllm_config)

    def compute_candidates(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.lm_head.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError("DFlash2 requires an unquantized target LM head for candidate TopK.")

        selector = self.model.candidate_selector
        logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)
        num_pad = self.lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        values, ids = torch.topk(logits, selector.top_k, dim=-1)
        ids = ids.to(torch.int64) + self.lm_head.shard_indices.org_vocab_start_index

        if get_tensor_model_parallel_world_size() > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = torch.topk(values, selector.top_k, dim=-1)
            ids = ids.gather(-1, selected)

        values = values.float() * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            values = torch.tanh(values / cap) * cap
        return ids, values

    def compute_candidates_with_base_logits(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the original lattice plus explicitly full-vocabulary base logits.

        The correction head still selects the tree. These base logits are an
        independent confidence source, not a full-vocabulary correction score.
        Keep the ordinary two-result interface and its fixed-policy work intact.
        """
        tp_size = get_tensor_model_parallel_world_size()
        if type(tp_size) is not int or tp_size not in (1, 2):
            raise ValueError("DFlash2 full-vocabulary base confidence requires TP=1 or TP=2")
        if tp_size == 2 and (not hasattr(self, "lm_head") or not hasattr(self, "logits_processor")):
            raise ValueError("DFlash2 TP2 base confidence requires loaded head/processor proof")
        if not isinstance(self.lm_head.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError("DFlash2 requires an unquantized target LM head for candidate TopK.")
        vocab_size = self.config.vocab_size
        top_k = self.model.candidate_selector.top_k
        if tp_size == 1:
            if self.lm_head.shard_indices.org_vocab_start_index != 0:
                raise ValueError("DFlash2 base confidence requires the complete identity-mapped target vocabulary")
            logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)
            num_pad = self.lm_head.shard_indices.num_org_vocab_padding
            if num_pad > 0:
                logits[..., -num_pad:] = -float("inf")
            values, ids = torch.topk(logits, top_k, dim=-1)
            ids = ids.to(torch.int64)
            if logits.shape[-1] - num_pad != vocab_size:
                raise ValueError("DFlash2 base confidence LM-head width does not match target vocabulary")
            logits = logits[..., :vocab_size]
        else:
            # All checks here use loaded host metadata, before any collective.
            # The caller also proves these facts for the actual TP group.
            head, processor = self.lm_head, self.logits_processor
            shard = head.shard_indices
            rank = get_tensor_model_parallel_rank()
            partition = getattr(head, "num_embeddings_per_partition", None)
            padded_vocab = getattr(head, "num_embeddings_padded", None)
            head_group = getattr(head, "parallel_group", None)
            if (
                type(rank) is not int
                or not 0 <= rank < tp_size
                or type(partition) is not int
                or partition < 1
                or type(padded_vocab) is not int
                or padded_vocab != partition * tp_size
                or type(vocab_size) is not int
                or not partition < vocab_size <= padded_vocab
                or type(top_k) is not int
                or not 1 <= top_k <= vocab_size - partition
                or type(getattr(head, "tp_size", None)) is not int
                or head.tp_size != tp_size
                or type(getattr(head, "tp_rank", None)) is not int
                or head.tp_rank != rank
                or head_group is not None
                or getattr(head, "disable_tp", False) is not False
            ):
                raise ValueError("DFlash2 TP2 base confidence has incompatible parallel group/shard dimensions")
            start, end = rank * partition, min((rank + 1) * partition, vocab_size)
            facts = (
                (getattr(head, "num_embeddings", None), vocab_size),
                (getattr(head, "org_vocab_size", None), vocab_size),
                (getattr(head, "num_added_embeddings", None), 0),
                (getattr(head, "org_vocab_size_padded", None), padded_vocab),
                (getattr(processor, "vocab_size", None), vocab_size),
                (getattr(processor, "org_vocab_size", None), vocab_size),
                (getattr(shard, "padded_org_vocab_start_index", None), start),
                (getattr(shard, "padded_org_vocab_end_index", None), start + partition),
                (getattr(shard, "org_vocab_start_index", None), start),
                (getattr(shard, "org_vocab_end_index", None), end),
                (getattr(shard, "num_org_vocab_padding", None), partition - (end - start)),
            )
            if (
                any(type(value) is not int or value != expected for value, expected in facts)
                or getattr(processor, "use_all_gather", None) is not True
                or getattr(processor, "logits_as_input", None) is not False
                or not callable(getattr(processor, "_get_logits", None))
                or getattr(self, "draft_id_to_target_id", None) is not None
                or head.weight.shape[0] != partition
            ):
                raise ValueError("DFlash2 TP2 base confidence requires contiguous target shards and raw all-gather")

            # _get_logits performs one head GEMM and one full-vocab gather.
            # Calling processor.forward would apply a different scale/softcap
            # order; calling compute_candidates here would repeat the GEMM.
            logits = processor._get_logits(hidden_states, head, None)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.shape != (hidden_states.shape[0], vocab_size)
                or logits.dtype != head.weight.dtype
                or logits.device != hidden_states.device
                or not logits.is_floating_point()
            ):
                raise ValueError("DFlash2 TP2 base confidence requires complete raw logits on every rank")

            # Recreate each original local TopK input, including its padding
            # and dtype. A direct full-vocab TopK could change BF16 tie IDs.
            # Every rank already owns full logits, so this needs no collective.
            rank_values, rank_ids = [], []
            for source_rank in range(tp_size):
                first = source_rank * partition
                real_count = min(partition, vocab_size - first)
                local = torch.empty(
                    (hidden_states.shape[0], partition), dtype=logits.dtype, device=logits.device
                )
                local[:, :real_count].copy_(logits[:, first : first + real_count])
                if real_count < partition:
                    local[:, real_count:].fill_(-float("inf"))
                local_values, local_ids = torch.topk(local, top_k, dim=-1)
                rank_values.append(local_values)
                rank_ids.append(local_ids.to(torch.int64) + first)
            values, selected = torch.topk(torch.cat(rank_values, dim=-1), top_k, dim=-1)
            ids = torch.cat(rank_ids, dim=-1).gather(-1, selected)

        values = values.float() * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            values = torch.tanh(values / cap) * cap

        # Slice padding before softcap: tanh(-inf) would turn padding into
        # finite -cap scores and incorrectly add it to the denominator.
        base_logits = logits.float() * self.output_multiplier
        if self.final_logit_softcapping is not None:
            base_logits = torch.tanh(base_logits / cap) * cap
        return ids, values, base_logits
