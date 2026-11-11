# Experimental bounded FULL graphs for the Domino tree scorer

The Qwen3 Domino tree scorer can capture a small set of Target verification
graphs and replay the smallest graph that fits the current tree. The real
query length `q` remains the root plus the actual verified nodes; graph
capacity `Q` does not change the verification budget or accepted path.

This support is experimental. Successful graph replay and component tests do
not establish autoregressive equivalence, answer accuracy, or a performance
improvement. An existing confidence-policy eager/FULL comparison produced
different output tokens. New same-input probes observe Draft context-batching
differences, but do not fully explain that original token divergence.

## Supported configuration

- Qwen3 Target and a DFlash draft with `projector_type=domino` and
  `shift_label=true`.
- Tree speculation enabled and
  `verification_confidence_source=scorer_logprob`. Both fixed and confidence
  verification use the same bounded Target graph path.
- Synchronous scheduling, `max_num_seqs=1`, and `max_concurrent_batches=1`.
- TP, PP, DP, PCP, and DCP are all 1.
- Target and Draft eager enforcement disabled; graph mode `FULL`.
- No LoRA, KV transfer, adaptive verification, or microbatches.
- One to nine distinct integer capture sizes in `[1, budget + 1]`, including
  both endpoints. For budget 64, the tested five sizes are
  `[1, 17, 33, 49, 65]`.

For example, these engine settings select the bounded graph path when the
other conditions above are satisfied:

```python
enforce_eager = False
max_num_seqs = 1
async_scheduling = False
compilation_config = {
    "mode": 0,
    "cudagraph_mode": "FULL",
    "cudagraph_capture_sizes": [1, 17, 33, 49, 65],
}
additional_config = {
    "tree_spec_config": {
        "enabled": True,
        "budget": 64,
        "topk": 4,
        "candidate_size": 64,
        "verification_confidence_source": "scorer_logprob",
        "verification_policy": "confidence",
        "verification_threshold": 0.1,
    }
}
```

Keep the model-specific DFlash configuration, draft depth, greedy sampling,
and other confidence-policy admission requirements in the existing
[confidence verification guide](echo-confidence-verification.md).
The example does not configure model weights or make unsupported deployments
eligible. The installed vLLM 0.31 runtime was exercised; compatibility with
other vLLM revisions still needs the relevant integration tests.

## Replay contracts

Normal prefill runs eagerly. During decode, dispatch must select the smallest
configured `Q >= q`, keep one real request, and reject a missing or mismatched
FULL descriptor. It must not invent a padding request.

Target FIA uses BSND storage with physical query/output capacity `Q`, real
`actual_seq_lengths=[q]`, the real KV length, and the current tree visibility
mask. Persistent mask storage is reused across transitions such as
`Q=65 -> Q=17 -> Q=65`; rows after `q` remain masked. Padding KV slots use
`PAD_SLOT_ID` and must not write any cache page.

The Target capture role is scoped and restored on exceptions. Draft capture
must not inherit the Target dummy tree mask. On vLLM 0.31 the Draft graph
includes the original context-KV prefix callback and query forward; the
remaining context is handled by the original runtime path. This is not a
query-only Draft graph. Tree construction and KV compaction retain their own
graphs. S1, S2, and S3 budget optimizations are not part of this change.

## Reproducible validation

From the repository root, the CPU contract suite runs without importing an
NPU worker. PyTorch is required for the two real tensor-mask tests; absence
of PyTorch is reported as a skip rather than a pass.

```bash
python -B tests/ut/worker/a2/test_tree_prefix_graph.py
```

On a supported runtime with exactly one isolated visible NPU, the native
suite checks seven full-cache padding sentinels and nine updatable FIA
eager/replay comparisons. This test allocates NPU memory; select the intended
device externally before invoking it.

```bash
python -B tests/ut/worker/a2/test_tree_prefix_graph_native.py
```

The packaged CPU suite and related admission/device-policy regressions passed
95 tests with no skips. The final packaged native suite passed all seven KV
sentinels and nine FIA comparisons with maximum FIA absolute error 0. These
fresh checks used physical card 4 on 80.48.37.148, logical device 0, TP=1, and
vLLM 0.31; CPU checks hid all accelerator devices.

Separate Qwen3-8B/Qwen3-8B-Domino-b16 model diagnostics used one GSM8K prompt,
greedy generation of 128 tokens, and scorer threshold 0.30. Three Target
rounds compared identical inputs/history at `q/Q = 65/65, 17/17, 41/49`.
Graph replay, eager execution with `Q` rows, and eager execution with `q`
rows produced identical hidden states and logits. All 82 prefix-KV tensors
in the shared metadata remained unchanged in each observed round.

Three Draft rounds compared the graph with eager execution that retained
the runtime's 17-row context prefix and its already computed context suffix.
Hidden states, logits, and all ten Draft KV tensors matched exactly. Computing
the entire context in one eager call instead produced maximum logit absolute
error 0.125 in all three rounds; sampled base-logit argmax rows stayed equal.
This is evidence of a context-batching numerical difference, not proof of
equivalent tree scores, generation, or the original divergence's complete cause.
Both diagnostic outputs matched the preserved FULL smoke output. Six saved
raw tensor records were independently rechecked on CPU, and both processes
exited 0 without timing out. Runtime finalizers also logged an invalid process
group during cleanup; the selected card was checked idle afterward.

The original fixed-policy eager/FULL smoke comparison produced the same 128
tokens for one input. The confidence-policy comparison at threshold 0.30
first differed at output index 67 (zero-based). Both completed 128 tokens;
FULL Target, Draft, and Tree replay were observed. Since both Target and
Draft execution modes changed, this comparison alone cannot attribute the
difference to Target padding, Draft computation, or another stage.

For full model-level equivalence, retain the first differing step's identical
input prefix, tree layout, logits, mask, and live KV state. Compare the same
inputs across eager and graph execution before treating the difference as
benign. Aggregate throughput, equal lengths, and FIA component tests do not
replace that check. The limited same-input checks above do not close full
generation equivalence. They include extra replay and synchronization and
must not be used as throughput measurements.

The release-note draft is
[`release_notes/drafts/echo-prefix-fullgraph.yaml`](../release_notes/drafts/echo-prefix-fullgraph.yaml).
When an actual PR number exists, move its entries into the matching numbered
fragment and add the `pr` field. Do not assign a fictitious PR number.
