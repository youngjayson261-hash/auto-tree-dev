# Qwen3.8 tree decoding: validated source series

## 1. P20-only selector ACLGraph

Capture the existing budget selection formula into a separate selector ACLGraph.
Prepare bindings after tree graph capture and fall back to the original selector
when a binding is not prepared or graph execution is unavailable. The bootstrap
mask is still copied each update: P19 caching is deliberately excluded.

Historical component validation: CPU manager/runtime tests and 108 real-NPU
dynamic packet comparisons. The original Qwen3/DOMINO three-pair performance
experiment included P19 on both sides; it does not establish P20-only end-to-end
speedup. The P20-only path is also present in the later frozen Qwen3.8 baseline.
CPU fake graph execution proves lifecycle/selector math, not native graph replay.

## 2. Qwen3.8 / DFlash2 TP2 and bounded FULL adaptation

Port the tested candidate-v011 adaptation: model/admission compatibility, TP2
budget broadcast, mRoPE positions, legacy graph interfaces, and explicit FULL
decode capture/dispatch capacities [1, 9, 17, 25, 33]. Includes the guarded
single-request hybrid path and leaves unsupported configurations on fallback.

Frozen source manifest SHA256:
`4093e9e64e4e3437a66b78c928f47b953d5f8bc5960e0dcf060d04a8768181b0`.
This experimental snapshot excludes P19. It was the original baseline in the
completed P25 formal comparison. Standalone NPU probes are manual diagnostics;
their historical card defaults do not authorize running them on new resources.

## 3. P25-A: reuse cropped decode depths

Reuse the existing tree depth metadata for eligible cropped decode work instead
of rebuilding it in every cache group. Retain the original fallback and physical
state buffer semantics. The source delta is two production files and one CPU
contract test, against candidate-v011.

Frozen manifest SHA256:
`58bcc65569595002c3ce6d1ee3a098879040865ce7a785ee3806eac5ce74f2e1`.
Three FULL128 correctness cases and three historical profile256 cases passed.
The later paired formal experiment includes this exact A implementation.

## 4. P25-B: share cropped GDN logical metadata per step

Share eligible BS1 actual FULL-decode topology/depth/conv/history/chunk layout
within a step, while binding physical pages independently for each cache group.
Preserve Q <= C <= S (Q includes root), fixed S chunk/buffer capacity and clearing
of invalid slots. Prefill, dummy/capture and root-only paths keep fallback.
Inactive state/init page slots use the installed NULL_BLOCK_ID (0), rather than
assuming that it equals the slot padding value (-1). No rollback modification,
P19 bootstrap cache or new production device-to-host copy is included.

Frozen manifest SHA256:
`970f4f5c4951d3a53ee495cfcc6976a0b0570b4df94dedac20f552d514393ff4`.
Recorded candidate CPU validation: 103 passed. The four ported repository
contract modules pass 101 local checks; the shared-contract default source path
was changed from its historical experiment folder to the current checkout.
B Fixed/tau010/tau020 FULL128 passed.
Independent final coverage on each rank exercised Q=[33,29,24,33,9,17,25,33],
C=[33,33,25,33,9,17,25,33], with 8 native FULL returns and 8 checked state commits.
C1/root-only and BS>1 optimization are not claimed.

### Formal performance and counters

Physical cards 2/6; original/A/B x Fixed/tau010/tau020 x two independent processes
x three requests = 18 processes / 54 requests. Input32768/output128, warmup
excluded, no profiler/state hooks; alternating order with round two reversed.
TP2, D15/B32/topk8, candidate16, BS1, FULL[1,9,17,25,33], memory0.90,
greedy seed42, prefix cache and async disabled. Target Qwen3.8-27B and
draft Qwen3.8-27B-DFlash2.

| Strategy | Original TPOT ms | A TPOT ms | B TPOT ms | B vs original |
| --- | ---: | ---: | ---: | ---: |
| Fixed | 11.4184 | 11.3529 | 11.4074 | -0.10% |
| tau0.10 | 14.2538 | 13.2384 | 12.1764 | -14.57% |
| tau0.20 | 16.0473 | 13.9807 | 12.0612 | -24.84% |

Two rounds are descriptive repeats, not statistical significance. B tau0.20
still has 5.63% higher TPOT than original Fixed. Per-version/strategy totals
(six requests) are 66 verification rounds; non-root logical candidates
2112/2046/1858 and accepted-before-trimming counts 752/752/732 for
Fixed/tau010/tau020 respectively. These exclude warmup; candidates exclude FULL
padding. Accepted counters cannot reconstruct the draft/target composition of
the final output. Full per-process and group data are under
`docs/benchmark_results/p25-20261118/`.

### Profiling and correctness boundaries

Exactly two requested profiles on cards1/3: original Fixed vs B tau0.20,
32768/256; each rank has 12 decode / 12 actual FULL / 0 active prefill steps,
full256 token equality to pinned AR256, independent offline analysis and input
SHA checks. Original target uses C33x12 per rank; B uses C33x8/C25x4. Profiling
changes both budget policy and implementation and is not a pure P25 ablation.
Formal per-call replay counts were not instrumented. All 54 formal outputs
match across versions per prompt; only prompt0 additionally has independent
AR256-prefix128 evidence. Internal AR K/V numeric equality remains unclosed
(`internal_state_equal_to_ar_validated=false`).

One final whole-change review and targeted replay-coverage fix are complete.
Fixed-buffer packing, cropped mask, old graph-internal index flattening and
budget consumer waiting remain documented follow-up directions. Host durations,
nested spans, overlapping kernels and uncovered trace intervals are not summed
as end-to-end cost or treated as hardware idle time. No P24 or paused older
proposal is included in this series.
