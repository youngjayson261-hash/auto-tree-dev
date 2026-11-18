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
