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
