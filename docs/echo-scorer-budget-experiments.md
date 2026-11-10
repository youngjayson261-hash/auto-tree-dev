# Exploratory Scorer tree-budget measurements

These user-requested measurements explore confidence pruning with Scorer
probabilities. They are single-request observations, not a completed formal
P18 comparison or evidence of a statistically significant speedup. Production
source was not changed for these measurements.

## Configuration

- Target: `Qwen3-8B`; draft: `Qwen3-8B-Domino-b16`.
- Confidence source: `scorer_logprob`; depth 16, top-k 8, candidate size 64.
- Non-root tree budgets: 48 and 64. Target capacities including root are 49
  and 65 respectively. Both use the kernel's capacity padding of 64.
- TP/PP/DP: 1; one physical Ascend 910B2, logical device 0 in the owned
  experiment container; eager execution and synchronous scheduling.
- One client request at a time. `max_num_seqs=2` is engine capacity, not
  client concurrency. `max_model_len=256`, `max_num_batched_tokens=512`,
  memory utilization 0.75, prefix caching and chunked prefill disabled.
- Input: 34 tokens, asking for the integers 1 through 50 separated by commas.
  Output: 128 tokens, temperature 0, seed 0, `ignore_eos=true`.
- Each group uses a fresh process and its own cache. A one-token warmup with
  the public input is excluded from measurement and profiling.
- Four groups per budget: default fixed, confidence thresholds 0.10, 0.20
  and 0.30. The fixed configuration omits the verification-policy override.
  Each effective group was measured once without a profiler. Additional
  repetitions canceled by the user were not restored.

The frozen production source manifest is
`69f6200fcdf899a775977d26440abae7bde5741b7aa7e7d24b2253cea747d28b`.
All effective outputs match the same-source canonical 128-token output.
That checks output consistency within this protocol, not answer accuracy on
a broader workload.

## Measurements without profiling

S counts generated non-root draft nodes in proposals actually consumed by
verification. V counts their verified non-root nodes. Acceptance counts are
the sampler's raw non-root accepted tokens, excluding root and bonus, before
the final frontend output clipping. A final proposal that was generated but
never consumed is excluded from S and V.

### Budget 48

| Group | Output token/s | Request seconds | S | V | Accepted | Accepted/V | Consumed/shrunk rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed | 9.710570 | 13.181513 | 432 | 432 | 123 | 28.47% | 9/0 |
| Confidence 0.10 | 8.636505 | 14.820811 | 432 | 428 | 123 | 28.74% | 9/1 |
| Confidence 0.20 | 8.730865 | 14.660632 | 432 | 380 | 121 | 31.84% | 9/4 |
| Confidence 0.30 | 9.545797 | 13.409042 | 432 | 362 | 118 | 32.60% | 9/4 |

### Budget 64

| Group | Output token/s | Request seconds | S | V | Accepted | Accepted/V | Consumed/shrunk rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed | 9.734706 | 13.148831 | 576 | 576 | 124 | 21.53% | 9/0 |
| Confidence 0.10 | 9.640654 | 13.277107 | 576 | 560 | 124 | 22.14% | 9/1 |
| Confidence 0.20 | 9.536804 | 13.421687 | 576 | 499 | 120 | 24.05% | 9/3 |
| Confidence 0.30 | 9.578075 | 13.363854 | 576 | 467 | 118 | 25.27% | 9/4 |

Budget 64 required six attempts for four effective samples. The original
0.20 attempt ran with another service occupying the device; the original
0.30 attempt failed during memory initialization and produced no performance
sample. Both original records remain preserved and excluded. Only those two
groups were replaced; valid fixed and 0.10 samples were not repeated.

No dynamic group exceeded fixed within either budget. At budget 64, threshold
0.10 reduces V by 2.78% with the same 124 accepted tokens, but throughput is
0.97% lower in these observations. Threshold 0.30 reduces V by 18.92%, with
118 accepted tokens and throughput 1.61% below fixed. Increased Accepted/V
does not by itself establish greater output throughput. Neither repeated
samples nor randomized ordering support a significance or cross-budget
speedup claim here.

## Two requested budget-64 profiles

The user requested one profile of the fastest observed confidence group and
one fixed control after the four-group scan. Threshold 0.10 was selected from
0.10/0.20/0.30 by the unprofiled throughput above. This is an observed maximum,
not a statistical winner. Fixed and 0.10 were then each captured once, in
that order, using the same configuration with profiling enabled. There was
no additional AR or duplicate fixed performance group.

Each capture uses the original Level2 native profiler with one complete
request in a single `RECORD_AND_SAVE` interval. Each native worker completed
one save step and stopped; both driver and supervisor exited 0, without
timeout or remaining live members. Warmup was outside the capture. Both
requests produced the identical canonical 128-token output.

| Recorded observation | Fixed | Confidence 0.10 |
| --- | ---: | ---: |
| CPU generation-marker window, seconds | 87.519544 | 97.418198 |
| Target forward primary compute, summed milliseconds | 180.131939 | 180.475971 |
| Draft primary compute, summed milliseconds | 106.074202 | 117.839474 |
| Attributed non-profiler device tasks | 24,979 | 31,683 |
| Explicit handler copy-event wait markers | 0 | 10 |
| Explicit handler wait, summed milliseconds | 0 | 584.605780 |
| Asynchronous device-to-host copies / bytes | 30 / 1,352 | 40 / 1,432 |
| Synchronous device-to-host copies / bytes | 20 / 80 | 21 / 84 |
| Asynchronous device-to-device copies / bytes | 1,174 / 27,726,405 | 1,674 / 27,609,365 |

The profiler and diagnostic recording substantially increase request time.
These durations do not replace the unprofiled throughput table. Primary
compute durations are sums of recorded task intervals and can overlap across
streams; they are not request wall-time shares. CPU named ranges include
instrumentation, submission and any blocking within those ranges.

Both groups generated 640 non-root draft nodes in ten proposals. Nine
proposals, containing 576 generated nodes, were consumed. The final proposal
of 64 nodes was not consumed. Target forward also includes prefill, whereas
the draft ranges include that final generation; ten recorded ranges must
not be relabeled as ten consumed verification rounds.

In the positive-confidence capture, additional primary kernel calls include
logical operations, comparisons and gathers. For example, LogicalOr calls
increase by 780 and Greater calls by 360. These observations identify work
to investigate, without proving that a specific function caused the entire
latency difference. This was not an ablation study.

The fixed arm does not require the instrumented handler copy-event wait,
whereas the positive-confidence arm does. Zero fixed markers do not imply
an absence of all synchronization. The measured dynamic wait includes
profiling overhead and cannot be treated as production incremental cost.

All TASK rows are inside each device's own recording window. Four fixed and
eight dynamic tail TASK intervals extend beyond their CPU marker and remain
explicitly recorded. This does not establish CPU/device clock alignment,
physical drain ordering, global trace completeness or universal buffer
ownership. Overall P17/P18, batch acceptance and new-budget or new-threshold
P16/native OFF/ON equivalence remain outside this result.

## Local evidence

The workspace results directory, outside the source checkout, retains the
original commands, configurations, receipts, full raw files, archives,
per-file SHA manifests and analysis. The two profiles contain 6,093 raw
members totaling 698,149,654 bytes; their archives total 74,357,699 bytes.
The local HTML report and machine-readable comparison provide direct links
to the traces and native databases.

| Artifact | SHA256 |
| --- | --- |
| Budget-48 summary v1 | `edf66d156e9bfe45d884e9cfff603f6e7f836e3ce60aebb03fc9b4a526e36f61` |
| Budget-64 summary v2 | `a8a1e37d0452a9b62cabdf9b33ff5a7bb9aa3e5afac9829f1bf8071df94438df` |
| Budget-64 effective completion v2 | `7a09ba9fb3dbe9addf4915986dbbf2fb9b0ae16aabfb5dfefce386fd09277d1f` |
| Profiling selection v1 | `e61cedc8c1cd83a7d6a4979ba05c55c6e7b8da5aaf2157742160861b2d15b3d1` |
| Fixed profiling archive | `5a0737f58996635b4cdb1a8b6c20f2bd46d2735d14abd1cb948c320c8301a940` |
| Confidence-0.10 profiling archive | `984a0e0a6d9b8f6efc50898d242ac76ecac62ef041daa27f59c9c067cf82f669` |

Budget 96 produced no performance result: its first fixed attempt failed
during initialization when `prune_tree_layout_kernel` exceeded the device's
unified-buffer limit. The remaining three groups were not run. Budget 72
was not executed after the user selected 64; it must not be reported as an
experimental failure. No kernel repair was included in this work.

Unrelated inherited CI cleanup remains excluded from ECHO. These results do
not authorize restoring reverted formatting, logger, import or naming changes.
