# ECHO confidence verification: evidence and remaining work

This record separates Base correctness from Scorer probability calibration.
The accompanying [evidence index](echo-validation/2026-11-09-summary.json) pins
the actual reports, raw archive hashes and source versions. Original raw tensor
archives remain in the experiment workspace; this Git repository contains an
index, not those archives. Results below are scoped to the recorded eager,
single-card, greedy protocol.

## Verified results on 2026-11-09

| Run | Result | Scope |
| --- | --- | --- |
| Base a5 on production code `a0f8a105` | P14/P15/P16 passed; issues empty | Base full-vocabulary correctness only |
| Base r5 off12 candidate | 12 modes, 83 cases; P14/P15/P16 passed; issues empty | Same-source candidate with observation disabled |
| Scorer r5 run2 | Strict probability calibration passed; issues empty | Calibration only; formal threshold remains unselected |

Both Base acceptance replays checked the original tensors and lifecycle/stop
proofs. Their source, raw data and proof files remained unchanged. The a5 archive
contains 3,212 verified files. The candidate off12 archive contains 3,398.
Neither Base result grants correctness to the Scorer confidence source.

The isolated r5 candidate has 4,728 source files and manifest
`69f6200fcdf899a775977d26440abae7bde5741b7aa7e7d24b2253cea747d28b`.
It contains pending P17 work and is distinct from production commit `a0f8a105`.
This documentation change does not apply that runtime candidate.

## What the Scorer calibration establishes

The model run used Qwen3-8B with Qwen3-8B-Domino-b16, one visible 910B2 device,
TP=1, PP=1, DP=1, eager execution and synchronous scheduling. The recorded
Scorer contract has full vocabulary 151,936, effective candidate count 64,
BF16 corrections and actual protected prefix length 1. The calibration uses
16 draft steps, tree width 8, budget 48 and trial threshold 0.5.

Four publications supplied four full-vocabulary prefix proofs and 60 complete
candidate-suffix proofs. Three consecutive consumed rounds had positive V;
two consumers naturally shortened the tree. The frozen comparison passed
without source or raw-data changes. CPU source checks passed 65 tests, and the
staged harness passed another 26, all with zero skips.

The first run's strict replay failure is retained. Its adapter incorrectly
expected an int64 root token tensor while the native input buffer is int32.
A separate v5 oracle corrected this exact type contract before the new run2
model launch. The new pre-run freeze is recorded in the evidence index. Old
raw diagnostics do not convert the first run into a successful calibration.

Run2 calibration does not establish AR output parity, full stop/commit state,
request reordering, preemption or slot reuse for Scorer. It does not validate
graph mode, random sampling, independent neural-kernel arithmetic or complete
weight contents. Threshold 0.5 is a trial value, not a selected production
threshold. Scorer P14/P15/P16 and both P17/P18 completion flags remain false.

## Reproduction and original reports

The experiment workspace is `results/echo-all-p-20261109` in the parent
inference-development project. Commands below run from that workspace with
the recorded Python environment. They replay completed raw data on CPU and
never start another model:

```text
python -X utf8 -B run_p17_r5_base_off_comparison_v1.py status
python -X utf8 -B run_scorer_raw_replay_v2.py status
```

Full reports are identified by workspace-relative paths, sizes and SHA-256
hashes in the evidence index. The comparators were frozen before consuming
the corresponding new model results. A successful model process alone was
not used as an acceptance result. Download failures and original partial
files were preserved; recovered archives passed exact SHA and member checks.

## Remaining P17 and P18 gates

The Base on12 counterpart must pass the same correctness comparator and the
paired metric/semantic checks. Native profiling must separately establish
operation, synchronization and D2H equivalence on actual model traces.
Tiny profiler controls and CPU metric tests do not grant that result.

Scorer still needs its own complete correctness and lifecycle matrix. Its
single-request calibration capture limits cannot be reused to truncate
long or multiple-request formal acceptance runs.

P18 has not run. Any performance conclusion requires the fixed matched
workload, cache initial state, warmup, repeated measurements and original
logs after correctness and P17 admission. There is no measured speedup claim.

## CI and change boundaries

Full repository CI still has separately recorded inherited failures. Those
failures do not grant permission to restore reverted cleanup. **Unrelated CI
cleanup must not be mixed back into ECHO**, including formatting unrelated
files, logger replacement, import cleanup, lint exemptions or C++ renaming.
Current feature regressions must be fixed within their actual scope.
