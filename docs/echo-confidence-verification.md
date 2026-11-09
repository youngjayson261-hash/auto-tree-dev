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
| Base r5 on12 candidate | 12 modes, 83 cases; P14/P15/P16 passed; issues empty | Same-source candidate with observation enabled |
| Base r5 paired replay | Logical counters and paired semantics passed | Native model profiling remains required |
| Scorer r5 run2 | Strict probability calibration passed; issues empty | Calibration only; formal threshold remains unselected |

All three Base acceptance replays checked the original tensors and lifecycle/stop
proofs. Their source, raw data and proof files remained unchanged. The a5 archive
contains 3,212 verified files. Each candidate off12/on12 archive contains 3,398.
No Base result grants correctness to the Scorer confidence source.

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

## Formal Scorer failure and diagnostic repair

The separate Scorer formal run1 captured 12 modes and 83 generation cases.
Its strict replay completed with exit 2 and seven issues. The six positive
threshold groups each recorded Scheduler order 2 followed by Worker prepare
order 1. An independently checked audit found two identical observer modules
with separate order and completed-get registries. The final lifetime gate
therefore rejected the evidence. This establishes a capture defect; it does
not by itself establish a model arithmetic failure.

The first CPU attempt timed out after 900 seconds. The single-thread replay
then completed in 1,755 seconds without timeout. Both attempts are retained.
Source, raw-data and oracle hashes remained unchanged. Scorer P14/P15/P16
remain ungranted.

A new diagnostic loader reuses the verified canonical module and checks both
source copies, the loaded path and function bindings. Nine regression tests
and 108 checks in the actual staged layout passed without skips; independent
review found no open P0/P1/P2 within capture preparation. A separate run2 uses
the same experiment and frozen oracle. Its formal acceptance remains pending.

## Native model save validation

The run3 schema diagnostic exited 0 in 367.3 seconds and recovered 4,037
original files, totaling 499,245,393 bytes, with exact archive and member
hash checks. The same profiler instance entered RECORD_AND_SAVE, advanced
once to NONE and stopped. The previous Stop RECORD warning was absent;
both SQLite databases returned `ok` from `PRAGMA quick_check`.

These facts verify the single-save path, not complete native attribution or
observer equivalence. A profiler no-warmup warning and optional-data export
warnings are retained in the evidence index. No model D2H, synchronization,
NPU-work equivalence or performance conclusion is granted by this diagnostic.

## Reproduction and original reports

The experiment workspace is `results/echo-all-p-20261109` in the parent
inference-development project. Commands below run from that workspace with
the recorded Python environment. They replay completed raw data on CPU and
never start another model:

```text
python -X utf8 -B run_p17_r5_base_off_comparison_v1.py status
python -X utf8 -B run_p17_r5_base_on_comparison_v1.py status
python -X utf8 -B run_scorer_raw_replay_v2.py status
python -X utf8 -B run_scorer_formal_r5_run1_comparison_v3.py status
```

Full reports are identified by workspace-relative paths, sizes and SHA-256
hashes in the evidence index. The comparators were frozen before consuming
the corresponding new model results. A successful model process alone was
not used as an acceptance result. Download failures and original partial
files were preserved; recovered archives passed exact SHA and member checks.

## Remaining P17 and P18 gates

The Base off12 and on12 counterparts passed the same correctness comparator.
The paired replay reports matching outputs, native state semantics and counters
across all 83 cases. Native profiling must still establish operation,
synchronization and D2H equivalence on actual model traces.

The v5 tiny calibration identified one 16,384-byte synchronous D2H copy and
one 16,384-byte direct asynchronous D2H copy. Its source-bound zero-copy marker
had zero copies; 26 CPU checks passed without skips. This calibrated marker
does not establish zero-copy behavior for arbitrary model traces or complete P17.

Scorer still needs its own complete correctness and lifecycle matrix. Its
independently reviewed full harness passed 257 CPU checks and another 99 checks
in the actual staged layout, all without skips. Formal model acceptance remains
pending. Prospective limits require complete logits for every publication, with
up to two request rows, 512 publications and 16 GiB of full logits per group.
Exceeding a limit fails capture; it never truncates evidence to claim success.

P18 has not run. Any performance conclusion requires the fixed matched
workload, cache initial state, warmup, repeated measurements and original
logs after correctness and P17 admission. There is no measured speedup claim.

## CI and change boundaries

Full repository CI still has separately recorded inherited failures. Those
failures do not grant permission to restore reverted cleanup. **Unrelated CI
cleanup must not be mixed back into ECHO**, including formatting unrelated
files, logger replacement, import cleanup, lint exemptions or C++ renaming.
Current feature regressions must be fixed within their actual scope.
