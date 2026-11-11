# ECHO confidence verification: evidence and remaining work

This record separates Base correctness, Scorer calibration and Scorer formal acceptance.
The accompanying [evidence index](echo-validation/2026-11-09-summary.json) pins
the actual reports, raw archive hashes and source versions. Original raw tensor
archives remain in the experiment workspace; this Git repository contains an
index, not those archives. Results below are scoped to the recorded eager,
single-card, greedy protocol.

Experimental single-request Qwen3 Domino FULL graph support is described in
[the bounded graph guide](echo-prefix-fullgraph.md). The eager acceptance
results below do not establish graph generation equivalence.

## Verified results on 2026-11-09

| Run | Result | Scope |
| --- | --- | --- |
| Base a5 on production code `a0f8a105` | P14/P15/P16 passed; issues empty | Base full-vocabulary correctness only |
| Base r5 off12 candidate | 12 modes, 83 cases; P14/P15/P16 passed; issues empty | Same-source candidate with observation disabled |
| Base r5 on12 candidate | 12 modes, 83 cases; P14/P15/P16 passed; issues empty | Same-source candidate with observation enabled |
| Base r5 paired replay | Logical counters and paired semantics passed | Finite native five-arm evidence below; overall P17 pending |
| Scorer calibration r5 run2 | Strict probability calibration passed; issues empty | Calibration only |
| Scorer formal r5 run2 | 12 modes, 83 cases; P14/P15/P16 passed; issues empty | Independent Scorer-only correctness and lifecycle suite |

All three Base acceptance replays checked the original tensors and lifecycle/stop
proofs. Their source, raw data and proof files remained unchanged. The a5 archive
contains 3,212 verified files. Each candidate off12/on12 archive contains 3,398.
No Base result grants correctness to the Scorer confidence source.

The isolated r5 candidate has 4,728 source files and manifest
`69f6200fcdf899a775977d26440abae7bde5741b7aa7e7d24b2253cea747d28b`.
Commit `48071690` delivered its exact nine reviewed runtime/test files on top
of production code `a0f8a105`. This documentation update changes no production
code. The index binds the delivered files to the tested snapshot; overall
P17 and P18 remain ungranted.

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
threshold. At that calibration stage, Scorer P14/P15/P16 were ungranted. The separate
formal result below supersedes that pending status; P17/P18 remain ungranted.

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
Source, raw-data and oracle hashes remained unchanged. This run1 failed
acceptance remains retained and is not converted into a pass.

A new diagnostic loader reuses the verified canonical module and checks both
source copies, the loaded path and function bindings. Nine regression tests
and 108 checks in the actual staged layout passed without skips; independent
review found no open P0/P1/P2 within capture preparation. A separate run2 uses
the same experiment and frozen oracle. Its completed result follows.

## Scorer formal run2 acceptance

The replacement model run completed all 12 modes and 83 generation cases with
driver and supervisor exit 0 in 2,606.9 seconds. Archive and individual member
checks recovered 3,421 original files totaling 7,361,963,004 bytes. The unchanged
156-file frozen oracle then completed its CPU replay with exit 0 in 1,740.2
seconds, without timeout or remaining process-group members.

The strict report independently grants Scorer P14/P15/P16 within this pinned
single-card eager protocol. All 566 acceptance gates passed; issues and changes
to source, raw data and proof files are empty. It excludes Base and grants
neither P17 nor P18. The earlier run1 failure remains preserved.

The actual successful formal suites used Base shortening/conservative thresholds
0.10/0.01 and separate Scorer thresholds 0.50/0.05. These source-specific values
must not be interchanged or presented as universal production recommendations.

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
python -X utf8 -B run_scorer_formal_r5_run2_comparison_v1.py status
```

Full reports are identified by workspace-relative paths, sizes and SHA-256
hashes in the evidence index. The comparators were frozen before consuming
the corresponding new model results. A successful model process alone was
not used as an acceptance result. Download failures and original partial
files were preserved; recovered archives passed exact SHA and member checks.

## Host budget accounting delivered

The new internal DEBUG-gated statistics distinguish produced S, published A,
verified V and raw sampler acceptance. Consumption captures the current request
cohort before the next proposal can replace it. Unconsumed work, unmatched
prefill outputs and discarded publications remain separate. These accepted-token
counts exclude the root/bonus and precede Scheduler/frontend terminal clipping;
they are not counts of tokens delivered through the API.

Statistics use existing host scalars and already delivered host token IDs. The
async output wrapper delegates the original get_output once per call; it adds no
device value read, synchronization or copy for metrics. Diagnostic errors mark
counts unknown and cannot replace native inference results or exceptions. No
public parameter, metrics endpoint or default policy is added.

The exact source passed 225 CPU tests with nine NPU-only skips. Its separate
component checks passed 54 CPU and 58 native checks without skips. Delivery
review independently passed another 37 tests and found no open P0/P1/P2.

## Remaining P17 and P18 gates

Base off12/on12 passed the same correctness comparator across 83 cases. The
five canonical34/128 native off/on pairs are now collected and replayed. Recorded
compute, copy parameters, synchronization calls and per-stream work agree after
classifying only the fixed zero-duration StreamSyncTaskFinish diagnostic point.
Ten canonical outputs match completely. The shortening arm has three consumed
shrinks and matched S/A/V/accepted totals of 432/408/408/123; the other four arms
have 432/432/432/123.

The original default strict comparison remains failed at 24 versus 23 diagnostic
completion points; strict equality also fails for explicit-fixed and tau0.
Every such point retains its unique synchronization parent. Fixed official
runtime sources and the separate reviewed classifier support the limited
diagnostic classification; the exact internal skipped branch was not observed.
Original failures are not rewritten. Neither global trace completeness nor
universal buffer ownership is claimed. This reviewed finite Base evidence still
requires the exact P18 provider and plan to be reviewed before P18 admission.

Scorer has its own formal correctness/lifecycle pass. Its separate five-arm
native off/on suite now completed all ten captures with thresholds 0.50/0.05.
The finite replay and independent review passed. Shortening consumed 11
proposals, including nine shrinks; matched S/A/V/accepted totals were
528/322/322/125. The other four arms consumed nine proposals each, without
shrinks, and had 432/432/432/123. All ten canonical outputs match.

Scorer explicit-fixed retains its original strict failure at 23 versus 24
diagnostic completion points. Its limited conditional comparison passed;
the other four strict comparisons passed. This does not establish global
trace completeness or universal buffer ownership. Scorer subsequently completed
its own full83 metrics-ON capture and frozen correctness/lifecycle replay.
The CPU command exited 0, P14/P15/P16 passed and issues were empty; source,
raw and proof snapshots did not change. These results use Scorer's own
thresholds 0.50/0.05 and do not borrow Base acceptance.

Its own OFF/ON logical pair passed all 12 modes and 83 cases, including
6,975 paired logical records. Independent full replay produced a byte-identical
report, and 30 targeted controls passed. Each side retained its own P16 report,
CPU terminal and original receipt binding. The 811 production inventory records
comprise 809 Python entities and two installed-native-library link records.
The native links were checked against stage/preflight target and hash evidence;
the collected raw data has no native .so entities to rehash locally.

This closes the reviewed Scorer logical metrics counterpart within the pinned
eager protocol. The five-arm native evidence is a separate finite result.
Neither result supplies a performance measurement, global trace completeness,
universal ownership or overall P17/P18 completion. Scorer still requires its
own P18 diagnostics and formal collection. The evidence index binds the exact
reports, independent review and original collection metadata.

P18 has no formal performance result. All six Base diagnostic captures now
completed with clean exit 0 and were recovered. The independently reviewed
V7 reader and V8 analyzer checked every original raw member, archive,
configuration, loaded-source binding, asset inventory, cache state and
same-source canonical output. The aggregate repeated all six full analyses
and required each entire report to equal its fresh original-data replay.

Independent result review checked all 17,760 raw files (2,727,568,702 bytes),
278,463,167 archive bytes, 211,787 recorded TASK rows and 16,288 paired
compact copy records. Recorded task denominators and raw copy count/byte
vectors close within each finite capture. One unattributed zero-duration
SQE and 38 timestamp diagnostics remain explicitly recorded. Unknown
ownership and work semantics are retained; zero work is not inferred.

The original AR analysis failure and failed dispatcher status remain intact.
In that AR capture, all 71,860 TASK rows lie inside the device's own
profiling-enable/disable window; ten intervals end after the CPU root marker
while keeping their original other-root submission attribution. Both source
synchronizations are inside the CPU marker; an additional outer pair is
separate. These observations do not prove CPU/device clock alignment,
physical drain ordering, global trace completeness or universal ownership.

Fixed/tau0 captures use the original non-structured-output placeholder
handler branch, which has no instrumented copy-event wait marker. Positive
confidence delivery uses its separately verified marker contract. All native
synchronization records remain counted; absent markers do not establish
absence of synchronization. The source-derived branch and long-path member
checks were independently reviewed before the six complete raw analyses.

This is finite Base diagnostic readiness only. Scorer requires its own
diagnostics. The Base formal collection preparation has since passed its
separate recovery review. Completing that collection and its result review
still requires matching outputs, fresh process/cache state, public warmup and
five repetitions per arm. No completed formal performance comparison or
speedup is claimed.

## Exploratory Scorer budget scans and profiles

The user-requested single-run Scorer budget-48 and budget-64 scans, and two
selected budget-64 profiles, are recorded in the
[Scorer experiment notes](echo-scorer-budget-experiments.md). The dynamic
group selected for profiling is threshold 0.10, the highest observed
unprofiled confidence throughput at budget 64; fixed is a separate control.
The two complete diagnostic captures are not a replacement for the formal
matrix or repeated-performance requirements above and do not close overall
P17/P18.

The later budget-64 scan changes only frontier width to 4; depth remains
16. The four initial groups were extended by two independent 0.40/0.50
measurements, without repeating the earlier groups. Its single-run table
and the historical width-8 comparison are in the same experiment notes.
This adds no new-width
P16/native acceptance or overall P17/P18 claim.

## CI and change boundaries

Full repository CI still has separately recorded inherited failures. Those
failures do not grant permission to restore reverted cleanup. **Unrelated CI
cleanup must not be mixed back into ECHO**, including formatting unrelated
files, logger replacement, import cleanup, lint exemptions or C++ renaming.
Current feature regressions must be fixed within their actual scope.
