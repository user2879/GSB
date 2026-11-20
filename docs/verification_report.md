# Verification Report

Generated: 2026-11-20T21:35:06Z

## Environment

- OS: Linux 6.18.33.2-microsoft-standard-WSL2
- Python: 3.11.0rc1
- NumPy: 1.26.4
- Container: none (local process only)
- Fixture seed: 20261121

The implementation uses only portable Python APIs for paths, file replacement,
and thread pools.  It is compatible with Python 3.10+ on Windows and was run in
the available local environment above.

## Acceptance Summary

**11/11 checks passed.**

| Check | Result | Detail |
|---|---|---|
| deterministic_fixtures | PASS | 64 samples, 19 commands, fixed seed 20261121 |
| all_three_schemes_distinct | PASS | three distinct position-vector fingerprints and three distinct score fingerprints |
| incremental_equals_full | PASS | all substantive diagnostic fields match blank full rebuilds |
| incremental_cache_audit | PASS | context change reuses all encoding stages; sample replacement recomputes only s007 |
| large_budget_completes | PASS | 19 responses, p95 426.563 ms |
| small_budget_pending_semantics | PASS | 19 pending responses with retained queue |
| pending_queue_advances | PASS | queue [72, 36, 0] |
| snapshot_resume_and_stale | PASS | atomic version load, exact reuse, stale version rejected |
| fault_isolation | PASS | one injected trace failure, five unrelated samples completed |
| concurrency_cap | PASS | observed 4 <= configured 4 |
| report_sections | PASS | diagnostic JSON includes all requested report sections and a ranked sensitivity list |

## Scheme Extrapolation Statistics

| Scheme | Over-tolerance samples | Mean max deviation | Max deviation | Position fingerprint prefix |
|---|---:|---:|---:|---|
| standard_rope | 38 | 5.835631 | 267.514337 | `5544213f359f8261` |
| linear_interpolation | 38 | 3.322718 | 44.399610 | `ed52fab40c64c9c9` |
| ntk_aware | 38 | 3.777277 | 96.802914 | `042efd1a23b1c143` |

## Incremental vs Full Rebuild

- Final incremental report equals a blank full rebuild on all substantive fields: `True`.
- Large-budget final report equals the full rebuild: `True`.
- Small-budget final report equals the full rebuild: `True`.
- The execution-only audit is intentionally excluded from equality; the warm run has cache hits while a blank rebuild cannot. Stage cache keys and all diagnostic fields are compared.

## Latency Budgets

- Large budget: 3000 ms; responses within budget 19/19; pending responses 0; p95 elapsed 426.563 ms.
- Small budget: 20 ms; responses within budget 20/20; pending responses 19; p95 elapsed 16.421 ms.
- Dedicated queue probe: completed samples [0, 0, 24], remaining stage tasks [72, 36, 0]; final business code `['OK']`.

The small-budget command sequence returns `E_INTERACT_BUDGET_PENDING` and retains
the background queue; the final large-budget poll drains it to a full report.

## Snapshot Recovery

- Committed global version: 7
- Loaded snapshot: `True`; entries: 36
- Version mismatch: `E_SNAPSHOT_STALE` (`ok=False`)
- Resumed cache hits: 36; recomputed jobs: 0

## Fault Isolation

- Invalid JSON, non-finite feature, dimension mismatch, duplicate ID, out-of-bounds key and empty file are reported while valid samples continue.
- Loader business codes: `{"E_BASE_INVALID": 1, "E_CONFIG_INVALID": 1, "E_DIM_MISMATCH": 1, "E_SAMPLE_INVALID": 5, "E_SCALE_INVALID": 1}`
- Interactive business codes: `{"E_INTERACT_BUDGET_PENDING": 19}`
- Injected stage failure: completed 5 samples, failed 1; code `E_INJECTED_STAGE` on stage `error_trace`.

## Top Over-tolerance Samples

| Rank | Sample | Relative distance | First exceeded stage | Max deviation |
|---:|---|---:|---|---:|
| 1 | s005 | 40 | position_vector | 267.514337 |
| 2 | s001 | 33 | position_vector | 78.847487 |
| 3 | s046 | 64 | position_vector | 52.483359 |
| 4 | s057 | 33 | position_vector | 35.139859 |
| 5 | s040 | 47 | position_vector | 16.225106 |
| 6 | s053 | 40 | position_vector | 15.671944 |
| 7 | s038 | 126 | position_vector | 15.044595 |
| 8 | s029 | 47 | position_vector | 10.521140 |
| 9 | s010 | 33 | position_vector | 9.267505 |
| 10 | s027 | 33 | position_vector | 8.886498 |

## Parameter Sensitivity Ranking

| Rank | Parameter | Variant | |Delta mean max deviation| |
|---:|---|---|---:|
| 1 | training_context_length | 64 | 3.791166 |
| 2 | training_context_length | 16 | 3.156938 |
| 3 | base | 5000.0 | 2.526511 |
| 4 | scheme | linear_interpolation | 2.512913 |
| 5 | scheme | ntk_aware | 2.058354 |
| 6 | pruning_tolerance | 1.0 | 1.868034 |
| 7 | base | 20000.0 | 1.449269 |
| 8 | pruning_policy | none | 0.000000 |

## Artifacts

- `artifacts/diagnostic_report.json`
- `artifacts/full_rebuild_report.json`
- `artifacts/interactive_large_responses.json`
- `artifacts/interactive_small_responses.json`
- `artifacts/fault_report.json`
- `artifacts/snapshot_verification.json`
- `artifacts/latency_verification.json`
- `artifacts/acceptance_report.json`
- `docs/verification_report.md`
