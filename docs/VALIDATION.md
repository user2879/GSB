# Validation record

Run date: 2026-11-22 (Asia/Shanghai). No Docker, network service, external account, or GPU was used by the application.

## 1. Environment

```text
Python 3.11.0rc1
SQLite 3.37.2
numpy 1.26.4
Tkinter: unavailable in this container (`No module named 'tkinter'`)
```

`python scripts/check_windows_env.py` output:

```json
{
  "numpy_ok": true,
  "numpy_version": "1.26.4",
  "python_ok": true,
  "python_version": "3.11.0rc1",
  "sqlite_ok": true,
  "sqlite_version": "3.37.2",
  "status": "OK",
  "tkinter_error": "No module named 'tkinter'",
  "tkinter_ok": false,
  "tkinter_version": null,
  "working_directory_writable": true
}
```

The delivered target is Windows with standard CPython Tcl/Tk. This build host has no Tk module, so the Tk window itself could not be launched here. The UI was syntax-compiled and the CLI environment boundary was verified:

```text
$ python -m mlpipe_workbench ui
E_ENV
location=environment.tkinter
exit=6
```

On the Windows target, run `check_env.bat` or:

```powershell
python scripts/check_windows_env.py --require-tk --open-window
```

That command opens a real Tk window for two seconds. `run_workbench.bat` then starts the diagnostic UI.

## 2. Workspace validation

```text
$ python -m mlpipe_workbench validate
status=OK
row_count=137
column_count=6
dataset_digest=94518dc0d38f96e9372cdc1d4422261cb83db53d3df9319dde6a20172dd0eb36
pipeline_digest=b88b2f6ddddb96d3760e3070bc4aea0bec6d19bbf559ab6122499fa44fcd65a6
fingerprint=0d77e2c666c544c2726e1908d19b7e032e491ca64edab28f4ae10e691ba84a66
```

## 3. Automated tests

```text
$ python -m unittest discover -s tests -v
...
Ran 33 tests in 10.190s

OK
```

Key tests:

| Test | Verified behavior |
|---|---|
| `test_stage_array_order_does_not_change_fingerprint` | JSON stage ordering and dict order do not affect fingerprint |
| `test_upstream_reference_changes_change_fingerprint` | Upstream-only chain rewrite changes fingerprint |
| `test_concurrent_loads_produce_identical_fingerprint` | 16 concurrent loads yield one fingerprint |
| `test_batch_boundaries_do_not_change_stage_or_final_digests` | Batch sizes 1/7/32/137 yield identical per-stage and final digests |
| `test_fixed_point_truncation_and_saturation` | Decimal truncation toward zero and signed 64-bit saturation |
| `test_remainder_shards_keep_or_drop_explicitly` | Keep/drop partial shard semantics |
| `test_sample_and_stage_version_evolution_replaces_view_and_reuses_shards` | 1 polluted shard, 4 byte-stable shards, spec v1 -> v2, whole view replacement |
| `test_sharding_only_change_preserves_upstream_outputs` | Prefix stage outputs remain identical; only `pack` affected |
| `test_plain_invalid_spec_rejects_wholesale_without_spec_or_run` | `E_SPEC` writes no spec/run rows |
| `test_three_anomaly_injections_have_distinct_codes_and_reports` | `E_DATA` / `E_PARAM` / `E_OUTLIER` are distinct and logged |
| `test_forced_termination_recovers_only_committed_stages_then_replays` | Committed prefix restored; unknown stages rerun; replay OK |
| `test_crash_after_final_stage_still_completes_ledger_transaction` | Termination after final stage still recovers completion |
| `test_tampered_stage_observation_raises_e_replay_with_first_field` | First divergent stage/field is `fill_missing/output_digest` |
| `test_tampered_shard_fingerprint_raises_e_replay` | First divergent field is `pack/shards[1].shard_fingerprint` |
| `test_session_normalize_scale_is_independent_and_replayable_integer_context` | Session scale `2` vs `4` changes integer outputs; run records/replays scale `4` |

## 4. Three-scenario demo

```text
$ python -m mlpipe_workbench demo

SCENARIO 1 COLD_START
status=COMPLETE
run_id=demo-cold
spec_version=1
fingerprint=0d77e2c666c544c2726e1908d19b7e032e491ca64edab28f4ae10e691ba84a66
rows=137
stages=canonicalize:COMMITTED,fill_missing:COMMITTED,normalize_amount:COMMITTED,clip_amount:COMMITTED,pack:COMMITTED
shards=5
shard_ranges=0-32,32-64,64-96,96-128,128-137
final_data_digest=132d9d7b96c96c6c5c9682bcdfd3a8d9ffc212cf274a6a7c6f45d34d2178a47c
cold_start_elapsed_ms=231
END SCENARIO 1

SCENARIO 2 SPEC_EVOLUTION
status=COMPLETE
old_spec_version=1
new_spec_version=2
old_fingerprint=0d77e2c666c544c2726e1908d19b7e032e491ca64edab28f4ae10e691ba84a66
new_fingerprint=b8d8e468883f1ae455f7d7986d37aed22969e839660e82b635bd2434c5febf12
dataset_changed=true
config_changed_stages=fill_missing
affected_stages=canonicalize,fill_missing,normalize_amount,clip_amount,pack
old_shards=5
new_shards=5
invalidated_shards=1
preserved_shards=4
preserved_shard_indices=0,2,3,4
unaffected_outputs_stable=true
view_replaced=true current_fingerprint=b8d8e468883f1ae455f7d7986d37aed22969e839660e82b635bd2434c5febf12
END SCENARIO 2

SCENARIO 3 CRASH_RECOVERY
crash_exit_code=75
status_before_recovery=RUNNING
exit_report_available=false
recovered_stages=canonicalize,fill_missing
unknown_stages=normalize_amount,clip_amount,pack
rerun_plan=normalize_amount,clip_amount,pack
replay_status=OK
stages_verified=5
shards_verified=5
final_data_digest=132d9d7b96c96c6c5c9682bcdfd3a8d9ffc212cf274a6a7c6f45d34d2178a47c
matches_cold_final=true
status_after_recovery=RECOVERED_COMPLETE
END SCENARIO 3
```

`cold_start_elapsed_ms=231` is the measured baseline on this host. The same line printed by `demo` on Windows is the baseline for Windows comparisons; it is intentionally the only timing value in the summary.

## 5. Ledger state after scenarios 1 and 2

```text
$ python -m mlpipe_workbench ledger --db demo_output/runtime/demo.db --run-id demo-evolved

counts = {
  "business_errors": 0,
  "conclusion_history": 2,
  "events": 14,
  "runs": 2,
  "shards": 10,
  "specs": 2,
  "stage_batches": 50,
  "stage_runs": 10
}
runs = [("demo-evolved", "COMPLETE", 2), ("demo-cold", "COMPLETE", 1)]
stages = [
  ("canonicalize", "COMMITTED", 137, "c327cf0854bc"),
  ("fill_missing", "COMMITTED", 137, "737522b84371"),
  ("normalize_amount", "COMMITTED", 137, "1d19698fdb62"),
  ("clip_amount", "COMMITTED", 137, "4223a0ec9ff4"),
  ("pack", "COMMITTED", 137, "4223a0ec9ff4")
]
```

The singleton conclusion points at `demo-evolved`; two history rows prove replacement rather than merge. Ten shard rows are five for each specification; only the current five are exposed in the current view.

## 6. Anomaly injection results

Each command exited with its dedicated code and produced a matching JSONL structured log.

| Injection | Exit | Code | Stage | Param / value | Affected rows | First location |
|---|---:|---|---|---|---:|---|
| `--inject-data-corrupt` | 2 | `E_DATA` | `<dataset>` | `nulls` / null | 83 | `data.sources[0].columns[0].nulls` |
| `--inject-param-bounds` | 2 | `E_PARAM` | `clip_amount` | `max_outlier_permille` / 1001 | - | `pipeline.stages[id=clip_amount].params.max_outlier_permille` |
| `--inject-outlier-ratio 300` | 3 | `E_OUTLIER` | `clip_amount` | `max_outlier_permille` / 100 | 22 | same stage parameter path |

The outlier run additionally reported `outlier_ratio_permille=160`, `non_null_rows=137`, `mean_fixed=160`, and `stddev_fixed=367`. The three log files each contained exactly their own `"code"` value; no code aliasing was observed.

## 7. Replay divergence proof

A completed run's ledger was altered only in `stage_runs.output_digest` for `fill_missing`, then replay was invoked:

```json
{
  "code": "E_REPLAY",
  "field": "output_digest",
  "first_field": "output_digest",
  "first_stage_id": "fill_missing",
  "stage_id": "fill_missing"
}
ledger_error_codes = ["E_REPLAY"]
```

A separate test alters `shards[1].shard_fingerprint`; replay reports `pack` / `shards[1].shard_fingerprint`. This confirms the comparison order and first-divergence requirement at both stage and shard granularity.

## 8. Reproduce

```powershell
python -m unittest discover -s tests -v
python -m mlpipe_workbench demo --out-dir demo_output
python -m mlpipe_workbench ledger --db demo_output/runtime/demo.db --run-id demo-evolved
```

The demo is destructive only inside the selected `--out-dir`; it recreates its runtime ledgers, evolved inputs, logs, and summaries so repeated runs have the same ordered conclusions.

## 9. Requirement-to-evidence map

| Business requirement | Implementation evidence | Runtime/test evidence |
|---|---|---|
| 1. Strong load and `E_SPEC` first location | `loader.py` validates both declarations before version/run allocation | 9 validation tests; invalid spec writes 0 `specs`/`runs` |
| 2. Deterministic fingerprint and monotonic revision | `canonical.py` WBP-FP-1; `Ledger.allocate_spec` under `BEGIN IMMEDIATE` | order-independent, upstream-sensitive, 16-way concurrent loads, v1 -> v2 tests |
| 3. Deterministic fixed-point streaming | `compute.py`; integer statistics; batch callbacks | batch sizes 1/7/32/137 produce identical stage/final digests |
| 4. Remainder sharding and stage binding | keep/drop partial rules; shard fingerprint; stale classifier | 5 shards `0-32,32-64,64-96,96-128,128-137`; staleness tests |
| 5. Invalidation propagation and view replacement | `evolution.py`; singleton delete+insert in `complete_run` | spec v1 -> v2, fill version 1 -> 2, one polluted shard, four preserved; sharding-only prefix stability |
| 6. Three anomaly injections and codes | CLI injection JSON writers; `business_errors` + JSONL logger | `E_DATA`, `E_PARAM`, `E_OUTLIER` table above with distinct exit codes/logs |
| 7. Ledger, forced termination, unknown state | batch/cursor transaction; `RecoveryManager.inspect` | exit 75, pre-recovery `RUNNING`, unknown later stages, explicit rerun plan |
| 8. Replay, Tk UI, repeatable demo | `recovery.py`, `ui.py`, `demo.py` | replay OK 5 stages/5 shards; tamper tests report first stage/field; demo order test; UI boundary verified and Windows preflight included |
