# Actual Validation Record

验证日期：2026-11-20（Asia/Shanghai）。执行环境为本地 Ubuntu/WSL2 工作区，未使用容器或外部数据。

## Runtime

```text
Python 3.10.12
numpy 2.2.6
PyYAML 6.0.3
```

依赖安装命令：

```text
python3.10 -m pip install --user "numpy>=1.23" "PyYAML>=6.0"
```

## Automated acceptance suite

Command:

```text
python3.10 -m unittest discover -s tests -v
```

Actual result:

```text
test_categorical_mapping_and_missing_group ... ok
test_min_ratio_constraint ... ok
test_numeric_missing_outliers_and_chi_square_tie ... ok
test_engine_returns_config_error_report_without_snapshot ... ok
test_invalid_values_are_not_defaulted ... ok
test_target_values_outside_binary_enum_fail ... ok
test_actual_sigkill_then_restart_recovers_exact_result ... ok
test_data_description_change_is_detected_as_stale ... ok
test_expected_global_version_is_checked_without_changing_model_identity ... ok
test_hard_exit_after_column_cache_recovers_exact_result ... ok
test_hard_exit_after_committed_stage_recovers_exact_result ... ok
test_sigterm_commits_current_work_and_exits_zero ... ok
test_stale_snapshot_is_rejected_then_explicit_revision_reuses_cache ... ok
test_ascending_conflicts_merge_until_satisfied ... ok
test_descending_is_already_monotonic ... ok
test_unresolved_preserves_current_state ... ok
test_repeated_woe_is_identical ... ok
test_zero_frequency_is_smoothed_and_iv_contributions_sum ... ok
test_column_exceptions_are_isolated_and_failures_are_sorted ... ok
test_monotonic_exception_is_isolated ... ok
test_two_fresh_states_produce_byte_identical_reports ... ok
test_unresolved_monotonic_state_is_preserved_and_excluded_from_evaluation ... ok
test_worker_count_does_not_change_results ... ok
test_edit_sequence_matches_full_rebuild_and_invalidates_minimally ... ok
test_single_column_replacement_keeps_target_and_other_columns_stable ... ok
test_shell_command_sequence_and_save ... ok

----------------------------------------------------------------------
Ran 26 tests in 21.783s

OK
```

The suite explicitly compares the following fields after every edit in a five-step edit sequence (direction switch, per-column min-ratio change, priority change, global max-bin change, single-column data replacement) against a fresh full rebuild:

- column bin boundaries, counts and missing flags,
- category-to-group mapping,
- WOE/IV and smoothed-bin flags,
- monotonic correction records,
- evaluation metrics,
- stable failure list.

It also asserts the minimal invalidation sets for those edits and verifies that SIGKILL, hard exit after an atomic column cache write, and hard exit after a committed stage all recover to content identical with an uninterrupted run. Additional recovery checks cover stale data descriptions, expected global-version mismatches, and explicit revision reuse. Two fresh independent states are also checked for byte-identical JSON reports.

## Sample experiment

First execution:

```text
[ok] snapshot=5 recovery_point=0 columns=7 failures=0 auc=0.747715670237 cache=0/22 report=.../first.json
```

Immediate restart with the same config/data/state directory:

```text
[ok] snapshot=5 recovery_point=5 columns=7 failures=0 auc=0.747715670237 cache=22/0 report=.../second.json
```

Recovery metadata from the second report:

```json
{
  "progress_completed_columns": [],
  "recovered": true,
  "refused": false,
  "resume_version": 5,
  "resumed_from_stage": "evaluation",
  "revision_of": null,
  "revision_started": false
}
```

Per-column bin count / IV / monotonic correction count:

```text
age              6  0.335339944955  0
bureau_score     5  0.366632249696  1
channel          6  0.009925347115  0
income           5  0.158741109186  0
merchant_cluster 6  0.217631898525  0
occupation       6  0.038088127424  0
tenure_months    2  0.000179743793  3
```

All five stage versions were present after the run: `load`, `binning`, `woe`, `monotonic`, `evaluation`.

## Notes

- No report can be promised after `SIGKILL`; the recovery test validates that the next process reaches the same final content using only complete atomic artifacts, manifest and progress state.
- The graceful SIGTERM test asserts process exit code 0 and successful subsequent recovery.
- A stale snapshot test confirms changed config without explicit `--revision` returns `E_SNAPSHOT_STALE` and leaves `manifest.json` byte-for-byte unchanged.
