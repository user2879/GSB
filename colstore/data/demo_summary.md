# Columnar Store Demo Verification Summary

Verdict: **PASS**

## Checks
- [PASS] canonical_query_id_stable_under_whitespace_and_parentheses: {"query_id": "97faee4fdec7880c"}
- [PASS] column_pruning_excludes_unreferenced_columns: {"pruned": ["sku_id", "warehouse_id", "qty"]}
- [PASS] scheduler_never_exceeds_memory_budget: {"peak_memory_bytes": 148, "budget_bytes": 65536}
- [PASS] all_tasks_record_lifecycle_timestamps: {"task_count": 9}
- [PASS] budget_version_monotonic_increase: {"budget_version": 2}
- [PASS] previous_generation_marked_stale: {"result_id": "59c3aa623d3d4993"}
- [PASS] replacement_lists_affected_blocks: {"affected": 9}
- [PASS] byte_identical_results_after_reschedule: {"first_hash": "767ad1c547e6beacaf4832a310d18eab4b0b6b7c6c3bdbacd6584c650c7cc95a", "second_hash": "767ad1c547e6beacaf4832a310d18eab4b0b6b7c6c3bdbacd6584c650c7cc95a"}
- [PASS] stale_generation_blocked_from_summary: {"current_result_id": "507f52606c51473a"}
- [PASS] snapshot_matches_disk_when_unchanged: {"diffs": []}
- [PASS] snapshot_detects_encoding_version_drift: {"diff_kinds": ["segment_field_changed", "segment_field_changed", "encoding_version_changed"]}
- [PASS] superseded_snapshot_frozen_read_only_with_diff_list: {"diff_count": 3}
- [PASS] restored_result_hash_matches_original_snapshot: {"restored_hash": "767ad1c547e6beacaf4832a310d18eab4b0b6b7c6c3bdbacd6584c650c7cc95a"}

## Scenarios
### first_scan
```json
{
  "result_id": "59c3aa623d3d4993",
  "rows_matched": 8,
  "result_hash": "767ad1c547e6beacaf4832a310d18eab4b0b6b7c6c3bdbacd6584c650c7cc95a",
  "task_timeline": [
    {
      "task_id": 0,
      "block": {
        "segment": "sales",
        "row_group": 0,
        "column": "order_id"
      },
      "encoding": "raw",
      "memory_bytes": 48,
      "rows": 12,
      "matched_rows": 12,
      "enqueue_time_ms": 10766855.578848,
      "wait_start_ms": 10766855.578848,
      "start_ms": 10766855.598632999,
      "end_ms": 10766856.670563,
      "status": "done",
      "budget_version": 1,
      "thread_name": "scan-worker-0",
      "wait_ms": 0.019784998148679733
    },
    {
      "task_id": 1,
      "block": {
        "segment": "sales",
        "row_group": 0,
        "column": "region_id"
      },
      "encoding": "dictionary",
      "memory_bytes": 60,
      "rows": 12,
      "matched_rows": 12,
      "enqueue_time_ms": 10766855.578848,
      "wait_start_ms": 10766855.578848,
      "start_ms": 10766855.971131,
      "end_ms": 10766856.919907,
      "status": "done",
      "budget_version": 1,
      "thread_name": "scan-worker-1",
      "wait_ms": 0.39228300005197525
    },
    {
      "task_id": 2,
      "block": {
        "segment": "sales",
        "row_group": 0,
        "column": "amount"
      },
      "encoding": "dictionary",
      "memory_bytes": 108,
      "rows": 12,
      "matched_rows": 12,
      "enqueue_time_ms": 10766855.578848,
      "wait_start_ms": 10766855.578848,
      "start_ms": 10766857.063553,
      "end_ms": 10766857.984137999,
      "status": "done",
      "budget_version": 1,
      "thread_name": "scan-worker-2",
      "wait_ms": 1.4847049992531538
    },
    {
      "task_id": 3,
      "block": {
        "segment": "sales",
        "row_group": 1,
        "column": "order_id"
      },
      "encoding": "raw",
      "memory_bytes": 40,
      "rows": 10,
      "matched_rows": 10,
      "enqueue_time_ms": 10766855.578848,
      "wait_start_ms": 10766855.578848,
      "start_ms": 10766857.483109001,
      "end_ms": 10766858.17124,
      "status": "done",
      "budget_version": 1,
      "thread_name": "scan-worker-3",
      "wait_ms": 1.9042610004544258
    },
    {
      "task_id": 4,
      "block": {
        "segment": "sales",
        "row_group": 1,
        "column": "region_id"
      },
      "encoding": "rle",
      "memory_bytes": 40,
      "rows": 10,
      "matched_rows": 10,
      "enqueue_time_ms": 10766855.578848,
      "wait_start_ms": 10766855.578848,
      "start_ms": 10766858.313736001,
      "end_ms": 10766859.192934,
      "status": "done",
      "budget_version": 1,
      "thread_name": "scan-worker-4",
      "wait_ms": 2.7348880004137754
    },
    {
      "task_id": 5,
      "block": {
        "segment": "sales",
        "row_group": 1,
        "column": "amount"
      },
      "encoding": "dictionary",
      "memory_bytes": 90,
      "rows": 10,
      "matched_rows": 10,
      "enqueue_time_ms": 10766855.578848,
      "wait_start_ms": 10766855.578848,
      "start_ms": 10766858.732757,
      "end_ms": 10766859.412968,
      "status": "done",
      "budget_version": 1,
      "thread_name": "scan-worker-5",
      "wait_ms": 3.1539089996367693
    },
    {
      "task_id": 6,
      "block": {
        "segment": "sales",
        "row_group": 2,
        "column": "order_id"
      },
      "encoding": "raw",
      "memory_bytes": 56,
      "rows": 14,
      "matched_rows": 14,
      "enqueue_time_ms": 10766855.578848,
      "wait_start_ms": 10766855.578848,
      "start_ms": 10766859.510016,
      "end_ms": 10766860.419789,
      "status": "done",
      "budget_version": 1,
      "thread_name": "scan-worker-6",
      "wait_ms": 3.9311679992824793
    },
    {
      "task_id": 7,
      "block": {
        "segment": "sales",
        "row_group": 2,
        "column": "region_id"
      },
      "encoding": "dictionary",
      "memory_bytes": 70,
      "rows": 14,
      "matched_rows": 
```
### budget_change
```json
{
  "replacement": {
    "replaced_result_id": "4bc9ff8e22d3405e",
    "budget_version": 2,
    "affected_blocks": [
      {
        "segment": "sales",
        "row_group": 0,
        "column": "order_id"
      },
      {
        "segment": "sales",
        "row_group": 0,
        "column": "region_id"
      },
      {
        "segment": "sales",
        "row_group": 0,
        "column": "amount"
      },
      {
        "segment": "sales",
        "row_group": 1,
        "column": "order_id"
      },
      {
        "segment": "sales",
        "row_group": 1,
        "column": "region_id"
      },
      {
        "segment": "sales",
        "row_group": 1,
        "column": "amount"
      },
      {
        "segment": "sales",
        "row_group": 2,
        "column": "order_id"
      },
      {
        "segment": "sales",
        "row_group": 2,
        "column": "region_id"
      },
      {
        "segment": "sales",
        "row_group": 2,
        "column": "amount"
      }
    ],
    "abandoned_tasks": 0,
    "reason": "budget parameters changed"
  },
  "new_result_id": "507f52606c51473a",
  "rows_matched": 8
}
```
### snapshot_restore
```json
{
  "snapshot_id": "demo-session-1",
  "diff_count": 3,
  "diffs": [
    {
      "kind": "segment_field_changed",
      "segment": "inventory",
      "field": "file_hash",
      "snapshot": "b720892d96eb2e0d68d74903d4d532f0253bca8ccf67700e8fb1dda3400a67ca",
      "current": "c76166d1261f11ec448f739b72dbcf6b911aa9429e296aea2e80b52b3ebbe696"
    },
    {
      "kind": "segment_field_changed",
      "segment": "sales",
      "field": "file_hash",
      "snapshot": "07b5167943192c31e0ad6b365742c2ec518cfc91aca862fba83d393de1c2f71d",
      "current": "bc8935dadc04d134408d4039e86696aab5c15dead23822900683c6187162b841"
    },
    {
      "kind": "encoding_version_changed",
      "segment": "*",
      "snapshot": 1,
      "current": 2
    }
  ]
}
```
