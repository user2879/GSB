# Acceptance Verification

- Status: **passed**
- Python: `3.11.0rc1`
- NumPy: `1.26.4`
- Ledger digest: `9574beb364c2d5aa2f4297a7a0b64c552461f0c15814bfc82c96f0d01b64da4f`
- Worker ports: `[18900, 18901]`

| Check | Result |
| --- | --- |
| `serial_concurrent_ledger_equality` | PASS |
| `incremental_full_ledger_equality` | PASS |
| `three_version_normalized_ledger` | PASS |
| `conversion_roundtrip_restores_all_fields` | PASS |
| `fault_injection_isolation` | PASS |
| `parameter_version_invalidation_boundary` | PASS |
| `normalization_budget_pending` | PASS |
| `previous_pending_priority` | PASS |
| `worker_quota_exceeded` | PASS |
| `unit_tests` | PASS |
| `snapshot_manifest_complete` | PASS |
| `final_continued_run_reuses_cache` | PASS |

## Details

```json
[
  {
    "check": "serial_concurrent_ledger_equality",
    "passed": true,
    "details": {
      "serial_digest": "9574beb364c2d5aa2f4297a7a0b64c552461f0c15814bfc82c96f0d01b64da4f",
      "concurrent_digest": "9574beb364c2d5aa2f4297a7a0b64c552461f0c15814bfc82c96f0d01b64da4f"
    }
  },
  {
    "check": "incremental_full_ledger_equality",
    "passed": true,
    "details": {
      "full_digest": "9574beb364c2d5aa2f4297a7a0b64c552461f0c15814bfc82c96f0d01b64da4f",
      "incremental_digest": "9574beb364c2d5aa2f4297a7a0b64c552461f0c15814bfc82c96f0d01b64da4f",
      "actual_normalization_computed": [],
      "actual_resource_computed": []
    }
  },
  {
    "check": "three_version_normalized_ledger",
    "passed": true,
    "details": {
      "receipt_versions": {
        "receipts/r1_batch_001.json": "r1",
        "receipts/r2_batch_002.json": "r2",
        "receipts/r3_batch_003.json": "r3",
        "receipts/r3_batch_004_overwrite.json": "r3",
        "receipts/r3_batch_005.json": "r3",
        "receipts/r3_batch_005_rank_conflict.json": "r3"
      },
      "counts": {
        "receipts_ok": 14,
        "authoritative_requests": 12,
        "processed_batches": 5,
        "resource_batches": 5,
        "failures": 0,
        "pending_batches": 0,
        "quota_blocked_batches": 0
      },
      "ledger_digest": "9574beb364c2d5aa2f4297a7a0b64c552461f0c15814bfc82c96f0d01b64da4f"
    }
  },
  {
    "check": "conversion_roundtrip_restores_all_fields",
    "passed": true,
    "details": {
      "completeness": "complete",
      "fields_equal": {
        "batch_id": true,
        "duration_ms": true,
        "output_digest": true,
        "overwritten_ref": true,
        "parameter_digest": true,
        "parameter_version": true,
        "request_id": true,
        "stable_rank": true,
        "worker_id": true
      },
      "restored_uid": "conversion/r1_from_r3.json#0",
      "source_uid": "receipts/r3_batch_003.json#0"
    }
  },
  {
    "check": "fault_injection_isolation",
    "passed": true,
    "details": {
      "injection_hits": [
        {
          "injection_id": "INJ-DUPLICATE-BATCH",
          "hit": true,
          "code": "E_DUPLICATE_BATCH_ID",
          "source": "fault_receipts/injected_duplicate_batch.json",
          "ordinal": 1
        },
        {
          "injection_id": "INJ-NONFINITE-DURATION",
          "hit": true,
          "code": "E_DURATION_NONFINITE",
          "source": "fault_receipts/injected_nonfinite_duration.json",
          "ordinal": 0
        },
        {
          "injection_id": "INJ-TRUNCATED",
          "hit": true,
          "code": "E_TRUNCATED_RECEIPT",
          "source": "fault_receipts/injected_truncated.json",
          "ordinal": 1
        },
        {
          "injection_id": "INJ-UNKNOWN-PARAM",
          "hit": true,
          "code": "E_UNKNOWN_PARAM_VERSION",
          "source": "fault_receipts/injected_unknown_parameter.json",
          "ordinal": 0
        }
      ],
      "failure_codes": [
        "E_DUPLICATE_BATCH_ID",
        "E_DURATION_NONFINITE",
        "E_TRUNCATED_RECEIPT",
        "E_UNKNOWN_PARAM_VERSION"
      ]
    }
  },
  {
    "check": "parameter_version_invalidation_boundary",
    "passed": true,
    "details": {
      "before_digest": "9574beb364c2d5aa2f4297a7a0b64c552461f0c15814bfc82c96f0d01b64da4f",
      "after_digest": "8f3381bc0999edc25d28e6806f35436d6404be90aa834aca9e2eecce84cee168",
      "changed_workers": [
        {
          "worker_id": "w-alpha",
          "previous_parameter_version": 3,
          "current_parameter_version": 5
        }
      ],
      "normalization_computed": [
        "batch-001",
        "batch-003",
        "batch-005"
      ],
      "normalization_reused": [
        "batch-002",
        "batch-004"
      ],
      "resource_computed": [
        "batch-001",
        "batch-003",
        "batch-005"
      ],
      "resource_reused": [
        "batch-002",
        "batch-004"
      ]
    }
  },
  {
    "check": "normalization_budget_pending",
    "passed": true,
    "details": {
      "budget": 9,
      "consumed": 9,
      "pending": [
        "batch-004",
        "batch-005"
      ],
      "failure_codes": [
        "E_NORMALIZE_BUDGET_PENDING",
        "E_NORMALIZE_BUDGET_PENDING"
      ]
    }
  },
  {
    "check": "previous_pending_priority",
    "passed": true,
    "details": {
      "budget": 4,
      "tickets": [
        {
          "sequence": 0,
          "batch_id": "batch-004",
          "worker_id": "w-beta",
          "points": 2,
          "priority_pending": true,
          "budget_before": 4,
          "budget_after": 2,
          "worker_used_before": 0,
          "worker_used_after": 2
        },
        {
          "sequence": 1,
          "batch_id": "batch-005",
          "worker_id": "w-alpha",
          "points": 2,
          "priority_pending": true,
          "budget_before": 2,
          "budget_after": 0,
          "worker_used_before": 0,
          "worker_used_after": 2
        }
      ],
      "consumed": 4
    }
  },
  {
    "check": "worker_quota_exceeded",
    "passed": true,
    "details": {
      "quota": 5,
      "blocked": [
        "batch-003"
      ],
      "failure_codes": [
        "E_WORKER_QUOTA_EXCEEDED"
      ]
    }
  },
  {
    "check": "unit_tests",
    "passed": true,
    "details": {
      "command": "python -m unittest discover -s tests -v",
      "returncode": 0,
      "tests": 30,
      "tail": "test_bad_records_are_isolated_and_codes_are_specific (test_receipts.ReceiptParsingTests.test_bad_records_are_isolated_and_codes_are_specific) ... ok\ntest_three_versions_share_one_internal_field_set (test_receipts.ReceiptParsingTests.test_three_versions_share_one_internal_field_set) ... ok\ntest_batch_size_change_invalidates_resources_not_batch_normalization (test_resources_scheduler.ResourceTests.test_batch_size_change_invalidates_resources_not_batch_normalization) ... ok\ntest_fixed_point_macs_utilization_and_quality (test_resources_scheduler.ResourceTests.test_fixed_point_macs_utilization_and_quality) ... ok\ntest_parameter_version_change_invalidates_only_linked_batches (test_resources_scheduler.ResourceTests.test_parameter_version_change_invalidates_only_linked_batches) ... ok\ntest_budget_pending_then_previous_pending_has_priority (test_resources_scheduler.SchedulerTests.test_budget_pending_then_previous_pending_has_priority) ... ok\ntest_worker_quota_blocks_batch_without_deducting_budget (test_resources_scheduler.SchedulerTests.test_worker_quota_blocks_batch_without_deducting_budget) ... ok\n\n----------------------------------------------------------------------\nRan 30 tests in 3.880s\n\nOK"
    }
  },
  {
    "check": "snapshot_manifest_complete",
    "passed": true,
    "details": {
      "global_version": 3,
      "config_hash": "d4a3b3f32519ac6670b1742dbcec8ff1ccb55fd1819f2b7300ded98146ffa420",
      "receipt_fingerprint_count": 6,
      "completed_workers": [
        "w-alpha",
        "w-beta"
      ],
      "consumed_budget": 13
    }
  },
  {
    "check": "final_continued_run_reuses_cache",
    "passed": true,
    "details": {
      "ledger_digest": "9574beb364c2d5aa2f4297a7a0b64c552461f0c15814bfc82c96f0d01b64da4f",
      "actual_invalidation": {
        "cold_cache": false,
        "changed_workers": [],
        "ownership_global_invalidated": false,
        "resource_global_invalidated": false,
        "normalization_invalidated_batches": [],
        "resource_invalidated_batches": [],
        "reused_batches": [
          "batch-001",
          "batch-002",
          "batch-003",
          "batch-004",
          "batch-005"
        ],
        "reasons_by_batch": {},
        "normalization_computed_batches": [],
        "normalization_reused_batches": [
          "batch-001",
          "batch-002",
          "batch-003",
          "batch-004",
          "batch-005"
        ],
        "resource_computed_batches": [],
        "resource_reused_batches": [
          "batch-001",
          "batch-002",
          "batch-003",
          "batch-004",
          "batch-005"
        ],
        "normalization_task_count": 0,
        "resource_task_count": 0
      }
    }
  }
]
```
