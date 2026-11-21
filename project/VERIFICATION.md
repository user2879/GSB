# Verification Record

- Overall: **PASS**
- Wall time: 63.37 seconds
- Python: `3.11.0rc1 (main, Jun 29 2026, 00:05:30) [GCC 11.4.0]`
- numpy: `1.26.4`
- Actual platform: `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35` (`os.name=posix`)
- Windows-specific behavior verified here: `False`
- The implementation uses only standard-library threads, JSON, struct and `os.replace`; this runner was executed on Linux/WSL2.

## Acceptance Checks

| Check | Result | Evidence |
|---|---|---|
| `sample_generation` | PASS | {"boundary_ticket": "T0047", "evidence_count": 211, "passed": true, "seed": 20261121, "ticket_count": 48} |
| `sample_reproducibility` | PASS | {"boundary_ticket": "T0047", "file_count": 10, "passed": true, "seed": 20261121} |
| `invalid_input_isolation` | PASS | {"codes": ["E_DUPLICATE_EVIDENCE", "E_EVIDENCE_TYPE_UNKNOWN", "E_JSON_INVALID", "E_ROLE_UNKNOWN", "E_TIMESTAMP_INVALID"], "failed_ticket_ids": ["T9001", "T9002", "T9003", "T9004", "T9005"], "passed": true, "valid_survivor_ticket": "T9006"} |
| `initial_full_run` | PASS | {"boundary_tickets": ["T0047"], "concurrency_conclusion": "PASS", "diagnostic_exists": true, "passed": true, "summary": {"boundary_ticket_ids": ["T0047"], "completed_ticket_count": 48, "current_upgrade_count": 5, "failure_count": 0, "int... |
| `version_interop` | PASS | {"passed": true, "record_count": 48, "v1_roundtrip_exact": true, "v1_sidecar_records": 48, "v2_roundtrip_exact": true, "v3_header_features": ["ticket_id", "upgrade_code", "priority_level", "priority_name", "precision", "reason_codes", "m... |
| `concurrency_and_faults` | PASS | {"fault_failure": {"code": "E_STAGE_EXCEPTION", "evidence_seq": null, "exception_type": "SyntheticRelationFailure", "line_no": null, "message": "acceptance relation failure", "stage": "relations", "ticket_id": "T0048"}, "fault_isolated_t... |
| `incremental_equivalence` | PASS | {"all_sequence_steps_equal": true, "fusion_scale_invariant": true, "passed": true, "persisted_inputs_full_rebuild_equal": true, "step_count": 11, "steps": [{"affected_count": 23, "artifact_hashes_equal": true, "incremental_equals_full": ... |
| `snapshot_resume` | PASS | {"passed": true, "recovery": {"available": true, "completed_ticket_count": 48, "config_hash": "5aba0d3ebb3447c7768d3ee30655824eb3828cefefd6ca474c5ed094016093b2", "generation": "gen-00000091-331a79d620cd", "global_version": 91, "partial":... |
| `interaction_budget` | PASS | {"background_finished": true, "code": "E_INTERACT_BUDGET_PENDING", "elapsed_ms": 5.05, "final_equals_full": true, "passed": true, "summary_retained": true} |
| `snapshot_stale` | PASS | {"cache_resume_disabled": true, "code": "E_SNAPSHOT_STALE", "passed": true, "stale_reasons": ["config_hash"]} |
| `graceful_signal_exit` | PASS | {"passed": true, "return_code": 0, "signals": [15], "status": "stopped", "summary": {"boundary_ticket_ids": [], "completed_ticket_count": 4, "current_upgrade_count": 1, "failure_count": 0, "interrupted": true, "snapshot_version": 1}} |
| `unit_and_integration_tests` | PASS | {"command": "python -m unittest discover -v", "passed": true, "return_code": 0, "summary_lines": ["----------------------------------------------------------------------", "Ran 29 tests in 37.955s", "", "OK"]} |

## Reproduce

```bash
cd project
PYTHONPATH=. python3 -m arbitration_workbench generate --root .
PYTHONPATH=. python3 -m arbitration_workbench run --root . --fresh
PYTHONPATH=. python3 scripts/run_acceptance.py
PYTHONPATH=. python3 -m unittest discover -v
```

The main-sample run report is `outputs/diagnostic_report.json`; the acceptance-run report is `outputs/acceptance_diagnostic_report.json`; the ledger is `outputs/acceptance_results.json`.
