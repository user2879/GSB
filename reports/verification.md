# Verification Report

- Sample: `sample`
- Seed: `20261122`
- Manifest version: `obs-2026-11-22-v1`
- Checks: **17/17 passed**
- Unit tests: `python -m unittest discover -v` — 19 passed

| Check | Result | Evidence |
| --- | --- | --- |
| `SAMPLE-DETERMINISM` | PASS | {"digest": "58a8a9a75334d64eefd4acfe73dbec53864e04925b6644f481f9360b31df5883", "second_digest": "58a8a9a75334d64eefd4acfe73dbec53864e04925b6644f481f9360b31df5883"} |
| `BASE-SUCCESS` | PASS | {"failed": 2, "invalid_observation_failures": 1, "keypoints_total": 16, "observation_final_total": 86, "observation_valid_total": 90, "pending": 0, "removed_observations_in_successful_points": 1, "residual_budget_consumed": 270, "residual_budget_remaining":... |
| `TRIANG-DETERMINISTIC` | PASS | {"max_residual_px": 0.11821366581410404, "point": [-0.00025650707032634317, 0.6799789309453204, 0.6198715971245506]} |
| `OUTLIER-REJECTION` | PASS | {"last_round_residual_px": 0.12261965125850514, "removed_views": [4], "rounds_executed": 1} |
| `DISPARITY-SCREEN` | PASS | {"error_codes": ["E_DISPARITY_INSUFFICIENT"], "excluded_pairs": [{"angle_deg": 9.688786851046036, "excluded": true, "views": [1, 2]}, {"angle_deg": 14.675286620441362, "excluded": true, "views": [1, 3]}, {"angle_deg": 4.986499769500694, "excluded": true, "v... |
| `INLIER-SHORTAGE` | PASS | {"error_codes": ["E_INLIER_SHORTAGE"], "trace": [{"decision": "inlier_shortage", "min_inliers": 3, "remaining_after_removal": 2, "round": 1, "state_after": null, "state_before": {"max_residual_px": 21.769725902176525, "max_residual_view": 4, "point": [-0.09... |
| `OBS-INVALID-CONTINUE` | PASS | {"error_codes": ["E_OBS_INVALID"], "final_observations": 6, "status": "ok"} |
| `INSTANCE-AMBIGUITY` | PASS | {"A": [-0.7900189491148175, -0.34017661522726095, -0.14002948189749276], "B": [0.1498290198541685, -0.1500326564783435, -0.10003616843552221]} |
| `SNAPSHOT-RECOVERY` | PASS | {"recomputed_points": [], "recovery_decision": "exact_snapshot_available", "reused_points": 16, "snapshot_decision": "unchanged"} |
| `INCREMENTAL-EQUIVALENCE` | PASS | {"core_equal": true, "fresh_snapshot": "fda68c4096bdfbf8f9ce487fef1000f32ce9de52bf5338ae264929e5c9c4a06c", "incremental_snapshot": "fda68c4096bdfbf8f9ce487fef1000f32ce9de52bf5338ae264929e5c9c4a06c", "recomputed_points": ["aux/marker@instance-A"], "reused_co... |
| `CONFIG-STAGE-INVALIDATION` | PASS | {"core_equal": true, "stage_recomputed": {"finalization": 14, "load": 0, "rejection": 15, "triangulation": 0}, "stage_reused": {"finalization": 0, "load": 16, "rejection": 0, "triangulation": 16}} |
| `CONCURRENCY-INVARIANCE` | PASS | {"core_equal": true, "parallel_summary": {"failed": 2, "invalid_observation_failures": 1, "keypoints_total": 16, "observation_final_total": 86, "observation_valid_total": 90, "pending": 0, "removed_observations_in_successful_points": 1, "residual_budget_con... |
| `FAULT-INJECTION-ISOLATION` | PASS | {"hits": {"INJ-001-NONFINITE-PIXEL": {"affected": ["rig/corner_00@instance-A"], "business_codes": ["E_OBS_INVALID"]}, "INJ-002-NONORTHOGONAL-ROTATION": {"affected": ["rig/corner_01@instance-B"], "business_codes": ["E_CAMERA_INVALID"]}, "INJ-003-DISPARITY-AB... |
| `GLOBAL-BUDGET` | PASS | {"budget": {"consumed": 0, "exhausted": true, "initial": 0, "ledger": [], "remaining": 0, "unit": "reprojection_evaluation"}, "pending_queue": ["aux/marker@instance-A", "bounds/tip@instance-A", "fragile/tip@instance-A", "low/parallax@instance-A", "rig/corne... |
| `POINT-QUOTA-AND-LEDGER` | PASS | {"budget": {"consumed": 87, "exhausted": false, "initial": 20000, "ledger": [{"after": 19994, "amount": 6, "before": 20000, "keypoint_id": "aux/marker@instance-A", "purpose": "point_solve", "sequence": 1}, {"after": 19988, "amount": 6, "before": 19994, "key... |
| `SNAPSHOT-MANIFEST-ATOMIC` | PASS | {"camera_hash": "19bef909e6490d83cb93623c45240691858fc0fa2037f9dd9539749a75f5e715", "completed_count": 16, "config_hash": "b065d3abbfb2dc1b85c9c4504ca14709e5d8e85dc339aa23a14c23e1c8b34105", "consumed_budget": 270, "global_version": "6ed872f03a283627d701476f... |
| `PRECISION-TRACE` | PASS | {"final_residual_count": 83, "final_residual_max_px": 0.14900253309921915, "final_residual_rmse_px": 0.0778931261374545, "first_tolerance_exceeded_stage_counts": {"triangulation": 7}, "float_format": "IEEE-754 binary64; JSON emits shortest round-trip decima... |

## Artifacts

- `reports/base_fresh.json`
- `reports/base_recovered.json`
- `reports/incremental_after_edit.json`
- `reports/fresh_after_edit.json`
- `reports/config_incremental.json`
- `reports/config_fresh.json`
- `reports/injection_run.json`
- `reports/budget_exhaustion.json`
- `reports/point_quota.json`

## Notes

- Residual budget units are active-observation reprojection evaluations.
- The report uses IEEE-754 binary64 values and never rounds JSON floats below 6 significant digits.
- Snapshot commits write a complete temporary tree and swap the `current` directory only when the content-addressed version changes.
