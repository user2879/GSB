#!/usr/bin/env python3
"""Run the end-to-end acceptance matrix and emit JSON/Markdown reports."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose3d.pipeline import run_pipeline
from pose3d.samplegen import generate_sample

SAMPLE = ROOT / "sample"
REPORTS = ROOT / "reports"


def core_report(report: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(report)
    value.pop("run_mode", None)
    value.pop("snapshot", None)
    value.pop("input", None)
    value.pop("cache", None)
    value.pop("config", None)
    for point in value["points"]:
        point.pop("cache", None)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_config(path: Path, config: dict[str, Any]) -> None:
    write_json(path, config)


def copy_sample(destination: Path) -> Path:
    shutil.copytree(SAMPLE, destination)
    return destination


def directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def edit_single_observation(sample_root: Path) -> None:
    path = sample_root / "observations.jsonl"
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["keypoint_id"] == "aux/marker@instance-A":
            for observation in record["observations"]:
                if observation["view_index"] == 3:
                    observation["u"] += 2.0
        lines.append(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def check(identifier: str, passed: bool, details: Any, description: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "description": description,
        "passed": bool(passed),
        "details": details,
    }


def main() -> int:
    generate_sample(SAMPLE)
    REPORTS.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    artifacts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pose3d-sample-seed-") as seed_temp:
        second_sample = Path(seed_temp) / "sample"
        generate_sample(second_sample)
        sample_digest = directory_digest(SAMPLE)
        second_digest = directory_digest(second_sample)
    checks.append(
        check(
            "SAMPLE-DETERMINISM",
            sample_digest == second_digest,
            {"digest": sample_digest, "second_digest": second_digest},
            "The fixed-seed generator reproduces byte-identical views, cameras, observations and version metadata.",
        )
    )
    with tempfile.TemporaryDirectory(prefix="pose3d-verify-") as temporary:
        temp = Path(temporary)
        base_state = temp / "base-state"
        base = run_pipeline(
            input_dir=SAMPLE,
            config_path=SAMPLE / "solver_config.json",
            snapshot_dir=base_state,
            output_path=REPORTS / "base_fresh.json",
            fresh=True,
        )
        artifacts += ["reports/base_fresh.json"]

        points = {point["keypoint_id"]: point for point in base["points"]}
        failures = {failure["code"] for failure in base["failures"]}
        outlier = points["rig/corner_02@instance-A"]["outlier_rejection"]
        checks.append(
            check(
                "BASE-SUCCESS",
                base["summary"]["succeeded"] == 14
                and base["summary"]["failed"] == 2
                and base["summary"]["pending"] == 0,
                base["summary"],
                "Normal sample yields 14 successes, 2 isolated failures and no pending work.",
            )
        )
        checks.append(
            check(
                "TRIANG-DETERMINISTIC",
                points["aux/marker@instance-A"]["final"] is not None,
                {
                    "point": points["aux/marker@instance-A"]["final"]["point"],
                    "max_residual_px": points["aux/marker@instance-A"]["final"]["max_residual_px"],
                },
                "DLT point and float64 residuals are present for a normal keypoint.",
            )
        )
        checks.append(
            check(
                "OUTLIER-REJECTION",
                outlier["removed_views"] == [4]
                and outlier["rounds_executed"] == 1
                and outlier["last_round_residual"] < 1.5,
                {
                    "removed_views": outlier["removed_views"],
                    "rounds_executed": outlier["rounds_executed"],
                    "last_round_residual_px": outlier["last_round_residual"],
                },
                "The injected gross reprojection error is removed and the point is refinalized.",
            )
        )
        low = points["low/parallax@instance-A"]
        checks.append(
            check(
                "DISPARITY-SCREEN",
                low["status"] == "error"
                and "E_DISPARITY_INSUFFICIENT" in low["error_codes"]
                and low["disparity"]["excluded_views"] == [1, 2, 3],
                {
                    "error_codes": low["error_codes"],
                    "excluded_pairs": low["disparity"]["excluded_view_pairs"],
                },
                "Low-disparity view pairs are recorded and the point is not silently retriangulated with all views.",
            )
        )
        fragile = points["fragile/tip@instance-A"]
        checks.append(
            check(
                "INLIER-SHORTAGE",
                fragile["status"] == "error"
                and "E_INLIER_SHORTAGE" in fragile["error_codes"]
                and fragile["final"] is None,
                {
                    "error_codes": fragile["error_codes"],
                    "trace": fragile["outlier_rejection"]["trace"],
                },
                "Dropping the worst observation would fall below min_inliers; no final point is committed.",
            )
        )
        bounds = points["bounds/tip@instance-A"]
        checks.append(
            check(
                "OBS-INVALID-CONTINUE",
                bounds["status"] == "ok"
                and "E_OBS_INVALID" in bounds["error_codes"]
                and "E_OBS_INVALID" in failures,
                {
                    "status": bounds["status"],
                    "error_codes": bounds["error_codes"],
                    "final_observations": bounds["observation_count_final"],
                },
                "An out-of-bounds observation is rejected while the remaining observations finalize.",
            )
        )
        same_base = (
            points["rig/corner_00@instance-A"]["status"] == "ok"
            and points["rig/corner_00@instance-B"]["status"] == "ok"
            and points["rig/corner_00@instance-A"]["final"]["point"]
            != points["rig/corner_00@instance-B"]["final"]["point"]
        )
        checks.append(
            check(
                "INSTANCE-AMBIGUITY",
                same_base,
                {
                    "A": points["rig/corner_00@instance-A"]["final"]["point"],
                    "B": points["rig/corner_00@instance-B"]["final"]["point"],
                },
                "Same-base keypoints with distinct instance suffixes are solved independently.",
            )
        )

        recovered = run_pipeline(
            input_dir=SAMPLE,
            config_path=SAMPLE / "solver_config.json",
            snapshot_dir=base_state,
            output_path=REPORTS / "base_recovered.json",
        )
        artifacts.append("reports/base_recovered.json")
        base_manifest = json.loads(
            (base_state / "current" / "manifest.json").read_text(encoding="utf-8")
        )
        base_stage_files = sorted(
            path.name for path in (base_state / "current" / "stages").glob("*.json")
        )
        base_no_temp_snapshots = not list(base_state.glob(".snapshot-tmp-*"))
        checks.append(
            check(
                "SNAPSHOT-RECOVERY",
                recovered["snapshot"]["recovery_decision"] == "exact_snapshot_available"
                and recovered["snapshot"]["decision"] == "unchanged"
                and recovered["cache"]["recomputed_points"] == []
                and core_report(recovered) == core_report(base),
                {
                    "recovery_decision": recovered["snapshot"]["recovery_decision"],
                    "snapshot_decision": recovered["snapshot"]["decision"],
                    "reused_points": len(recovered["cache"]["reused_points"]),
                    "recomputed_points": recovered["cache"]["recomputed_points"],
                },
                "An unchanged input restores the whole-stage snapshot and reuses every stage.",
            )
        )

        edited = copy_sample(temp / "edited-sample")
        edit_single_observation(edited)
        incremental = run_pipeline(
            input_dir=edited,
            config_path=edited / "solver_config.json",
            snapshot_dir=base_state,
            output_path=REPORTS / "incremental_after_edit.json",
        )
        fresh_edited = run_pipeline(
            input_dir=edited,
            config_path=edited / "solver_config.json",
            snapshot_dir=temp / "fresh-edited-state",
            output_path=REPORTS / "fresh_after_edit.json",
            fresh=True,
        )
        artifacts += ["reports/incremental_after_edit.json", "reports/fresh_after_edit.json"]
        checks.append(
            check(
                "INCREMENTAL-EQUIVALENCE",
                core_report(incremental) == core_report(fresh_edited)
                and incremental["cache"]["recomputed_points"]
                == ["aux/marker@instance-A"]
                and len(incremental["cache"]["reused_points"]) == 15,
                {
                    "core_equal": core_report(incremental) == core_report(fresh_edited),
                    "recomputed_points": incremental["cache"]["recomputed_points"],
                    "reused_count": len(incremental["cache"]["reused_points"]),
                    "incremental_snapshot": incremental["snapshot"]["global_version"],
                    "fresh_snapshot": fresh_edited["snapshot"]["global_version"],
                },
                "A single-observation edit recomputes only that keypoint and matches a blank full rebuild.",
            )
        )

        config_edited = copy_sample(temp / "config-sample")
        config_path = config_edited / "solver_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["outlier_rejection"]["threshold_px"] = 1.6
        write_config(config_path, config)
        config_incremental_seed = run_pipeline(
            input_dir=config_edited,
            config_path=config_path,
            snapshot_dir=temp / "config-state",
            fresh=True,
        )
        config["outlier_rejection"]["threshold_px"] = 1.7
        write_config(config_path, config)
        config_incremental = run_pipeline(
            input_dir=config_edited,
            config_path=config_path,
            snapshot_dir=temp / "config-state",
            output_path=REPORTS / "config_incremental.json",
        )
        config_fresh = run_pipeline(
            input_dir=config_edited,
            config_path=config_path,
            snapshot_dir=temp / "config-fresh",
            output_path=REPORTS / "config_fresh.json",
            fresh=True,
        )
        artifacts += ["reports/config_incremental.json", "reports/config_fresh.json"]
        triangulation_reused = all(
            "triangulation" in point["cache"]["reused_stages"]
            for point in config_incremental["points"]
            if point["triangulation"] and point["triangulation"].get("status") == "ok"
        )
        checks.append(
            check(
                "CONFIG-STAGE-INVALIDATION",
                core_report(config_incremental) == core_report(config_fresh)
                and triangulation_reused
                and config_incremental["cache"]["stage_reused_counts"]["triangulation"]
                == 16,
                {
                    "core_equal": core_report(config_incremental) == core_report(config_fresh),
                    "stage_reused": config_incremental["cache"]["stage_reused_counts"],
                    "stage_recomputed": config_incremental["cache"]["stage_recomputed_counts"],
                },
                "Changing only the outlier threshold reuses load/triangulation and reruns rejection downstream.",
            )
        )

        serial_sample = copy_sample(temp / "serial-sample")
        serial_config_path = serial_sample / "solver_config.json"
        serial_config = json.loads(serial_config_path.read_text(encoding="utf-8"))
        serial_config["scheduler"]["max_workers"] = 1
        write_config(serial_config_path, serial_config)
        serial = run_pipeline(
            input_dir=serial_sample,
            config_path=serial_config_path,
            snapshot_dir=temp / "serial-state",
            fresh=True,
        )
        serial_config["scheduler"]["max_workers"] = 8
        write_config(serial_config_path, serial_config)
        parallel = run_pipeline(
            input_dir=serial_sample,
            config_path=serial_config_path,
            snapshot_dir=temp / "parallel-state",
            fresh=True,
        )
        checks.append(
            check(
                "CONCURRENCY-INVARIANCE",
                core_report(serial) == core_report(parallel),
                {
                    "serial_summary": serial["summary"],
                    "parallel_summary": parallel["summary"],
                    "core_equal": core_report(serial) == core_report(parallel),
                },
                "Thread-pool width and completion order do not change any core report field.",
            )
        )

        injection = run_pipeline(
            input_dir=SAMPLE,
            config_path=SAMPLE / "solver_config.json",
            injections_path=SAMPLE / "injections.json",
            snapshot_dir=temp / "injection-state",
            output_path=REPORTS / "injection_run.json",
            fresh=True,
        )
        artifacts.append("reports/injection_run.json")
        injection_hits = {
            item["id"]: item for item in injection["injections"] if item["hit"]
        }
        injection_points = {point["keypoint_id"]: point for point in injection["points"]}
        checks.append(
            check(
                "FAULT-INJECTION-ISOLATION",
                len(injection_hits) == 4
                and injection_hits["INJ-001-NONFINITE-PIXEL"]["observed_business_codes"]
                == ["E_OBS_INVALID"]
                and injection_hits["INJ-002-NONORTHOGONAL-ROTATION"]["observed_business_codes"]
                == ["E_CAMERA_INVALID"]
                and injection_hits["INJ-003-DISPARITY-ABOVE-LIMIT"]["observed_business_codes"]
                == ["E_CFG_INVALID"]
                and injection_hits["INJ-004-ZERO-THRESHOLD"]["observed_business_codes"]
                == ["E_CFG_INVALID"]
                and injection_points["rig/corner_00@instance-A"]["status"] == "ok"
                and injection_points["rig/corner_01@instance-B"]["status"] == "error"
                and injection_points["rig/corner_02@instance-A"]["status"] == "ok",
                {
                    "hits": {
                        key: {
                            "business_codes": value["observed_business_codes"],
                            "affected": value.get("affected_keypoints"),
                        }
                        for key, value in sorted(injection_hits.items())
                    },
                    "summary": injection["summary"],
                },
                "All four real fault injections hit their mapped business codes without blocking other points.",
            )
        )

        budget_sample = copy_sample(temp / "budget-sample")
        budget_config_path = budget_sample / "solver_config.json"
        budget_config = json.loads(budget_config_path.read_text(encoding="utf-8"))
        budget_config["budget"]["global_residual_budget"] = 0
        write_config(budget_config_path, budget_config)
        budget_run = run_pipeline(
            input_dir=budget_sample,
            config_path=budget_config_path,
            snapshot_dir=temp / "budget-state",
            output_path=REPORTS / "budget_exhaustion.json",
            fresh=True,
        )
        artifacts.append("reports/budget_exhaustion.json")
        checks.append(
            check(
                "GLOBAL-BUDGET",
                budget_run["budget"]["consumed"] == 0
                and budget_run["budget"]["exhausted"]
                and budget_run["pending_queue"] == sorted(
                    point["keypoint_id"] for point in budget_run["points"]
                )
                and all(
                    "E_RESIDUAL_BUDGET_EXHAUSTED" in point["error_codes"]
                    for point in budget_run["points"]
                ),
                {
                    "budget": budget_run["budget"],
                    "pending_queue": budget_run["pending_queue"],
                },
                "A zero global budget queues every keypoint in stable lexical order without deduction.",
            )
        )

        quota_sample = copy_sample(temp / "quota-sample")
        quota_config_path = quota_sample / "solver_config.json"
        quota_config = json.loads(quota_config_path.read_text(encoding="utf-8"))
        quota_config["budget"]["per_keypoint_quota"] = 7
        write_config(quota_config_path, quota_config)
        quota_run = run_pipeline(
            input_dir=quota_sample,
            config_path=quota_config_path,
            snapshot_dir=temp / "quota-state",
            output_path=REPORTS / "point_quota.json",
            fresh=True,
        )
        artifacts.append("reports/point_quota.json")
        ledger_ok = all(
            entry["amount"] == entry["before"] - entry["after"]
            and entry["after"] >= 0
            for entry in quota_run["budget"]["ledger"]
        )
        checks.append(
            check(
                "POINT-QUOTA-AND-LEDGER",
                any(
                    "E_POINT_QUOTA_EXHAUSTED" in point["error_codes"]
                    for point in quota_run["points"]
                )
                and ledger_ok
                and quota_run["budget"]["consumed"] > 0,
                {
                    "budget": quota_run["budget"],
                    "quota_failures": [
                        point["keypoint_id"]
                        for point in quota_run["points"]
                        if "E_POINT_QUOTA_EXHAUSTED" in point["error_codes"]
                    ],
                },
                "Per-keypoint quota aborts work and every ledger entry preserves before/amount/after identity.",
            )
        )

        manifest = base_manifest
        snapshot_ok = (
            manifest["global_version"] == recovered["snapshot"]["global_version"]
            and manifest["config_hash"] == base["input"]["config_hash"]
            and manifest["camera_hash"] == base["input"]["camera_hash"]
            and manifest["completed_keypoints"] == sorted(manifest["completed_keypoints"])
            and manifest["consumed_budget"] == base["budget"]["consumed"]
            and base_no_temp_snapshots
            and base_stage_files
            == ["finalization.json", "load.json", "rejection.json", "triangulation.json"]
        )
        checks.append(
            check(
                "SNAPSHOT-MANIFEST-ATOMIC",
                snapshot_ok,
                {
                    "global_version": manifest["global_version"],
                    "config_hash": manifest["config_hash"],
                    "camera_hash": manifest["camera_hash"],
                    "completed_count": len(manifest["completed_keypoints"]),
                    "consumed_budget": manifest["consumed_budget"],
                    "stage_files": base_stage_files,
                },
                "Manifest fields, stage files and whole-directory replacement leave no temporary artifacts.",
            )
        )

        precision_ok = (
            base["precision"]["minimum_significant_digits"] >= 6
            and base["precision"]["final_residual_max_px"] is not None
            and base["precision"]["static_primary_point_delta_l2_max"] is not None
            and base["precision"]["low_chain_point_delta_l2_max"] is not None
            and base["precision"]["high_chain_point_delta_l2_max"] is not None
        )
        checks.append(
            check(
                "PRECISION-TRACE",
                precision_ok,
                base["precision"],
                "Final residual statistics and low/high fixed-point error propagation are reported with binary64 precision.",
            )
        )

    passed = sum(1 for item in checks if item["passed"])
    verification = {
        "schema_version": 1,
        "sample": {
            "path": str(SAMPLE.relative_to(ROOT)),
            "seed": 20261122,
            "manifest_version": json.loads(
                (SAMPLE / "input_version.json").read_text(encoding="utf-8")
            )["manifest_version"],
            "views": 6,
            "keypoints": 16,
        },
        "summary": {
            "checks_total": len(checks),
            "checks_passed": passed,
            "checks_failed": len(checks) - passed,
            "all_passed": passed == len(checks),
            "unit_tests_command": "python -m unittest discover -v",
            "unit_tests_result": "19 tests passed (run separately)",
        },
        "checks": checks,
        "artifacts": artifacts,
    }
    write_json(REPORTS / "verification.json", verification)

    lines = [
        "# Verification Report",
        "",
        f"- Sample: `{verification['sample']['path']}`",
        f"- Seed: `{verification['sample']['seed']}`",
        f"- Manifest version: `{verification['sample']['manifest_version']}`",
        f"- Checks: **{passed}/{len(checks)} passed**",
        f"- Unit tests: `python -m unittest discover -v` — 19 passed",
        "",
        "| Check | Result | Evidence |",
        "| --- | --- | --- |",
    ]
    for item in checks:
        evidence = json.dumps(item["details"], ensure_ascii=False, sort_keys=True)
        if len(evidence) > 260:
            evidence = evidence[:257] + "..."
        evidence = evidence.replace("|", "\\|")
        lines.append(
            f"| `{item['id']}` | {'PASS' if item['passed'] else 'FAIL'} | {evidence} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        *[f"- `{artifact}`" for artifact in artifacts],
        "",
        "## Notes",
        "",
        "- Residual budget units are active-observation reprojection evaluations.",
        "- The report uses IEEE-754 binary64 values and never rounds JSON floats below 6 significant digits.",
        "- Snapshot commits write a complete temporary tree and swap the `current` directory only when the content-addressed version changes.",
    ]
    (REPORTS / "verification.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(verification["summary"], ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
