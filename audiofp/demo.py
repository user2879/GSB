"""Full deterministic demo: generate corpus, run twice, validate, and report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .atomic import atomic_write_json
from .engine import run_task
from .samplegen import generate_demo
from .task import load_task


def _time_overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def validate_demo(workspace: Path, report: dict[str, Any], task_path: Path) -> dict[str, Any]:
    task_raw = json.loads(task_path.read_text(encoding="utf-8"))
    track_rates = {
        track["id"]: track["sample_rate"]
        for group in ("references", "queries")
        for track in report["tracks"][group]
        if track.get("business_code") == "E_OK"
    }
    pair_lookup = {(pair["query_id"], pair["reference_id"]): pair for pair in report["pairs"]}
    localization: list[dict[str, Any]] = []
    max_error_seconds = 0.0
    for query_id, truth_data in task_raw.get("ground_truth", {}).items():
        truth_rate = int(truth_data["rate"])
        for truth_index, truth in enumerate(truth_data["segments"]):
            truth_ref_start = truth["reference_start_sample"] / truth_rate
            truth_ref_end = truth["reference_end_sample"] / truth_rate
            truth_query_start = truth["query_start_sample"] / truth_rate
            truth_query_end = truth["query_end_sample"] / truth_rate
            pair = pair_lookup.get((query_id, "ref_song"), {"fragments": []})
            best = None
            best_overlap = -1.0
            for fragment in pair["fragments"]:
                query_rate = track_rates[query_id]
                fragment_start = fragment["query_start_sample"] / query_rate
                fragment_end = fragment["query_end_sample"] / query_rate
                overlap = _time_overlap(fragment_start, fragment_end, truth_query_start, truth_query_end)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = fragment
            if best is None or best_overlap <= 0:
                localization.append(
                    {
                        "query_id": query_id,
                        "segment_index": truth_index,
                        "matched": False,
                        "reason": "no overlapping fragment",
                    }
                )
                continue
            reference_rate = track_rates[pair["reference_id"]]
            errors = {
                "reference_start": best["reference_start_sample"] / reference_rate - truth_ref_start,
                "reference_end": best["reference_end_sample"] / reference_rate - truth_ref_end,
                "query_start": best["query_start_sample"] / track_rates[query_id] - truth_query_start,
                "query_end": best["query_end_sample"] / track_rates[query_id] - truth_query_end,
            }
            max_abs = max(abs(value) for value in errors.values())
            max_error_seconds = max(max_error_seconds, max_abs)
            localization.append(
                {
                    "query_id": query_id,
                    "segment_index": truth_index,
                    "matched": True,
                    "query_overlap_seconds": best_overlap,
                    "errors_ms": {key: round(value * 1000.0, 3) for key, value in errors.items()},
                    "max_abs_error_ms": round(max_abs * 1000.0, 3),
                    "score": best["score"],
                    "integer_frame_offset": best["integer_frame_offset"],
                    "offset_reference_samples_float": best["offset_reference_samples_float"],
                    "offset_conversion_error_samples": best["offset_conversion_error_samples"],
                    "drift": best["drift"],
                }
            )
    injection_checks = [
        {
            "target_id": item["target_id"],
            "expected_code": item["expected_code"],
            "actual_code": item["actual_code"],
            "matched": item["matched"],
        }
        for item in report["injections"]
    ]
    offset_tolerance = report["parameters"]["tolerance"]["time_offset_tolerance_samples"]
    offset_checks = []
    for pair in report["pairs"]:
        for fragment in pair["fragments"]:
            offset_checks.append(
                {
                    "query_id": pair["query_id"],
                    "reference_id": pair["reference_id"],
                    "conversion_error_samples": fragment["offset_conversion_error_samples"],
                    "within_tolerance": fragment["offset_conversion_error_samples"] <= offset_tolerance,
                }
            )
    frame_checks = []
    for track in report["tracks"]["references"] + report["tracks"]["queries"]:
        for frame in track.get("frame_map", []):
            frame_checks.append(
                max(abs(frame["start_mapping_error_samples"]), abs(frame["end_mapping_error_samples"]))
            )
    speed_drift_checks = [
        {
            "query_id": item["query_id"],
            "segment_index": item["segment_index"],
            "direction": item.get("drift", {}).get("direction"),
            "drift_percent": item.get("drift", {}).get("drift_percent"),
            "passed": item.get("drift", {}).get("direction") == "decreasing"
            and abs(item.get("drift", {}).get("drift_percent", 0.0)) >= 0.5,
        }
        for item in localization
        if item["query_id"] == "q_speed_noise" and item.get("matched")
    ]
    speed_segments = task_raw.get("ground_truth", {}).get("q_speed_noise", {}).get("segments", [])
    insertion_gap_seconds = None
    if len(speed_segments) >= 2:
        rate = int(task_raw["ground_truth"]["q_speed_noise"]["rate"])
        insertion_gap_seconds = (
            speed_segments[1]["query_start_sample"] - speed_segments[0]["query_end_sample"]
        ) / rate
    return {
        "business_code": "E_OK"
        if all(item["matched"] for item in injection_checks)
        and max_error_seconds <= 0.36
        and all(item["within_tolerance"] for item in offset_checks)
        and (not frame_checks or max(frame_checks) <= 1.0)
        and all(item["passed"] for item in speed_drift_checks)
        else "E_PARTIAL",
        "localization_tolerance_seconds": 0.36,
        "max_localization_error_ms": round(max_error_seconds * 1000.0, 3),
        "localization": localization,
        "injection_checks": injection_checks,
        "offset_conversion_checks": offset_checks,
        "frame_mapping_max_error_samples": max(frame_checks) if frame_checks else None,
        "speed_drift_checks": speed_drift_checks,
        "local_insertion_gap_seconds": insertion_gap_seconds,
        "first_exceed_tolerance_present": all(
            fragment.get("first_exceed_score_tolerance") is not None
            for pair in report["pairs"]
            for fragment in pair["fragments"]
        ),
        "business_codes": {
            pair["query_id"] + "/" + pair["reference_id"]: pair["business_code"]
            for pair in report["pairs"]
        },
        "resource_control": report["resource_control"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic audio fingerprint end-to-end demo")
    parser.add_argument("--workspace", default=".", help="workspace directory")
    parser.add_argument("--no-clean", action="store_true", help="keep generated corpus before starting")
    parser.add_argument("--runs", type=int, default=2, help="number of full runs for determinism checks")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve()
    task_path = generate_demo(workspace, clean=not args.no_clean)
    task = load_task(task_path)
    reports: list[dict[str, Any]] = []
    result_hashes: list[str] = []
    for _ in range(max(1, args.runs)):
        report = run_task(task)
        reports.append(report)
        result_hashes.append(hashlib.sha256(task.results_path.read_bytes()).hexdigest())
    deterministic = len(set(result_hashes)) == 1
    validation = validate_demo(workspace, reports[-1], task_path)
    validation["deterministic_across_runs"] = deterministic
    validation["result_sha256"] = result_hashes[-1]
    validation["run_count"] = len(reports)
    if not deterministic:
        validation["business_code"] = "E_PARTIAL"
    validation_path = workspace / "output" / "demo_validation.json"
    atomic_write_json(validation_path, validation)
    print(
        json.dumps(
            {
                "business_code": validation["business_code"],
                "report": str(task.report_path),
                "results": str(task.results_path),
                "validation": str(validation_path),
                "max_localization_error_ms": validation["max_localization_error_ms"],
                "deterministic_across_runs": deterministic,
                "injections": validation["injection_checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if validation["business_code"] == "E_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
