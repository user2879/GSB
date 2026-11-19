#!/usr/bin/env python3
"""Run the complete local acceptance demonstration and write verification facts."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voicecal.engine import EnginePaths, run_engine
from voicecal.util import file_hash, json_hash, read_json, write_json

SAMPLES = ROOT / "samples"
RUNS = ROOT / "demo_runs"
COMPARED = (
    "initial_evidence.json",
    "calibration.json",
    "attribution.json",
    "calibration_ledger.json",
    "failures.json",
)


def paths(name: str, *, injections: Path | None = None, config: Path | None = None) -> EnginePaths:
    return EnginePaths(
        sample_dir=SAMPLES,
        dialogue_path=SAMPLES / "dialogue.json",
        config_path=config or SAMPLES / "calibration.json",
        edits_path=SAMPLES / "edits.json",
        injections_path=injections,
        snapshot_dir=RUNS / name / "snapshot",
        output_dir=RUNS / name / "output",
    )


def fresh_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    incremental_root = RUNS / "incremental"
    full_root = RUNS / "full"
    fresh_dir(incremental_root)
    fresh_dir(full_root)

    iterations = 0
    first_pending_code: str | None = None
    max_reuse = {stage: 0 for stage in ("extract", "initial", "calibration", "attribution")}
    while True:
        iterations += 1
        report = run_engine(paths("incremental"), fresh=False, process_all_batches=True)
        for stage, count in report["ledger"]["hit_counts"].items():
            max_reuse[stage] = max(max_reuse[stage], count)
        if "E_CALIB_BUDGET_PENDING" in report.get("business_codes", []):
            first_pending_code = "E_CALIB_BUDGET_PENDING"
        if report["state"]["complete"]:
            break
        if iterations > 30:
            raise RuntimeError("incremental demonstration did not converge")

    full_report = run_engine(paths("full"), fresh=True, process_all_batches=True)
    incremental_output = incremental_root / "output"
    full_output = full_root / "output"
    comparisons: dict[str, Any] = {}
    all_equal = True
    for filename in COMPARED:
        incremental_hash = file_hash(incremental_output / filename)
        full_hash = file_hash(full_output / filename)
        equal = incremental_hash == full_hash
        all_equal = all_equal and equal
        comparisons[filename] = {"equal": equal, "incremental_sha256": incremental_hash, "full_sha256": full_hash}

    one_thread_config_path = RUNS / "calibration_concurrency_1.json"
    config = read_json(SAMPLES / "calibration.json")
    config["concurrency"] = 1
    write_json(one_thread_config_path, config)
    serial_root = RUNS / "serial-reference"
    fresh_dir(serial_root)
    serial_report = run_engine(paths("serial-reference", config=one_thread_config_path), fresh=True)
    serial_comparisons: dict[str, Any] = {}
    serial_equal = True
    for filename in COMPARED:
        parallel_hash = file_hash(full_output / filename)
        serial_hash = file_hash(serial_root / "output" / filename)
        equal = parallel_hash == serial_hash
        serial_equal = serial_equal and equal
        serial_comparisons[filename] = {"equal": equal, "parallel_sha256": parallel_hash, "serial_sha256": serial_hash}

    injection_root = RUNS / "injection-validation"
    fresh_dir(injection_root)
    injection_paths = EnginePaths(
        sample_dir=SAMPLES,
        dialogue_path=SAMPLES / "dialogue.json",
        config_path=SAMPLES / "calibration.json",
        edits_path=SAMPLES / "no_edits.json",
        injections_path=SAMPLES / "injections_validation.json",
        snapshot_dir=injection_root / "snapshot",
        output_dir=injection_root / "output",
    )
    injection_report = run_engine(injection_paths, fresh=True)
    injection_codes = {(hit["injection_id"], hit["business_code"]) for hit in injection_report["injection"]["hits"]}
    expected_codes = read_json(SAMPLES / "expected_business_codes.json")["injection_expectations"]
    injection_expectations_met = all(
        (item["injection_id"], item["expected_code"]) in injection_codes for item in expected_codes
    )
    failed_turn_ids = {failure.get("turn_id") for failure in injection_report["failures"] if failure.get("turn_id")}
    injected_turns_excluded = all(item["turn_id"] in failed_turn_ids for item in expected_codes)

    final_ledger = read_json(incremental_output / "calibration_ledger.json")
    final_attribution = read_json(incremental_output / "attribution.json")
    verification = {
        "schema_version": 1,
        "incremental_iterations": iterations,
        "first_pending_code_observed": first_pending_code,
        "incremental_vs_full_equal": all_equal,
        "incremental_vs_full": comparisons,
        "serial_vs_parallel_equal": serial_equal,
        "serial_vs_parallel": serial_comparisons,
        "max_reuse_counts_observed": max_reuse,
        "final_turn_count": len(final_ledger["entries"]),
        "final_trajectory_count": len(final_attribution["trajectories"]),
        "final_failure_count": len(read_json(incremental_output / "failures.json")["failures"]),
        "injection_expectations_met": injection_expectations_met,
        "injected_turns_excluded_from_trajectories": injected_turns_excluded,
        "injection_hits": injection_report["injection"]["hits"],
        "injection_failures": injection_report["failures"],
        "full_report_business_codes": full_report["business_codes"],
        "serial_report_business_codes": serial_report["business_codes"],
        "final_ledger_digest": json_hash(final_ledger),
    }
    write_json(ROOT / "docs" / "verification.json", verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    if not all_equal or not serial_equal or not injection_expectations_met or not injected_turns_excluded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
