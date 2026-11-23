#!/usr/bin/env python3
"""Re-run local samples and produce a deterministic acceptance summary."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rerank_lab.canonical import digest
from rerank_lab.engine import run_experiment


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _clean_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(value), encoding="utf-8", newline="\n")


def _report_text(report: dict[str, Any]) -> str:
    return json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_numeric_invariants(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for request in report.get("requests", []):
        lower = request["score_min_fixed"]
        upper = request["score_max_fixed"]
        for score in request["fusion"]["candidates"].values():
            contribution_sum = sum(
                item["contribution_fixed"] for item in score["contributions"]
            )
            if contribution_sum != score["raw_score_fixed"]:
                failures.append(
                    f"{request['request_id']}/{score['item_id']}: contribution sum mismatch"
                )
            if not lower <= score["score_fixed"] <= upper:
                failures.append(
                    f"{request['request_id']}/{score['item_id']}: fusion score outside clamp"
                )
        for item_id, score_fixed in request["rerank"]["scores_after_rerank"].items():
            if score_fixed < lower:
                failures.append(
                    f"{request['request_id']}/{item_id}: rerank score below lower bound"
                )
        if len(request["order"]) != len(request["candidate_ids"]):
            failures.append(f"{request['request_id']}: final order length mismatch")
    return failures


def _run_default(default_sample: Path, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if output.exists():
        shutil.rmtree(output)
    report = run_experiment(default_sample, output)
    _clean_write(output / "report.json", report)
    with tempfile.TemporaryDirectory(prefix="rerank-repeat-") as temp:
        repeat = run_experiment(default_sample, Path(temp))
    text_match = _report_text(report) == _report_text(repeat)
    return report, {
        "report_byte_equivalent": text_match,
        "report_hash": digest(report),
        "repeat_report_hash": digest(repeat),
        "projection_match": report.get("consistency", {}).get(
            "fresh_cold_match", False
        ),
        "numeric_invariant_violations": _validate_numeric_invariants(report),
        "snapshot_manifest_exists": (output / "snapshot" / "manifest.json").is_file(),
    }


def _serial_parallel_check(sample: Path, root: Path) -> dict[str, Any]:
    serial = root / "serial-sample"
    parallel = root / "parallel-sample"
    if serial.exists():
        shutil.rmtree(serial)
    if parallel.exists():
        shutil.rmtree(parallel)
    shutil.copytree(sample, serial)
    shutil.copytree(sample, parallel)
    for target, concurrency in ((serial, 1), (parallel, 8)):
        config_path = target / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["concurrency"] = concurrency
        config["global_mac_budget"] = 5000
        config_path.write_text(_json_text(config), encoding="utf-8", newline="\n")
        (target / "commands.jsonl").write_text("", encoding="utf-8", newline="\n")
    serial_report = run_experiment(serial, root / "serial-output", enable_snapshots=False)
    parallel_report = run_experiment(
        parallel, root / "parallel-output", enable_snapshots=False
    )
    return {
        "requests_equal": serial_report["requests"] == parallel_report["requests"],
        "mac_units_equal": serial_report["schedule"]["mac_units_charged"]
        == parallel_report["schedule"]["mac_units_charged"],
        "serial_concurrency": serial_report["schedule"]["concurrency"],
        "parallel_concurrency": parallel_report["schedule"]["concurrency"],
    }


def _latency_check(sample: Path, root: Path) -> dict[str, Any]:
    target = root / "latency-sample"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(sample, target)
    config_path = target / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "interaction_latency_budget_ms": 0.05,
            "min_command_batch": 5,
            "latency_units_per_ms": 1000,
        }
    )
    config_path.write_text(_json_text(config), encoding="utf-8", newline="\n")
    report = run_experiment(target, root / "latency-output", enable_snapshots=False)
    return {
        "all_responses_stale": all(
            record["response"]["status"] == "stale"
            for record in report["commands"]
        ),
        "stale_marker_present": all(
            "E_RESULT_STALE" in record["response"]["error_codes"]
            for record in report["commands"]
        ),
        "measurement_scope": sorted(
            {
                record["response"]["measurement"]["scope"]
                for record in report["commands"]
            }
        ),
        "version_sources": [
            record["response"]["version_source"] for record in report["commands"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-sample", default="samples/default")
    parser.add_argument("--budget-sample", default="samples/budget")
    parser.add_argument("--injection-sample", default="samples/injection")
    parser.add_argument("--output-dir", default="artifacts/acceptance")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    default_sample = Path(args.default_sample)
    budget_sample = Path(args.budget_sample)
    injection_sample = Path(args.injection_sample)

    default_report, determinism = _run_default(
        default_sample, output_root / "default-output"
    )
    budget_report = run_experiment(
        budget_sample, output_root / "budget-output", clean=True
    )
    _clean_write(output_root / "budget-output" / "report.json", budget_report)
    injection_report = run_experiment(
        injection_sample, output_root / "injection-output", clean=True
    )
    _clean_write(
        output_root / "injection-output" / "report.json", injection_report
    )
    serial_parallel = _serial_parallel_check(
        budget_sample, output_root / "serial-parallel-temp"
    )
    latency = _latency_check(default_sample, output_root / "latency-temp")

    checks = {
        "default_status_ok": default_report["summary"]["status"] == "ok",
        "determinism": determinism,
        "budget": {
            "completed_requests": budget_report["summary"]["completed_requests"],
            "pending_requests": budget_report["summary"]["pending_requests"],
            "pending_codes": sorted(
                {item["code"] for item in budget_report["pending"]}
            ),
            "charged": budget_report["schedule"]["mac_units_charged"],
            "remaining": budget_report["schedule"]["mac_units_remaining"],
        },
        "injection": {
            "hit_ids": [
                item["injection_id"] for item in injection_report["injection_hits"]
            ],
            "completed_requests": injection_report["summary"]["completed_requests"],
            "failed_requests": injection_report["summary"]["failed_requests"],
            "eligible_requests": injection_report["summary"]["eligible_requests"],
            "failure_codes": sorted(
                {item["code"] for item in injection_report["failures"]}
            ),
        },
        "serial_parallel": serial_parallel,
        "latency": latency,
    }
    passed = (
        checks["default_status_ok"]
        and determinism["report_byte_equivalent"]
        and determinism["projection_match"]
        and not determinism["numeric_invariant_violations"]
        and determinism["snapshot_manifest_exists"]
        and checks["budget"]["completed_requests"] == 1
        and checks["budget"]["pending_requests"] == 3
        and checks["budget"]["pending_codes"] == ["E_GLOBAL_BUDGET_PENDING"]
        and checks["injection"]["hit_ids"] == ["INJ-001", "INJ-002", "INJ-003"]
        and checks["injection"]["completed_requests"] == 1
        and checks["injection"]["failed_requests"] == 3
        and checks["injection"]["eligible_requests"] == 1
        and serial_parallel["requests_equal"]
        and serial_parallel["mac_units_equal"]
        and serial_parallel["serial_concurrency"] == 1
        and serial_parallel["parallel_concurrency"] == 8
        and latency["all_responses_stale"]
        and latency["stale_marker_present"]
        and latency["measurement_scope"] == ["min_command_batch"]
    )
    for transient in (
        output_root / "serial-parallel-temp",
        output_root / "latency-temp",
    ):
        if transient.exists():
            shutil.rmtree(transient)

    summary = {
        "status": "pass" if passed else "fail",
        "checks": checks,
        "artifacts": {
            "default_report": "default-output/report.json",
            "budget_report": "budget-output/report.json",
            "injection_report": "injection-output/report.json",
        },
        "environment": {
            "python": "3.11+",
            "dependencies": ["numpy==1.26.4", "PyYAML==6.0.2"],
            "network_required": False,
            "wall_clock_in_canonical_reports": False,
        },
    }
    _clean_write(output_root / "acceptance.json", summary)
    print(_json_text(summary), end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
