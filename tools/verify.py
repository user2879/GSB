#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workbench.engine import RunConfig, Workbench  # noqa: E402
from workbench.errors import SnapshotStaleError  # noqa: E402
from workbench.loader import load_plan  # noqa: E402
from workbench.util import read_json, stable_hash  # noqa: E402


SAMPLES = ROOT / "samples"
DEMO = ROOT / "demo"


def run(
    output: Path,
    step: int,
    *,
    mode: str = "incremental",
    serial: bool = False,
    budget: int | None = None,
) -> dict:
    report = Workbench(
        RunConfig(
            samples_dir=SAMPLES,
            output_dir=output,
            step=step,
            mode=mode,
            serial=serial,
            budget_override=budget,
        )
    ).run()
    print(
        f"run mode={mode} step={step} serial={serial} budget={budget} "
        f"version={report['run']['global_version']} "
        f"candidates={report['summary']['unique_candidate_count']} "
        f"pending={report['summary']['pending_count']} "
        f"failures={report['summary']['failure_count']} "
        f"canonical={report['canonical_hash'][:12]}",
        flush=True,
    )
    return report


def clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    clean(DEMO)
    incremental = DEMO / "incremental"
    full = DEMO / "full"
    serial = DEMO / "serial"
    low_budget = DEMO / "low_budget"
    stale_probe = DEMO / "stale_probe"

    run(incremental, 0)
    step_reports = []
    for step_def in load_plan(SAMPLES)["steps"]:
        step = step_def["step"]
        report = run(incremental, step)
        step_reports.append(
            {
                "step": step,
                "global_version": report["run"]["global_version"],
                "canonical_hash": report["canonical_hash"],
                "shards": report["summary"]["shard_count"],
                "candidates": report["summary"]["unique_candidate_count"],
                "pending": report["summary"]["pending_count"],
                "failures": report["summary"]["failure_count"],
                "cache_invalidated": report["summary"]["invalidated_cache_entries"],
                "cache_reused": report["summary"]["reused_cache_entries"],
            }
        )
    final_step = max(step["step"] for step in load_plan(SAMPLES)["steps"])
    incremental_report = read_json(incremental / "report.json")
    incremental_canonical = read_json(incremental / "canonical.json")
    full_report = run(full, final_step, mode="full")
    full_canonical = read_json(full / "canonical.json")
    serial_report = run(serial, final_step, mode="full", serial=True)
    serial_canonical = read_json(serial / "canonical.json")

    budget_round_records = []
    for _ in range(20):
        report = run(low_budget, final_step, budget=3)
        budget_round_records.append(
            {
                "round": report["budget"]["current_round"]["round"],
                "consumed": report["budget"]["current_round"]["consumed"],
                "pending_after": len(report["pending"]),
                "deductions": [
                    [entry["request_id"], entry["item_id"]]
                    for entry in report["budget"]["current_round"]["deductions"]
                ],
            }
        )
        if not report["pending"]:
            break
    budget_report = read_json(low_budget / "report.json")
    budget_canonical = read_json(low_budget / "canonical.json")

    run(stale_probe, 0)
    manifest_path = stale_probe / "live" / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["config_hash"] = "tampered-for-verification"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stale_code = None
    try:
        run(stale_probe, 1)
    except SnapshotStaleError as exc:
        stale_code = exc.code

    boundary_counts = Counter(
        sample["criterion"] for sample in incremental_report["boundary_samples"]
    )
    verification = {
        "ok": (
            incremental_canonical == full_canonical
            and full_canonical == serial_canonical
            and full_canonical == budget_canonical
            and stale_code == "E_SNAPSHOT_STALE"
            and not budget_report["pending"]
        ),
        "python": platform.python_version(),
        "sample_seed": read_json(SAMPLES / "seed_manifest.json")["seed"],
        "incremental_steps": step_reports,
        "final_incremental": {
            "global_version": incremental_report["run"]["global_version"],
            "canonical_hash": incremental_report["canonical_hash"],
            "cache_invalidated": incremental_report["cache"]["totals"]["invalidated"],
            "cache_reused": incremental_report["cache"]["totals"]["reused"],
            "failures": incremental_report["failure_list"],
            "injection_hits": [
                hit for hit in incremental_report["injection_hits"] if hit["hit"]
            ],
            "boundary_counts": dict(sorted(boundary_counts.items())),
            "discount_count": len(incremental_report["discount_records"]),
            "stage_versions": incremental_report["snapshots"]["stage_versions"],
        },
        "incremental_vs_full": {
            "equal": incremental_canonical == full_canonical,
            "incremental_hash": stable_hash(incremental_canonical),
            "full_hash": stable_hash(full_canonical),
        },
        "thread_pool_vs_serial": {
            "equal": full_canonical == serial_canonical,
            "thread_pool_hash": full_report["canonical_hash"],
            "serial_hash": serial_report["canonical_hash"],
        },
        "budget": {
            "budget_per_round": 3,
            "rounds": budget_round_records,
            "pending_empty": not budget_report["pending"],
            "canonical_equals_full": budget_canonical == full_canonical,
            "canonical_hash": budget_report["canonical_hash"],
            "audit_round_count": len(budget_report["budget"]["rounds"]),
        },
        "snapshot_stale_probe": {
            "expected_code": "E_SNAPSHOT_STALE",
            "observed_code": stale_code,
            "error_report": read_json(stale_probe / "error_report.json"),
        },
    }
    (DEMO / "verification_report.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (DEMO / "comparison.json").write_text(
        json.dumps(
            {
                "incremental_vs_full": {
                    "equal": incremental_canonical == full_canonical,
                    "left_hash": stable_hash(incremental_canonical),
                    "right_hash": stable_hash(full_canonical),
                },
                "thread_pool_vs_serial": {
                    "equal": full_canonical == serial_canonical,
                    "left_hash": stable_hash(full_canonical),
                    "right_hash": stable_hash(serial_canonical),
                },
                "low_budget_vs_full": {
                    "equal": budget_canonical == full_canonical,
                    "left_hash": stable_hash(budget_canonical),
                    "right_hash": stable_hash(full_canonical),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(stale_probe)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0 if verification["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
