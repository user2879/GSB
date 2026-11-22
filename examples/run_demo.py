"""Reproducible end-to-end demonstration using repository-relative paths."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corpus_workbench.demo_data import generate_demo_workspace  # noqa: E402
from corpus_workbench.pipeline import Workbench  # noqa: E402


def make_workbench(output: str, run_id: str, scheduler: str, dedup: str = "dedup.json") -> Workbench:
    return Workbench(
        data_path=ROOT / "data" / "corpus",
        balance_path=ROOT / "configs" / "balance.json",
        dedup_path=ROOT / "configs" / dedup,
        scheduler_path=ROOT / "configs" / scheduler,
        injection_path=ROOT / "configs" / "injection.json",
        output_root=ROOT / output,
        run_id=run_id,
    )


def summary(report: dict) -> dict:
    return {
        "status": report["status"],
        "complete": report["complete"],
        "version": report["global_version"],
        "samples": report.get("sampling", {}).get("sample_count"),
        "unique_sampled": report.get("documents", {}).get("sampled_unique_before_dedup"),
        "after_dedup": report.get("documents", {}).get("after_dedup"),
        "groups": report.get("dedup", {}).get("duplicate_group_count"),
        "consumed_points": report.get("budget", {}).get("consumed_points"),
        "pending": len(report.get("pending_queue", [])),
        "failures": report.get("failure_count"),
        "result_digest": report.get("deterministic_result_digest"),
    }


def main() -> int:
    generate_demo_workspace(ROOT)
    for directory in ("demo-serial", "demo-parallel", "demo-budget", "demo-stale"):
        shutil.rmtree(ROOT / "output" / directory, ignore_errors=True)

    serial = make_workbench("output/demo-serial", "serial", "scheduler_serial.json").run(reset=True)
    parallel = make_workbench("output/demo-parallel", "parallel", "scheduler_parallel.json").run(reset=True)
    assert serial["status"] == "complete" and parallel["status"] == "complete"
    assert serial["deterministic_result_digest"] == parallel["deterministic_result_digest"]

    budget = make_workbench("output/demo-budget", "budget", "scheduler_tight.json")
    interrupted = budget.run(reset=True)
    assert interrupted["status"] == "budget_exhausted"
    assert interrupted["pending_queue"]
    assert {"E_SOURCE_QUOTA_EXCEEDED", "E_GLOBAL_BUDGET_EXHAUSTED"} <= {
        failure["code"] for failure in interrupted["failures"]
    }

    resumed = make_workbench("output/demo-budget", "budget", "scheduler_expanded.json").run(resume=True)
    assert resumed["status"] == "complete"
    assert resumed["pending_queue"] == []

    stale_base = make_workbench("output/demo-stale", "stale", "scheduler.json")
    stale_base_report = stale_base.run(reset=True)
    assert stale_base_report["status"] == "complete"
    stale = make_workbench(
        "output/demo-stale",
        "stale",
        "scheduler.json",
        dedup="dedup_threshold_090.json",
    ).run(resume=True)
    assert stale["status"] == "stale"
    assert stale["failures"][0]["code"] == "E_SNAPSHOT_STALE"

    threshold = make_workbench(
        "output/demo-budget",
        "threshold-reuse",
        "scheduler.json",
        dedup="dedup_threshold_090.json",
    ).run(reset=True)
    signature_audit = threshold["cache_audit"]["signatures"]
    assert signature_audit["hits"] > 0
    assert signature_audit["misses"] == 0
    assert threshold["budget"]["consumed_points"] == 0

    result = {
        "serial": summary(serial),
        "parallel": summary(parallel),
        "budget_first": summary(interrupted),
        "budget_resumed": summary(resumed),
        "stale_resume": summary(stale),
        "threshold_reuse": {
            **summary(threshold),
            "signature_hits": signature_audit["hits"],
            "signature_misses": signature_audit["misses"],
            "reused_sources": signature_audit["reused_sources"],
        },
        "reports": {
            "serial": "output/demo-serial/reports/serial.json",
            "parallel": "output/demo-parallel/reports/parallel.json",
            "budget": "output/demo-budget/reports/budget.json",
            "threshold": "output/demo-budget/reports/threshold-reuse.json",
            "stale": "output/demo-stale/reports/stale.json",
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
