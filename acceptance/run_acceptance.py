"""End-to-end acceptance driver for the four mandated verification scenarios.

It regenerates the deterministic demo workspace and verifies:

1. incremental edit sequence vs single full rebuild are field-for-field equal;
2. feature-point budget pending work drains across batches and snapshots resume
   after a simulated interruption without corrupting the manifest;
3. serial (threads=1) and concurrent execution produce identical reports;
4. injected input faults are isolated, mapped to business codes and reported
   with their injection ids.

All run outputs are written under ``acceptance/runs/<timestamp-free-id>/``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlimgedit.config import load_config_dict, load_config
from mlimgedit.datagen import generate_dataset
from mlimgedit.features import feature_point_cost
from mlimgedit.injection import load_injections
from mlimgedit.plan import EditPlan
from mlimgedit.pipeline import Pipeline
from mlimgedit.runner import (
    comparable_state,
    full_run,
    run_plan,
    rebuild_final_state,
    write_report,
)
from mlimgedit.snapshot import SnapshotStore
from scripts.bootstrap_demo import (
    build_config,
    build_edit_plan,
    build_injections,
    SAMPLE_COUNT,
)


def _prepare_demo(base: Path) -> tuple[Path, EditPlan, dict]:
    demo = base / "demo"
    if demo.exists():
        shutil.rmtree(demo)
    demo.mkdir(parents=True)
    config_dict = build_config(threads=4, budget=50000)
    (demo / "config.json").write_text(
        json.dumps(config_dict, sort_keys=True, indent=2), encoding="utf-8"
    )
    (demo / "edit_plan.json").write_text(
        json.dumps(build_edit_plan(), sort_keys=True, indent=2), encoding="utf-8"
    )
    (demo / "injections.json").write_text(
        json.dumps(build_injections(), sort_keys=True, indent=2), encoding="utf-8"
    )
    config = load_config_dict(config_dict)
    generate_dataset(demo / "samples", config, SAMPLE_COUNT)
    return demo, EditPlan.load(demo / "edit_plan.json"), config_dict


def scenario_incremental_vs_full(base: Path, demo: Path, plan: EditPlan) -> dict:
    out = base / "incremental_vs_full"
    reports = run_plan(plan, out / "work", out / "inc_store", reports_dir=out / "reports")
    final_config, final_samples = rebuild_final_state(plan, out / "final")
    full_report = full_run(final_config, final_samples, out / "full_store")
    write_report(out / "reports" / "report_full_rebuild.json", full_report)
    equal = comparable_state(reports[-1]) == comparable_state(full_report)
    result = {
        "scenario": "incremental_vs_full",
        "equal": equal,
        "batches": len(reports),
        "final_samples": len(full_report["samples"]),
        "final_reuse": reports[-1]["counts"]["reuse"],
        "final_recompute": reports[-1]["counts"]["recompute"],
    }
    write_report(out / "result.json", result)
    return result


def scenario_budget_and_snapshot(base: Path, demo: Path, plan: EditPlan) -> dict:
    out = base / "budget_snapshot"
    config = load_config(demo / "config.json")
    cost = feature_point_cost(config)
    tight = config.with_updates(
        {"runtime": {"threads": 1, "feature_point_budget": cost * 3 + 1}}
    )
    sample_dir = demo / "samples"

    store = SnapshotStore(out / "store")
    pipeline = Pipeline(store)
    pending_seen = False
    last_report = None
    batch_index = 0
    while True:
        report = pipeline.run_batch(tight, sample_dir, batch_index)
        last_report = report
        if report["counts"]["pending"]:
            pending_seen = True
        if not report["downstream_blocked_by_budget"]:
            break
        batch_index += 1
        if batch_index > 10:
            raise RuntimeError("budget pending queue did not drain")

    # Snapshot interruption: start a second store and interrupt one batch.
    interrupt_store = SnapshotStore(out / "interrupt_store")
    interrupt_pipeline = Pipeline(interrupt_store)

    def interrupt_once(batch_index, stage):
        if batch_index == 1 and stage == "scoring":
            interrupt_pipeline.interruption = None
            raise KeyboardInterrupt("simulated SIGINT during acceptance run")

    interrupt_pipeline.interruption = interrupt_once
    interrupted = False
    try:
        for index in range(len(plan.batches)):
            interrupt_pipeline.run_batch(config, sample_dir, index)
    except KeyboardInterrupt:
        interrupted = True
    cursor_after = interrupt_store.load_manifest()["batch_cursor"]
    for index in range(cursor_after, len(plan.batches)):
        interrupt_pipeline.run_batch(config, sample_dir, index)
    cursor_final = interrupt_store.load_manifest()["batch_cursor"]

    result = {
        "scenario": "budget_and_snapshot",
        "pending_observed": pending_seen,
        "drain_batches": batch_index + 1,
        "final_pending": last_report["counts"]["pending"],
        "interruption_raised": interrupted,
        "cursor_after_interrupt": cursor_after,
        "cursor_after_resume": cursor_final,
        "deduction_sequences": last_report["budget"]["deductions"][:3],
        "recovered": cursor_final == len(plan.batches),
    }
    write_report(out / "result.json", result)
    return result


def scenario_concurrency(base: Path, demo: Path) -> dict:
    out = base / "concurrency"
    config = load_config(demo / "config.json")
    sample_dir = demo / "samples"
    serial_config = config.with_updates({"runtime": {"threads": 1}})
    parallel_config = config.with_updates({"runtime": {"threads": 16}})
    serial = full_run(serial_config, sample_dir, out / "serial_store")
    parallel = full_run(parallel_config, sample_dir, out / "parallel_store")
    equal = comparable_state(serial) == comparable_state(parallel)
    result = {
        "scenario": "concurrency_serial_equivalence",
        "equal": equal,
        "serial_threads": 1,
        "parallel_threads": parallel_config.threads,
        "serial_counts": serial["counts"],
        "parallel_counts": parallel["counts"],
    }
    write_report(out / "result.json", result)
    return result


def scenario_fault_isolation(base: Path, demo: Path) -> dict:
    out = base / "fault_isolation"
    config = load_config(demo / "config.json")
    injections = load_injections(demo / "injections.json")
    report = full_run(
        config, demo / "samples", out / "store", injections=injections
    )
    write_report(out / "report.json", report)
    hits = {
        failure["injection_id"]: failure["code"]
        for failure in report["failures"]
        if failure["injection_id"]
    }
    expected = {
        "INJ-001": "E_SAMPLE_FORMAT",
        "INJ-002": "E_LABEL_UNKNOWN",
        "INJ-003": "E_BBOX_OUT_OF_RANGE",
        "INJ-004": "E_SAMPLE_FORMAT",
    }
    result = {
        "scenario": "fault_injection_isolation",
        "expected": expected,
        "observed": hits,
        "matched": hits == expected,
        "valid_samples_with_predictions": len(report["per_sample_predictions"]),
        "failures": report["failures"],
    }
    write_report(out / "result.json", result)
    return result


def main() -> int:
    base = ROOT / "acceptance" / "runs" / "acceptance"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    demo, plan, _ = _prepare_demo(base)
    results = [
        scenario_incremental_vs_full(base, demo, plan),
        scenario_budget_and_snapshot(base, demo, plan),
        scenario_concurrency(base, demo),
        scenario_fault_isolation(base, demo),
    ]
    summary = {
        "all_passed": all(
            item.get("equal", True)
            and item.get("matched", True)
            and item.get("recovered", True)
            and item.get("pending_observed", True)
            for item in results
        ),
        "results": results,
    }
    write_report(base / "acceptance_summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
