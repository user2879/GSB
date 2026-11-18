from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import shutil
import sqlite3
from typing import Any

import numpy as np

from .database import write_json_atomic
from .ledger import export_ledger
from .manifest import load_manifest
from .migrations import MigrationRunner, perturb_scale
from .replay import replay_snapshot
from .sample import build_sample_database
from .snapshot import SnapshotManager, restore_snapshot
from .verify import verify_drift


DRIFT_COLUMNS = [
    "discount_rate",
    "net_cents",
    "tax_cents",
    "total_cents",
    "unit_price_cents",
]


def _drift_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "conclusion": report["conclusion"],
        "compared_rows": report["compared_rows"],
        "row_set_mismatch_count": report["row_set_mismatch_count"],
        "drift_columns": report["drift_columns"],
        "report_digest": report["report_digest"],
        "columns": [
            {
                "column": column["column"],
                "comparison_scale": column["comparison_scale"],
                "nonzero_deviation_rows": column["nonzero_deviation_rows"],
                "out_of_bound_rows": column["out_of_bound_rows"],
                "max_abs_diff_units": column["max_abs_diff_units"],
                "max_comparison_bound_units": column["max_comparison_bound_units"],
                "first_nonzero": column["first_nonzero"],
                "first_out_of_bound": column["first_out_of_bound"],
            }
            for column in report["columns"]
        ],
    }


def _step_summary(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for step in ledger["steps"]:
        result.append(
            {
                "step_id": step["step_id"],
                "version": step["version"],
                "operation": step["operation"],
                "table": step["table"],
                "row_count": step["row_count"],
                "content_digest": step["content_digest"],
                "chain_digest": step["chain_digest"],
                "column_errors": [
                    {
                        "column": column["column"],
                        "max_error_bound_units": column["max_error_bound_units"],
                        "representative_row_key": column["representative_row_key"],
                        "source_paths": column["source_paths"],
                    }
                    for column in step["result"]["column_errors"]
                ],
            }
        )
    return result


def run_demo(root: Path | None = None) -> dict[str, Any]:
    project_root = (root or Path(__file__).resolve().parents[2]).resolve()
    demo_root = project_root / "artifacts" / "demo"
    if demo_root.exists():
        shutil.rmtree(demo_root)
    for directory in ("ledger", "snapshots", "work"):
        (demo_root / directory).mkdir(parents=True, exist_ok=True)
    base_db = demo_root / "sample_base.sqlite3"
    working_db = demo_root / "orders.sqlite3"
    manifest_path = project_root / "data" / "migrations" / "orders_migration.json"
    sample = build_sample_database(base_db, seed=20261118, rows=48, overwrite=True)
    shutil.copy2(base_db, working_db)
    manifest = load_manifest(manifest_path, working_db)
    runner = MigrationRunner(
        working_db,
        manifest,
        ledger_dir=demo_root / "ledger",
        batch_size=7,
    )
    initial_status = runner.status()
    run_result = runner.run()
    drift_initial = verify_drift(
        base_db,
        working_db,
        table="orders",
        columns=DRIFT_COLUMNS,
        before_calibrations=run_result.manifest,
        row_key=["id"],
        batch_size=7,
        raise_on_drift=False,
    )
    verification_config = {
        "table": "orders",
        "columns": DRIFT_COLUMNS,
        "row_key": ["id"],
        "batch_size": 7,
    }
    snapshot_manager = SnapshotManager(
        working_db,
        run_result.manifest,
        demo_root / "snapshots",
        base_db_path=base_db,
        verification_config=verification_config,
    )
    original_snapshot = snapshot_manager.save(
        name="session-original.json",
        conclusions={"drift": drift_initial},
        base_db_path=base_db,
    )
    recalibration = perturb_scale(
        working_db,
        run_result.manifest,
        table="orders",
        column="discount_rate",
        new_scale=2,
        new_truncation="toward_zero",
        batch_size=7,
    )
    drift_perturbed = verify_drift(
        base_db,
        working_db,
        table="orders",
        columns=DRIFT_COLUMNS,
        before_calibrations=run_result.manifest,
        row_key=["id"],
        batch_size=7,
        raise_on_drift=False,
    )
    perturbed_snapshot = snapshot_manager.save(
        name="session-perturbed.json",
        conclusions={"drift": drift_perturbed},
        base_db_path=base_db,
    )
    restored_stale = restore_snapshot(
        demo_root / "snapshots" / "session-original.json",
        working_db,
        ledger_dir=demo_root / "ledger",
    )
    replay_original = replay_snapshot(
        demo_root / "snapshots" / "session-original.json",
        base_db,
        demo_root / "work" / "replay-original",
        batch_size=7,
    )
    replay_perturbed = replay_snapshot(
        demo_root / "snapshots" / "session-perturbed.json",
        base_db,
        demo_root / "work" / "replay-perturbed",
        batch_size=7,
    )
    report = {
        "environment": {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "numpy": np.__version__,
            "platform": platform.system(),
        },
        "sample": {
            "seed": sample["seed"],
            "rows": sample["rows"],
            "semantic_digest": sample["semantic_digest"],
            "table_content_digest": sample["table_content_digest"],
        },
        "manifest": {
            "manifest_id": run_result.manifest.manifest_id,
            "sha256": run_result.manifest.sha256,
            "execution_order": list(run_result.manifest.execution_order),
            "derivation_edges": [
                list(edge) for edge in run_result.manifest.derivation_edges
            ],
            "derivation_order": list(run_result.manifest.derivation_order),
        },
        "initial_status": initial_status,
        "migration": {
            "complete": run_result.complete,
            "status": run_result.status,
            "committed_steps": run_result.committed_steps,
            "unknown_steps": run_result.unknown_steps,
            "steps": _step_summary(run_result.ledger),
        },
        "drift_initial": _drift_summary(drift_initial),
        "snapshot_original": {
            "session_version": original_snapshot["session_version"],
            "ledger_version": original_snapshot["ledger_version"],
            "calibration_version": original_snapshot["calibration_version"],
            "conclusion_version": original_snapshot["conclusion_version"],
        },
        "scale_perturbation": {
            "event_id": recalibration.event_id,
            "table": recalibration.table,
            "column": recalibration.column,
            "old_scale": recalibration.old_scale,
            "new_scale": recalibration.new_scale,
            "affected_columns": recalibration.affected_columns,
            "affected_steps": recalibration.affected_steps,
            "backfills": [
                {
                    "target": item["target"],
                    "scale": item["scale"],
                    "rows_evaluated": item["rows_evaluated"],
                    "max_error_bound_units": item["max_error_bound_units"],
                }
                for item in recalibration.report["backfills"]
            ],
            "drift": _drift_summary(drift_perturbed),
        },
        "snapshot_perturbed": {
            "session_version": perturbed_snapshot["session_version"],
            "ledger_version": perturbed_snapshot["ledger_version"],
            "calibration_version": perturbed_snapshot["calibration_version"],
            "conclusion_version": perturbed_snapshot["conclusion_version"],
        },
        "snapshot_restore": {
            "restored": restored_stale["restored"],
            "replaced": restored_stale["replaced"],
            "merge_policy": restored_stale.get("merge_policy"),
            "version_differences": [
                item["version"] for item in restored_stale["version_differences"]
            ],
            "affected_columns": restored_stale["affected_columns"],
            "affected_steps": restored_stale["affected_steps"],
            "old_conclusions_retained": bool(
                restored_stale["session_state"].get("conclusions")
            ),
        },
        "replay_original": {
            "reproducible": replay_original["reproducible"],
            "steps_compared": replay_original["steps_compared"],
            "replayed_data_version": replay_original["replayed_data_version"],
            "replayed_ledger_version": replay_original["replayed_ledger_version"],
        },
        "replay_perturbed": {
            "reproducible": replay_perturbed["reproducible"],
            "steps_compared": replay_perturbed["steps_compared"],
            "replayed_data_version": replay_perturbed["replayed_data_version"],
            "replayed_ledger_version": replay_perturbed["replayed_ledger_version"],
        },
    }
    write_json_atomic(demo_root / "report.json", report)
    return report


def _print_summary(report: dict[str, Any], root: Path) -> None:
    print("dbml deterministic migration demo")
    print(
        f"environment: Python {report['environment']['python']}, "
        f"SQLite {report['environment']['sqlite']}, NumPy {report['environment']['numpy']}"
    )
    print(f"sample: {report['sample']['rows']} rows, seed {report['sample']['seed']}")
    print("migration ledger:")
    for step in report["migration"]["steps"]:
        bounds = ", ".join(
            f"{item['column']}={item['max_error_bound_units']}"
            for item in step["column_errors"]
            if item["max_error_bound_units"] > 0
        )
        print(
            f"  v{step['version']} {step['step_id']} {step['operation']} "
            f"rows={step['row_count']} digest={step['content_digest'][:12]} "
            f"bounds=[{bounds}]"
        )
    print(
        f"initial drift: {report['drift_initial']['conclusion']} "
        f"({report['drift_initial']['compared_rows']} rows)"
    )
    perturbation = report["scale_perturbation"]
    print(
        f"scale perturbation: {perturbation['table']}.{perturbation['column']} "
        f"{perturbation['old_scale']}->{perturbation['new_scale']}; "
        f"affected steps={perturbation['affected_steps']}"
    )
    print(
        f"perturbed drift: {perturbation['drift']['conclusion']}; "
        f"max diff units={max(item['max_abs_diff_units'] for item in perturbation['drift']['columns'])}"
    )
    restore = report["snapshot_restore"]
    print(
        f"snapshot restore: replaced={restore['replaced']} "
        f"columns={restore['affected_columns']} steps={restore['affected_steps']}"
    )
    print(
        f"replay: original={report['replay_original']['reproducible']} "
        f"perturbed={report['replay_perturbed']['reproducible']}"
    )
    print(f"report: {root / 'artifacts' / 'demo' / 'report.json'}")
    print(f"ledger: {root / 'artifacts' / 'demo' / 'ledger' / 'ledger.json'}")
    print(f"snapshots: {root / 'artifacts' / 'demo' / 'snapshots'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic dbml demo")
    parser.add_argument("--root", default=None, help="project root")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]
    report = run_demo(root)
    _print_summary(report, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
