"""High-level entry points shared by the CLI, tests and acceptance scripts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import load_config
from .injection import load_injections
from .pipeline import Pipeline
from .plan import EditPlan, Workspace
from .snapshot import SnapshotStore, atomic_write_json

COMPARABLE_REPORT_KEYS = (
    "samples",
    "per_sample_predictions",
    "metrics_summary",
    "decision_records",
    "boundary_samples",
    "failures",
    "pending",
    "quantization",
    "feature_mode",
    "downstream_blocked_by_budget",
)


def write_report(path: str | Path, report: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, report)
    return path


def comparable_state(report: dict) -> dict:
    """Strip run-instance-only audit counters so two runs can be compared."""
    return {key: report.get(key) for key in COMPARABLE_REPORT_KEYS}


def fresh_store(store_path: str | Path) -> SnapshotStore:
    store_path = Path(store_path)
    if store_path.exists():
        shutil.rmtree(store_path)
    return SnapshotStore(store_path)


def full_run(
    config,
    sample_dir: str | Path,
    store_path: str | Path,
    injections=None,
    interruption=None,
) -> dict:
    """Run one full pass from an empty snapshot store."""
    store = fresh_store(store_path)
    pipeline = Pipeline(store, injections=injections, interruption=interruption)
    return pipeline.run_batch(config, sample_dir, batch_index=0)


def run_plan(
    plan: EditPlan,
    workdir: str | Path,
    store_path: str | Path,
    injections=None,
    *,
    fresh: bool = True,
    interruption=None,
    reports_dir: str | Path | None = None,
) -> list[dict]:
    """Replay every edit batch incrementally against one persistent store."""
    workdir = Path(workdir)
    if fresh:
        store = fresh_store(store_path)
    else:
        store = SnapshotStore(store_path)
    pipeline = Pipeline(store, injections=injections, interruption=interruption)
    base_config = load_config(plan.base_config_path)
    workspace = Workspace(workdir / "workspace", base_config, plan.base_sample_dir)
    reports: list[dict] = []
    for index, batch in enumerate(plan.batches):
        config = workspace.apply_batches(plan.batches, max_index=index)
        report = pipeline.run_batch(config, workspace.samples_dir, batch_index=index)
        report["batch_id"] = batch.batch_id
        reports.append(report)
        if reports_dir is not None:
            write_report(Path(reports_dir) / f"report_batch_{index:02d}.json", report)
    return reports


def rebuild_final_state(
    plan: EditPlan,
    workdir: str | Path,
) -> tuple:
    """Materialize the final sample set and config after all edits (full rebuild)."""
    workdir = Path(workdir)
    workspace = Workspace(
        workdir / "final_workspace",
        load_config(plan.base_config_path),
        plan.base_sample_dir,
    )
    config = workspace.apply_batches(plan.batches)
    return config, workspace.samples_dir


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_from_paths(
    config_path: str | Path,
    sample_dir: str | Path,
    store_path: str | Path,
    report_path: str | Path,
    injection_path: str | Path | None = None,
    interruption=None,
) -> dict:
    config = load_config(config_path)
    injections = load_injections(injection_path)
    report = full_run(
        config, sample_dir, store_path, injections=injections, interruption=interruption
    )
    write_report(report_path, report)
    return report
