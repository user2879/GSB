"""Reproducible acceptance experiment for incremental/full equality."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from . import codes
from .compare import compare_cache_arrays, compare_reports
from .pipeline import ScenePipeline, compact_report
from .sample import generate_sample
from .state import SceneState, StateManager
from .util import atomic_write_json, hash_value, read_json
from .validation import load_injections, load_project, materialize_injected_sample


def finalize_sample_from_edits(sample_dir: Path, out_dir: Path) -> Path:
    project = load_project(sample_dir)
    if project.config is None:
        raise RuntimeError("cannot finalize a sample with invalid config")
    if project.failures:
        raise RuntimeError("cannot finalize a sample with validation failures: " + "; ".join(item.message for item in project.failures))
    with tempfile.TemporaryDirectory(prefix="gsscene-finalize-") as temp:
        manager = StateManager(Path(temp))
        state = manager.initialize(project)
        for edit in project.edits:
            manager.apply_edit(state, edit)
        manager.save(state)
        state.write_final_sample(out_dir)
    return out_dir


def run_incremental_sequence(sample_dir: Path, work_dir: Path, *, max_steps: int | None = None) -> dict[str, Any]:
    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    pipeline = ScenePipeline(sample_dir, work_dir, replay_edits=True)
    baseline = pipeline.run(max_new_edits=0)
    steps: list[dict[str, Any]] = []
    total = len(pipeline.loaded.edits) if max_steps is None else min(max_steps, len(pipeline.loaded.edits))
    for index in range(total):
        started = time.perf_counter()
        report = pipeline.run(max_new_edits=1)
        steps.append(
            {
                "edit": pipeline.loaded.edits[index]["type"],
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "report": compact_report(report),
            }
        )
    final = pipeline.run(max_new_edits=0)
    return {"baseline": baseline, "steps": steps, "final": final, "work_dir": work_dir, "pipeline": pipeline}


def run_equivalence_experiment(
    sample_dir: Path,
    out_dir: Path,
    *,
    regenerate: bool = False,
    width: int = 64,
    height: int = 64,
    views: int = 6,
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if regenerate or not (sample_dir / "config.json").exists():
        generate_sample(sample_dir, width=width, height=height, views=views, force=True)

    incremental = run_incremental_sequence(sample_dir, out_dir / "work_incremental")
    final_sample = finalize_sample_from_edits(sample_dir, out_dir / "final_sample")
    full_pipeline = ScenePipeline(final_sample, out_dir / "work_full", replay_edits=False)
    full = full_pipeline.run(max_new_edits=0)
    serial_pipeline = ScenePipeline(final_sample, out_dir / "work_serial", replay_edits=False, worker_override=1)
    serial = serial_pipeline.run(max_new_edits=0)

    incremental_full = compare_reports(incremental["final"], full)
    concurrent_serial = compare_reports(full, serial)
    array_incremental_full = compare_cache_arrays(out_dir / "work_incremental", out_dir / "work_full")
    array_concurrent_serial = compare_cache_arrays(out_dir / "work_full", out_dir / "work_serial")
    noop_reuse = incremental["final"]["actual_invalidation"]

    budget_report = run_budget_experiment(out_dir)
    injection_report = run_injection_experiment(sample_dir, out_dir)

    comparison = {
        "concurrent_equals_serial": concurrent_serial,
        "concurrent_serial_arrays_equal": array_concurrent_serial,
        "concurrent_serial_invalidation_equal": full["actual_invalidation"] == serial["actual_invalidation"],
        "concurrent_serial_failure_order_equal": full["failures"] == serial["failures"],
        "incremental_equals_full": incremental_full,
        "incremental_full_arrays_equal": array_incremental_full,
        "invalidation_reuse_after_stable_run": {
            "recompute_kernels": len(noop_reuse.get("recompute_kernels", [])),
            "recompute_tiles": len(noop_reuse.get("recompute_tiles", [])),
            "recompute_composites": len(noop_reuse.get("recompute_composites", [])),
            "reused": noop_reuse.get("reused", 0),
        },
        "budget_codes": budget_report["observed_codes"],
        "injection_codes": sorted(injection_report["observed_codes"]),
        "injection_hits": injection_report["all_hits"],
        "ok": (
            incremental_full["equal"]
            and concurrent_serial["equal"]
            and array_incremental_full["equal"]
            and array_concurrent_serial["equal"]
            and full["actual_invalidation"] == serial["actual_invalidation"]
            and not noop_reuse.get("recompute_kernels")
            and not noop_reuse.get("recompute_tiles")
            and not noop_reuse.get("recompute_composites")
            and codes.E_KERNEL_BUDGET_EXHAUSTED in budget_report["observed_codes"]
            and codes.E_POINT_BUDGET_PENDING in budget_report["observed_codes"]
            and injection_report["all_hits"]
        ),
    }

    atomic_write_json(out_dir / "incremental_steps.json", [step for step in incremental["steps"]])
    atomic_write_json(out_dir / "incremental_final.json", compact_report(incremental["final"]))
    atomic_write_json(out_dir / "full_rebuild.json", compact_report(full))
    atomic_write_json(out_dir / "serial.json", compact_report(serial))
    atomic_write_json(out_dir / "comparison.json", comparison)
    atomic_write_json(out_dir / "budget.json", budget_report)
    atomic_write_json(out_dir / "injection.json", injection_report)
    atomic_write_json(
        out_dir / "run_metadata.json",
        {
            "incremental_cache": str((out_dir / "work_incremental" / "cache").relative_to(out_dir)),
            "final_sample": str(final_sample.relative_to(out_dir)),
            "full_cache": str((out_dir / "work_full" / "cache").relative_to(out_dir)),
            "serial_cache": str((out_dir / "work_serial" / "cache").relative_to(out_dir)),
        },
    )
    return comparison


def run_injection_experiment(sample_dir: Path, out_dir: Path) -> dict[str, Any]:
    injections, failures, _hits = load_injections(Path(sample_dir))
    if failures:
        return {"cases": [], "all_hits": False, "observed_codes": [], "errors": [item.as_dict() for item in failures]}
    cases: list[dict[str, Any]] = []
    observed_codes: set[str] = set()
    all_hits = True
    base = Path(out_dir) / "injections"
    if base.exists():
        shutil.rmtree(base)
    for index, injection in enumerate(injections):
        injection_id = str(injection["injection_id"])
        case_dir = base / f"case_{index:02d}_{injection_id.replace('-', '_')}"
        sample = materialize_injected_sample(sample_dir, case_dir / "sample", selected_injection_ids={injection_id})
        pipeline = ScenePipeline(sample, case_dir / "work", replay_edits=False)
        report = pipeline.run(max_new_edits=0)
        actual = [item for item in report.get("failures", []) if item.get("injection_id") == injection_id]
        actual_codes = sorted({item["code"] for item in actual})
        observed_codes.update(actual_codes)
        expected = codes.INJECTION_CODE_MAP[injection["kind"]]
        hit = expected in actual_codes
        all_hits = all_hits and hit
        cases.append(
            {
                "actual_codes": actual_codes,
                "diagnostics_count": len(report.get("diagnostics", [])),
                "excluded_levels": report.get("excluded_levels", []),
                "excluded_views": report.get("excluded_views", []),
                "expected_code": expected,
                "hit": hit,
                "injection_id": injection_id,
                "kind": injection["kind"],
                "level": injection.get("level"),
                "view_index": injection.get("view_index"),
            }
        )
    return {
        "all_hits": all_hits,
        "cases": cases,
        "observed_codes": sorted(observed_codes),
    }


def run_budget_experiment(out_dir: Path) -> dict[str, Any]:
    base = Path(out_dir) / "budget"
    if base.exists():
        shutil.rmtree(base)
    sample = base / "sample"
    generate_sample(sample, width=16, height=16, views=2, force=True)
    config_path = sample / "config.json"
    config = read_json(config_path)
    config["kernel_budgets"] = [100, 2, 1]
    config["point_budget"] = 5
    config["snapshot_interval"] = 1
    atomic_write_json(config_path, config)
    pipeline = ScenePipeline(sample, base / "work", replay_edits=False)
    report = pipeline.run(max_new_edits=0)
    observed_codes = sorted({item["code"] for item in report.get("failures", [])})
    statuses = [
        {
            "allowed_kernel": entry.get("allowed_kernel"),
            "level": entry.get("level"),
            "pending_count": entry.get("pending_count"),
            "status": entry.get("status"),
            "view_index": entry.get("view_index"),
        }
        for entry in report.get("budget_ledger", {}).get("kernel_scheduler", {}).get("entries", [])
    ]
    return {
        "ledger_entries": statuses,
        "observed_codes": observed_codes,
        "pending": report.get("pending", []),
        "report_digest": hash_value(compact_report(report)),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the GS scene incremental/full equivalence experiment")
    parser.add_argument("--sample", type=Path, default=Path("artifacts/run/sample"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/run/experiment"))
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--views", type=int, default=6)
    args = parser.parse_args(argv)
    result = run_equivalence_experiment(
        args.sample,
        args.out,
        regenerate=args.regenerate,
        width=args.width,
        height=args.height,
        views=args.views,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
