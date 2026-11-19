#!/usr/bin/env python3
"""End-to-end acceptance driver for the deterministic segmentation component.

It exercises the acceptance criteria on the local deterministic sample set and
writes a machine-readable diagnostic report:

* full rebuild vs. incremental chain pixel equality across the edit sequence,
* budgeted multi-batch convergence and E_RECOMPUTE_BUDGET_PENDING markers,
* injection fault ids (truncated PNG, non-finite band, asymmetric adjacency),
* atomic snapshot recovery (partial temp directories are ignored),
* concurrent vs. single-thread identical outputs and ordering.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from rsseg.config import SegmentationConfig, load_config
from rsseg.edits import apply_edit, copy_sample_set
from rsseg.engine import IncrementalEngine
from rsseg.errors import ErrorCode
from rsseg.runner import build_inputs, validate_index_or_report
from rsseg.snapshot import SnapshotManager


def _grid(work: Path) -> np.ndarray:
    return np.load(SnapshotManager(work).latest().path / "global_grid.npy")


def run_acceptance(sample_dir: Path, out_dir: Path) -> dict:
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"checks": [], "injection_hits": []}

    def record(name: str, ok: bool, detail: dict) -> None:
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    # 1. Full rebuild baseline.
    full_sources = out_dir / "full_sources"
    full_work = out_dir / "full_work"
    copy_sample_set(sample_dir, full_sources)
    full_report = IncrementalEngine(build_inputs(full_sources, full_work)).run(force_full=True)
    full_grid = _grid(full_work)
    record("full_rebuild", full_report.snapshot_version == 1,
           {"tiles": len(full_report.recomputed_tiles),
            "fusions": len(full_report.recomputed_fusions),
            "grid_shape": list(full_grid.shape),
            "classes": np.unique(full_grid).tolist()})

    # 2. Incremental chain through the complete edit sequence vs. clean rebuilds.
    inc_sources = out_dir / "inc_sources"
    inc_work = out_dir / "inc_work"
    copy_sample_set(sample_dir, inc_sources)
    IncrementalEngine(build_inputs(inc_sources, inc_work)).run()
    sequence = json.loads((sample_dir / "edit_sequence.json").read_text(encoding="utf-8"))
    step_results = []
    for step in sequence["steps"]:
        apply_edit(inc_sources, step)
        kwargs = {"patches_dir": inc_sources / "patches"
                  if (inc_sources / "patches").exists() else None}
        IncrementalEngine(build_inputs(inc_sources, inc_work, **kwargs)).run()
        clean_sources = out_dir / f"clean_{step['id']}"
        clean_work = out_dir / f"clean_work_{step['id']}"
        copy_sample_set(inc_sources, clean_sources)
        clean_kwargs = {"patches_dir": clean_sources / "patches"
                        if (clean_sources / "patches").exists() else None}
        IncrementalEngine(build_inputs(clean_sources, clean_work, **clean_kwargs)).run(force_full=True)
        equal = np.array_equal(_grid(inc_work), _grid(clean_work))
        step_results.append({"step": step["id"], "pixel_equal": bool(equal)})
    record("edit_sequence_pixel_equality", all(item["pixel_equal"] for item in step_results),
           {"steps": step_results})

    # 3. Budgeted convergence.
    budget_sources = out_dir / "budget_sources"
    budget_work = out_dir / "budget_work"
    copy_sample_set(sample_dir, budget_sources)
    base_cfg = load_config(budget_sources / "segmentation_config.json").to_dict()
    base_cfg["recompute_budget"] = 2
    cfg = SegmentationConfig.from_dict(base_cfg)
    rounds = []
    pending = True
    while pending:
        run = IncrementalEngine(build_inputs(budget_sources, budget_work,
                                             config_override=cfg.to_dict())).run()
        pending = bool(run.pending_tiles)
        rounds.append({"recomputed": [list(k) for k in run.recomputed_tiles],
                       "pending": [list(k) for k in run.pending_tiles],
                       "budget_pending_codes":
                           sorted({f.code for f in run.failures
                                   if f.code == ErrorCode.E_RECOMPUTE_BUDGET_PENDING})})
    budget_equal = np.array_equal(_grid(budget_work), full_grid)
    record("budgeted_convergence", budget_equal, {"rounds": rounds})

    # 4. Injection faults.
    injection_checks = []
    for injection in json.loads((sample_dir / "injections.json").read_text())["injections"]:
        if injection["kind"] == "asymmetric_adjacency":
            failure = validate_index_or_report(sample_dir, sample_dir / "injections.json")
            ok = bool(failure and failure.injection_id == injection["id"]
                      and failure.code == ErrorCode.E_INDEX_INVALID)
            injection_checks.append({"id": injection["id"], "ok": ok, "code":
                                     failure.code if failure else None})
            if failure:
                report["injection_hits"].append(failure.injection_id)
            continue
        sources = out_dir / f"inj_{injection['id']}"
        work = out_dir / f"inj_work_{injection['id']}"
        copy_sample_set(sample_dir, sources)
        plan = sources / "one_injection.json"
        plan.write_text(json.dumps({"injections": [injection]}), encoding="utf-8")
        run = IncrementalEngine(build_inputs(sources, work, injections_path=plan)).run()
        hit = next((f for f in run.failures if f.injection_id == injection["id"]), None)
        if injection["kind"] == "zero_budget":
            ok = bool(hit and hit.code == ErrorCode.E_RECOMPUTE_BUDGET_PENDING)
        else:
            ok = hit is not None
        injection_checks.append({"id": injection["id"], "ok": ok,
                                 "code": hit.code if hit else None})
        if hit:
            report["injection_hits"].append(hit.injection_id)
    record("injection_isolation", all(item["ok"] for item in injection_checks),
           {"injections": injection_checks})

    # 5. Snapshot recovery ignores partial temp directories.
    manager = SnapshotManager(full_work)
    fake = manager.snapshots_dir / "snapshot.tmp.crash_000999"
    fake.mkdir()
    (fake / "global_grid.npy").write_bytes(b"partial")
    recovered = manager.latest()
    record("snapshot_recovery", recovered is not None and recovered.version == 1
           and not fake.exists(), {"recovered_version": recovered.version if recovered else None})

    # 6. Concurrency equivalence.
    parallel_work = out_dir / "parallel_work"
    serial_work = out_dir / "serial_work"
    parallel = IncrementalEngine(build_inputs(
        full_sources, parallel_work, config_override={"workers": 4})).run(force_full=True)
    serial = IncrementalEngine(build_inputs(
        full_sources, serial_work, config_override={"workers": 1})).run(force_full=True)
    equal_grid = np.array_equal(_grid(parallel_work), _grid(serial_work))
    record("concurrency_equivalence",
           equal_grid and parallel.recomputed_tiles == serial.recomputed_tiles
           and parallel.failure_dicts() == serial.failure_dicts(),
           {"parallel_tiles": [list(k) for k in parallel.recomputed_tiles],
            "serial_tiles": [list(k) for k in serial.recomputed_tiles]})

    report["all_ok"] = all(item["ok"] for item in report["checks"])
    report_path = out_dir / "diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", default="data/sample_set")
    parser.add_argument("--out", default="data/acceptance")
    args = parser.parse_args()
    report = run_acceptance(Path(args.sample_dir), Path(args.out))
    print(json.dumps({"all_ok": report["all_ok"],
                      "checks": [(c["name"], c["ok"]) for c in report["checks"]],
                      "injection_hits": report["injection_hits"]}, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
