"""End-to-end local verification driver (no network, stdlib + numpy + Pillow).

It runs, in a fresh temp workspace:
  1. deterministic sample generation and a full render;
  2. fault injection (corrupt PNG + invalid config) and failure isolation;
  3. the edit sequence, comparing incremental results with a from-scratch run;
  4. concurrency determinism (workers 1 vs 8) and budget-resume completion.

A machine-readable summary is written next to the chosen output directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_samples import write_bundle  # noqa: E402
from texmat.testkit import (apply_edit, materialize_full_reference,  # noqa: E402
                            run_incremental, snapshot_arrays)


def _cli(sample: Path, snapshot: Path, report: Path, extra=None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "texmat", "run",
           "--sample-dir", str(sample), "--snapshot-dir", str(snapshot),
           "--report", str(report)] + (extra or [])
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)


def verify(workdir: Path) -> dict:
    summary: dict = {"checks": []}

    def record(name: str, ok: bool, detail: str = "") -> None:
        summary["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    sample = workdir / "sample"
    write_bundle(sample, size=24, workers=4, budget=100000)

    # 1. full render
    snap = workdir / "snap_full"
    report = workdir / "report_full.json"
    proc = _cli(sample, snap, report)
    payload = json.loads(report.read_text())
    record("full_render_completes", proc.returncode == 0 and payload["status"] == "complete",
           f"exit={proc.returncode}, status={payload['status']}")
    record("report_has_four_stage_attribution",
           set(payload["error_attribution"]["stages"]) ==
           {"noise_overlay", "quantize", "material_blend", "channel_compose"})

    # 2a. corrupt texture isolation
    (sample / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    report_bad = workdir / "report_bad_texture.json"
    proc = _cli(sample, workdir / "snap_bad", report_bad)
    bad_payload = json.loads(report_bad.read_text())
    codes = {item["code"] for item in bad_payload["failures"]}
    record("corrupt_texture_isolated", "E_TEXTURE_DECODE" in codes and bad_payload["status"] == "complete",
           f"codes={sorted(codes)}")
    (sample / "broken.png").unlink()

    # 2b. invalid config
    bad_cfg_dir = workdir / "bad_cfg"
    shutil.copytree(sample, bad_cfg_dir)
    cfg = json.loads((bad_cfg_dir / "generation_config.json").read_text())
    cfg["resolution_tiers"] = [64, 24]
    (bad_cfg_dir / "generation_config.json").write_text(json.dumps(cfg, sort_keys=True))
    proc = _cli(bad_cfg_dir, workdir / "snap_cfg", workdir / "report_cfg.json")
    cfg_payload = json.loads((workdir / "report_cfg.json").read_text())
    record("invalid_config_flagged", proc.returncode != 0 and cfg_payload.get("code") == "E_CONFIG_INVALID",
           f"exit={proc.returncode}")

    # 3. edit sequence incremental vs from-scratch
    inc_sample = workdir / "inc_sample"
    shutil.copytree(sample, inc_sample)
    (inc_sample / "broken.png").unlink(missing_ok=True)
    inc_snap = workdir / "snap_inc"
    edits = json.loads((inc_sample / "edit_sequence.json").read_text())
    for edit in edits:
        apply_edit(inc_sample, edit)
        run_incremental(inc_sample, inc_snap)
    full_snap = workdir / "snap_reference"
    materialize_full_reference(inc_sample, full_snap)
    inc_arrays = snapshot_arrays(inc_snap)
    full_arrays = snapshot_arrays(full_snap)
    equal = set(inc_arrays) == set(full_arrays) and all(
        np.array_equal(inc_arrays[k], full_arrays[k]) for k in inc_arrays)
    record("incremental_equals_full_rebuild", equal,
           f"payloads={len(inc_arrays)}")

    # 4. concurrency determinism
    base = None
    concurrency_ok = True
    for workers in (1, 8):
        wdir = workdir / f"workers_{workers}"
        engine_sample = workdir / f"wsample_{workers}"
        if engine_sample.exists():
            shutil.rmtree(engine_sample)
        shutil.copytree(inc_sample, engine_sample)
        from texmat.testkit import make_engine
        engine = make_engine(engine_sample, wdir)
        object.__setattr__(engine.config, "workers", workers)
        result = engine.execute_run()
        while not result.complete and not result.stopped_by_signal:
            engine = make_engine(engine_sample, wdir)
            object.__setattr__(engine.config, "workers", workers)
            result = engine.execute_run()
        arrays = snapshot_arrays(wdir)
        if base is None:
            base = arrays
        elif set(base) != set(arrays) or any(
                not np.array_equal(base[k], arrays[k]) for k in base):
            concurrency_ok = False
    record("worker_count_pixel_identical", concurrency_ok)

    # 5. tight budget resume reaches completion
    budget_sample = workdir / "budget_sample"
    shutil.copytree(inc_sample, budget_sample)
    bcfg = json.loads((budget_sample / "generation_config.json").read_text())
    bcfg["point_budget"] = 180
    (budget_sample / "generation_config.json").write_text(json.dumps(bcfg, sort_keys=True))
    proc = _cli(budget_sample, workdir / "snap_budget", workdir / "report_budget.json")
    bpayload = json.loads((workdir / "report_budget.json").read_text())
    record("budget_resume_completes",
           proc.returncode == 0 and bpayload["status"] == "complete" and bpayload["batches"] > 1,
           f"batches={bpayload['batches']}")

    summary["all_ok"] = all(item["ok"] for item in summary["checks"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="outputs/verification")
    args = parser.parse_args()
    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    summary = verify(workdir)
    out = workdir / "verification_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    for item in summary["checks"]:
        print(f"[{'PASS' if item['ok'] else 'FAIL'}] {item['name']} {item['detail']}")
    print("ALL_OK:", summary["all_ok"])
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
