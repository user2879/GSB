#!/usr/bin/env python3
"""End-to-end demonstration and acceptance driver.

Generates deterministic local samples, runs full rebuilds, an incremental edit
sequence (texture replacement, layer frequency/amplitude, blend mode, bit
width and rounding mode), a budget-exhaustion case and a snapshot-resume case.
All heavy assertions live in the unittest suite; this script emits a compact
machine-readable summary plus a Markdown report.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from texmat_pipeline.models import CHANNELS, load_generation_config, load_materials  # noqa: E402
from texmat_pipeline.pipeline import TexturePipeline  # noqa: E402


def generate(out: Path, seed: int, tiers, layers, budget, frac_bits=8, round_mode="half_even",
             blend_mode="normalized_weight") -> Path:
    cmd = [
        sys.executable, str(ROOT / "tools" / "generate_samples.py"),
        "--out", str(out), "--seed", str(seed),
        "--tiers", *[str(t) for t in tiers], "--layers", str(layers),
        "--threads", "4", "--point-budget", str(budget),
        "--frac-bits", str(frac_bits), "--round-mode", round_mode, "--blend-mode", blend_mode,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def edit_texture(sample: Path, resolution: int) -> None:
    target = sample / f"stone_{resolution}.png"
    img = Image.open(target)
    arr = np.asarray(img).copy()
    yy, xx = np.mgrid[0: arr.shape[0], 0: arr.shape[1]]
    delta = ((xx * 7 + yy * 13) % 11).astype(np.uint8)
    Image.fromarray(np.clip(arr.astype(np.uint16) + delta, 0, 255).astype(np.uint8),
                    mode=img.mode).save(target, format="PNG", optimize=True)


def edit_config(sample: Path, mutate) -> None:
    path = sample / "generation_config.json"
    cfg = json.loads(path.read_text())
    mutate(cfg)
    path.write_text(json.dumps(cfg, indent=2))


def state_digest(result) -> dict:
    return {
        "noise_fingerprints": [e["fingerprint"] for e in result.noise_result["layers"]],
        "composite_keys": result.composite_keys,
        "channel_means": {
            f"{m['material_id']}:{c}": float(
                m["channels"][c]["image"].astype(np.float64).mean()
                / (1 << result.specs_meta["frac_bits"])
            )
            for m in result.material_result["materials"] for c in CHANNELS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(ROOT / "outputs" / "demo"))
    parser.add_argument("--resolution", type=int, default=32)
    args = parser.parse_args()

    ws = Path(args.workspace)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    summary: dict = {"steps": []}

    # 1. Full rebuild across multi-tier samples.
    sample = generate(ws / "sample", seed=20261121, tiers=[16, 32, 64], layers=4, budget=4096)
    config, _ = load_generation_config(sample / "generation_config.json")
    materials, _ = load_materials(sample / "materials.json")
    full_pipe = TexturePipeline(sample, ws / "full_work", config, materials)
    full = full_pipe.run(args.resolution)
    summary["steps"].append({
        "step": "full_rebuild",
        "recompute": len(full.recompute_keys),
        "reuse": len(full.reuse_keys),
        "failures": full.failure_dicts(),
        "scheduler_serial_equivalent": full.scheduler["serial_equivalent"],
    })

    # 2. Incremental edit sequence, each compared with a fresh full rebuild.
    # Seed the incremental workspace from the full rebuild's cache so that
    # unaffected artifacts are genuinely reused rather than recomputed cold.
    live = ws / "live_sample"
    shutil.copytree(sample, live)
    live_work = ws / "live_work"
    shutil.copytree(ws / "full_work", live_work, symlinks=True)
    edits = [
        ("texture_replace", lambda: edit_texture(live, args.resolution)),
        ("layer_frequency_amplitude", lambda: edit_config(
            live, lambda c: (c["layers"][1].__setitem__("frequency", c["layers"][1]["frequency"] + 2.5),
                             c["layers"][1].__setitem__("amplitude", round(c["layers"][1]["amplitude"] * 0.75, 4))))),
        ("blend_mode", lambda: edit_config(live, lambda c: c.__setitem__("blend_mode", "weighted_sum"))),
        ("frac_bits", lambda: edit_config(
            live, lambda c: c["fixed_point"].__setitem__("frac_bits", c["fixed_point"]["frac_bits"] + 2))),
        ("round_mode", lambda: edit_config(
            live, lambda c: c["fixed_point"].__setitem__("round_mode", "half_up"))),
    ]
    for name, edit in edits:
        edit()
        cfg2, _ = load_generation_config(live / "generation_config.json")
        mats2, _ = load_materials(live / "materials.json")
        incremental = TexturePipeline(live, live_work, cfg2, mats2).run(args.resolution)
        fresh_sample = ws / f"fresh_{name}"
        shutil.copytree(live, fresh_sample)
        fresh = TexturePipeline(fresh_sample, ws / f"fresh_work_{name}", cfg2, mats2).run(
            args.resolution, force_rebuild=True
        )
        equal = state_digest(incremental) == state_digest(fresh)
        summary["steps"].append({
            "step": name,
            "incremental_reuse": len(incremental.reuse_keys),
            "incremental_recompute": len(incremental.recompute_keys),
            "matches_full_rebuild": equal,
            "affected": {k: len(v) for k, v in incremental.affected_scope.items()},
        })

    # 3. Budget exhaustion sample.
    tight = generate(ws / "tight", seed=77, tiers=[16], layers=4, budget=128)
    tcfg, _ = load_generation_config(tight / "generation_config.json")
    tmats, _ = load_materials(tight / "materials.json")
    budget_result = TexturePipeline(tight, ws / "tight_work", tcfg, tmats).run(16)
    summary["steps"].append({
        "step": "point_budget",
        "pending": budget_result.scheduler["pending"],
        "failure_codes": sorted({f.code for f in budget_result.failures}),
        "serial_equivalent": budget_result.scheduler["serial_equivalent"],
    })

    # 4. Snapshot resume / stale detection.
    state = full_pipe.snapshots.inspect(config.config_hash, config.fixed_point.frac_bits)
    summary["snapshot"] = {
        "global_version": state.global_version,
        "stale": state.stale,
        "completed_batches": state.completed_batches,
    }
    summary["final_report"] = json.loads((ws / "full_work" / "report.json").read_text())

    out_json = ws / "demo_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
