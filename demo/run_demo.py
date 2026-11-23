"""End-to-end reproducible demonstration.

Runs without network access and writes every artefact under ``demo_out/``:

1. deterministic sample generation;
2. cold full run + warm cache run (must be pixel-identical);
3. a band patch -> only the touched tile and its fusion bands recompute;
4. a weight change -> all tiles invalidate, fusion fingerprints change;
5. fusion strategy switch -> fusion rules and raster change;
6. snapshot staleness via a global version bump;
7. budget enforcement (tiny global budget + small fusion batch);
8. fault injection suite and diagnostic report;
9. serial vs concurrent equivalence and incremental-vs-clean equivalence.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mstc.config import load_config
from mstc.injection import load_injections, run_injection_suite
from mstc.loader import load_manifest
from mstc.paths import to_path
from mstc.pipeline import Pipeline
from mstc.report import build_report, write_report
from mstc.samplegen import generate_sample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo_out"


def fresh_workspace(name: str) -> Path:
    ws = OUT / name
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    generate_sample(ws, 3, 3, edge=16, overlap=4)
    shutil.copy(ROOT / "configs" / "config.yaml", ws / "config.yaml")
    return ws


def run_pipeline(ws: Path, version: int = 1, config_name: str = "config.yaml"):
    config = load_config(ws / config_name)
    manifest = load_manifest(to_path(config.manifest_path, ws), ws)
    grid = (max(t.row for t in manifest.tiles) + 1, max(t.col for t in manifest.tiles) + 1)
    pipeline = Pipeline(config, ws, global_version=version)
    return pipeline.run(grid, manifest.tile_edge, manifest.overlap), config


def raster_sig(run):
    return np.array(run.global_raster.raster).tolist()


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    summary = {}

    # 1+2 cold then warm
    ws = fresh_workspace("main")
    cold, config = run_pipeline(ws)
    warm, _ = run_pipeline(ws)
    summary["cold_recomputed"] = cold.recomputed_tiles
    summary["warm_cached_tiles"] = warm.cached_tiles
    summary["cold_equals_warm"] = raster_sig(cold) == raster_sig(warm)

    # 3 band patch on t11
    band_path = to_path("samples\\tiles\\t11\\R.npy", ws)
    arr = np.load(band_path)
    arr[0, 0] = 255
    np.save(band_path, arr)
    patched, _ = run_pipeline(ws)
    summary["patch_invalidated_tiles"] = patched.invalidated_tiles
    summary["patch_recomputed_fusions"] = patched.recomputed_fusions

    # clean rebuild with identical final inputs
    clean_ws = fresh_workspace("clean")
    np.save(to_path("samples\\tiles\\t11\\R.npy", clean_ws), arr)
    clean_run, _ = run_pipeline(clean_ws)
    summary["incremental_equals_clean"] = raster_sig(patched) == raster_sig(clean_run)
    summary["tile_summary_equals_clean"] = all(
        patched.tile_results[t].score_summary == clean_run.tile_results[t].score_summary
        for t in patched.tile_results
    )

    # 4 weight change
    cfg_path = ws / "config.yaml"
    text = cfg_path.read_text()
    cfg_path.write_text(text.replace("vegetation: 1.0", "vegetation: 2.5"))
    weighted, _ = run_pipeline(ws)
    summary["weight_change_all_tiles_recomputed"] = len(weighted.recomputed_tiles) == 9
    summary["weight_raster_changed"] = raster_sig(weighted) != raster_sig(patched)
    cfg_path.write_text(text)

    # 5 fusion strategy switch
    fv2_path = ws / "config_fv2.yaml"
    fv2_path.write_text(text.replace("fusion_strategy_version: FV1", "fusion_strategy_version: FV2"))
    fv2_run, fv2_config = run_pipeline(ws, config_name="config_fv2.yaml")
    fv1_run, _ = run_pipeline(ws)
    summary["fusion_strategy_changes_raster"] = raster_sig(fv2_run) != raster_sig(fv1_run)

    # 6 snapshot staleness via version bump
    stale, _ = run_pipeline(ws, version=2)
    summary["snapshot_stale_detected"] = "E_SNAPSHOT_STALE" in stale.failures.codes()

    # 7a global pixel budget enforcement (each tile needs 256 px)
    pix_ws = fresh_workspace("pixel_budget")
    pix_cfg = (pix_ws / "config.yaml").read_text().replace(
        "global_pixel_budget: 100000", "global_pixel_budget: 600")
    (pix_ws / "config.yaml").write_text(pix_cfg)
    pix_run, _ = run_pipeline(pix_ws)
    summary["pixel_budget_codes"] = sorted(set(pix_run.failures.codes()))

    # 7b per-tile quota enforcement
    quota_ws = fresh_workspace("tile_quota")
    quota_cfg = (quota_ws / "config.yaml").read_text().replace(
        "tile_pixel_quota: 1000", "tile_pixel_quota: 100")
    (quota_ws / "config.yaml").write_text(quota_cfg)
    quota_run, _ = run_pipeline(quota_ws)
    summary["tile_quota_codes"] = sorted(set(quota_run.failures.codes()))

    # 7c fusion batch limit: all tiles finish, only 2 of 12 pairs per batch
    bud_ws = fresh_workspace("fusion_budget")
    bud_cfg = (bud_ws / "config.yaml").read_text().replace(
        "fusion_batch_limit: 32", "fusion_batch_limit: 2")
    (bud_ws / "config.yaml").write_text(bud_cfg)
    bud_run, _ = run_pipeline(bud_ws)
    summary["fusion_budget_codes"] = sorted(set(bud_run.failures.codes()))
    summary["fusion_pending"] = bud_run.pending_fusions
    # each subsequent batch processes two more deferred pairs first, so the
    # backlog shrinks deterministically until it is fully drained
    backlog = list(bud_run.pending_fusions)
    backlogs = [len(backlog)]
    current = bud_run
    while current.pending_fusions:
        current, _ = run_pipeline(bud_ws)
        backlogs.append(len(current.pending_fusions))
    summary["fusion_backlog_per_batch"] = backlogs
    strictly_shrinks = all(later < earlier
                           for earlier, later in zip(backlogs, backlogs[1:]))
    summary["fusion_backlog_drains"] = backlogs[-1] == 0 and strictly_shrinks

    # 8 fault injection
    inj_ws = fresh_workspace("inject")
    shutil.copy(ROOT / "configs" / "injections.json", inj_ws / "injections.json")
    injections = load_injections(inj_ws / "injections.json")
    inj_doc = run_injection_suite(
        inj_ws, "config.yaml", "samples\\manifest.json", injections,
        (3, 3), 16, 4, inj_ws / "reports" / "injection_report.json",
    )
    summary["injection_hits"] = inj_doc["summary"]

    # 9 serial vs concurrent equivalence
    ser_ws = fresh_workspace("serial")
    ser_cfg = (ser_ws / "config.yaml").read_text().replace("max_workers: 4", "max_workers: 1")
    (ser_ws / "config.yaml").write_text(ser_cfg)
    serial_run, _ = run_pipeline(ser_ws)
    summary["serial_equals_concurrent"] = raster_sig(serial_run) == raster_sig(cold)

    write_report(cold, ws / "reports" / "diagnostic_report.json")
    (OUT / "demo_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
