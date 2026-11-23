"""End-to-end orchestration with precise incremental invalidation.

Stages: load -> classify -> fuse -> postprocess.  Tile cache keys derive from
band content + band-set version + weight fingerprint; fusion fingerprints
derive from both tile keys + probability summaries + geometry + strategy
version.  On a rerun only invalidated tiles and the fusion bands they
participate in are recomputed; everything else is served from cache.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .classifier import TileResult, classify_tile, make_cache_key
from .config import Config
from .diagnostics import BoundaryRecord, evaluate_boundary
from .errors import Code, FailureList
from .fixedpoint import FixedConfig
from .fusion import FusionResult, fuse_pair, pair_direction
from .loader import load_manifest, validate_and_load
from .model import CLASSES, Tile
from .paths import to_path
from .postprocess import GlobalRaster, assemble_global, remove_small_components
from .scheduler import Scheduler
from .snapshot import SnapshotStore, config_hash


@dataclass
class RunReport:
    config_hash: str
    global_version: int
    tile_results: Dict[str, TileResult]
    fusions: List[FusionResult]
    global_raster: Optional[GlobalRaster]
    failures: FailureList
    recomputed_tiles: List[str]
    cached_tiles: List[str]
    invalidated_tiles: List[str]
    recomputed_fusions: List[str]
    cached_fusions: List[str]
    pending_fusions: List[str]
    boundary_records: List[BoundaryRecord]
    budget: Dict[str, Any]
    stages_committed: List[str]
    manifest_doc: Dict[str, Any]

    def diagnostic_dicts(self) -> List[dict]:
        return [r.__dict__ for r in self.boundary_records]


class Pipeline:
    def __init__(self, config: Config, root: str | Path, global_version: int = 1):
        self.config = config
        self.root = Path(root)
        self.global_version = global_version
        self.failures = FailureList()
        self.store = SnapshotStore(to_path(config.snapshot_dir, root), config)
        self.scheduler = Scheduler(config, self.failures)

    # ------------------------------------------------------------------ stage 1
    def stage_load(self) -> Tuple[Dict[str, Tile], Any]:
        manifest_native = to_path(self.config.manifest_path, self.root)
        manifest = load_manifest(manifest_native, self.root)
        tiles, load_failures = validate_and_load(manifest, self.config, self.root)
        for f in load_failures:
            self.failures.add(f.code, f.target, f.message, f.injection_id, f.detail)
        self._commit_stage("load", {"validation.json": {
            "valid_tiles": sorted(tiles.keys()),
            "failures": self.failures.to_dicts(),
        }})
        return tiles, manifest

    def _commit_stage(self, stage: str, files: Dict[str, Any]) -> None:
        self.store._replace_stage_dir(stage, files)

    # ------------------------------------------------------------------ stage 2
    def stage_classify(self, tiles: Dict[str, Tile],
                       previous: Optional[dict]) -> Tuple[Dict[str, TileResult], List[str], List[str], List[str]]:
        tile_results: Dict[str, TileResult] = {}
        desired_keys: Dict[str, str] = {}
        invalidated: List[str] = []

        for tile in tiles.values():
            key, _, _ = make_cache_key(tile, self.config)
            desired_keys[tile.tile_id] = key

        def cache_hit(tile: Tile) -> bool:
            key = desired_keys[tile.tile_id]
            cached = self.store.load_tile(key)
            if cached is not None:
                tile_results[tile.tile_id] = cached
                return True
            invalidated.append(tile.tile_id)
            return False

        results, pending, recomputed, cached = self.scheduler.run_classification(
            tiles, compute_fn=lambda t: classify_tile(t, self.config), cache_hit_fn=cache_hit
        )
        for result in results:
            self.store.save_tile(result)
            tile_results[result.tile_id] = result

        self._commit_stage("classify", {
            "tile_keys.json": {"desired": desired_keys, "recomputed": recomputed, "cached": cached},
        })
        return tile_results, recomputed, cached, sorted(set(invalidated))

    # ------------------------------------------------------------------ stage 3
    def _ordered_pairs(self, tiles: Dict[str, Tile],
                       tile_results: Dict[str, TileResult]) -> List[Tuple[str, TileResult, TileResult]]:
        ids = set(tile_results.keys())
        pairs: List[Tuple[str, TileResult, TileResult]] = []
        seen: set[str] = set()
        for tile in sorted(tiles.values(), key=lambda t: (t.spec.row, t.spec.col, t.tile_id)):
            if tile.tile_id not in ids:
                continue
            a = tile_results[tile.tile_id]
            for nb in sorted(tile.spec.neighbors):
                if nb not in ids:
                    continue
                b = tile_results[nb]
                direction = pair_direction(a, b, self.config.window.size)
                if direction not in ("right", "down"):
                    continue
                key = f"{a.tile_id}|{b.tile_id}"
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((key, a, b))
        pairs.sort(key=lambda p: p[0])
        return pairs

    def stage_fuse(self, tiles: Dict[str, Tile], tile_results: Dict[str, TileResult],
                   edge: int, overlap: int) -> Tuple[List[FusionResult], List[str], List[str], List[str]]:
        pairs = self._ordered_pairs(tiles, tile_results)
        strategy = self.config.fusion_strategy_version

        def cache_hit(pair_key: str) -> bool:
            key, a, b = next(p for p in pairs if p[0] == pair_key)
            from .fusion import fusion_fingerprint, overlap_geometry
            direction = pair_direction(a, b, edge - overlap)
            geom = overlap_geometry(a, b, edge, overlap, direction)
            fp = fusion_fingerprint(a, b, geom, strategy)
            cached = self.store.load_fusion(fp)
            if cached is not None:
                self._fusion_cache[pair_key] = cached
                return True
            self._fusion_fp[pair_key] = fp
            return False

        self._fusion_cache: Dict[str, FusionResult] = {}
        self._fusion_fp: Dict[str, str] = {}

        def fuse_fn(a: TileResult, b: TileResult) -> FusionResult:
            return fuse_pair(a, b, edge, overlap, strategy)

        results, pending, recomputed, cached = self.scheduler.run_fusion_batch(
            pairs, fuse_fn=fuse_fn, cache_hit_fn=cache_hit
        )
        for fusion in results:
            self.store.save_fusion(fusion)
        for key, fusion in self._fusion_cache.items():
            if fusion not in results:
                results.append(fusion)
        results.sort(key=lambda f: f.pair_key)
        cached_fusion_ids = [k for k, v in self._fusion_cache.items()]
        self._commit_stage("fuse", {"fusion_keys.json": {
            "recomputed": recomputed, "cached": sorted(cached_fusion_ids), "pending": pending,
        }})
        return results, pending, recomputed, sorted(cached_fusion_ids)

    # ------------------------------------------------------------------ stage 4
    def stage_postprocess(self, tile_results: Dict[str, TileResult], fusions: List[FusionResult],
                          grid_rows: int, grid_cols: int, edge: int, overlap: int) -> GlobalRaster:
        global_ras = assemble_global(tile_results, fusions, grid_rows, grid_cols, edge, overlap)
        global_ras = remove_small_components(global_ras, self.config.min_component_pixels)
        self.store.save_global(global_ras, {"strategy": self.config.fusion_strategy_version})
        self._commit_stage("postprocess", {"status.json": {"committed": True}})
        return global_ras

    # ------------------------------------------------------------------ run
    def run(self, expected_grid: Tuple[int, int], edge: int, overlap: int) -> RunReport:
        existing = self.store.check_resume(self.global_version, self.failures)
        if not existing:
            # version/config change: rebuild from the affected earliest stage.
            self.store.reset_stages_from("load")

        tiles, manifest = self.stage_load()
        tile_results, recomputed_t, cached_t, invalidated_t = self.stage_classify(tiles, existing)
        fusions, pending_f, recomputed_f, cached_f = self.stage_fuse(
            tiles, tile_results, edge, overlap
        )

        boundary: List[BoundaryRecord] = []
        fcfg = FixedConfig(self.config.fixed.fractional_bits, self.config.fixed.gear)
        unit = fcfg.scale >> fcfg.gear_shift
        by_id = tile_results
        for fusion in fusions:
            boundary.extend(
                evaluate_boundary(
                    fusion, by_id[fusion.tile_a], by_id[fusion.tile_b], fcfg,
                    self.config.fusion_strategy_version, overlap, unit,
                )
            )
        for rec in boundary:
            if rec.flipped:
                self.failures.add(
                    Code.E_BOUNDARY_FLIP,
                    rec.pair_key,
                    f"boundary flip at ({rec.global_row},{rec.global_col})",
                    detail={"attribution": rec.attribution},
                )

        global_ras = None
        if not pending_f:
            global_ras = self.stage_postprocess(
                tile_results, fusions, expected_grid[0], expected_grid[1], edge, overlap
            )

        # Total pixels represented by completed tiles (cached and recomputed).
        consumed_quota = sum(r.pixels_classified for r in tile_results.values())
        manifest_doc = {
            "global_version": self.global_version,
            "config_hash": config_hash(self.config),
            "bandset_versions": sorted({t.spec.bandset for t in tiles.values()}),
            "fusion_strategy_version": self.config.fusion_strategy_version,
            "completed_tiles": sorted(tile_results.keys()),
            "consumed_pixel_quota": consumed_quota,
            "newly_consumed_pixel_quota": self.scheduler.budget.pixels_used,
            "remaining_pixel_budget": self.scheduler.budget.remaining,
            "stages": ["load", "classify", "fuse"] + (["postprocess"] if global_ras else []),
            "failures": self.failures.to_dicts(),
        }
        self.store.write_manifest(manifest_doc)
        self.store.cleanup_tmp()

        return RunReport(
            config_hash=config_hash(self.config),
            global_version=self.global_version,
            tile_results=tile_results,
            fusions=fusions,
            global_raster=global_ras,
            failures=self.failures,
            recomputed_tiles=recomputed_t,
            cached_tiles=sorted(cached_t),
            invalidated_tiles=invalidated_t,
            recomputed_fusions=recomputed_f,
            cached_fusions=cached_f,
            pending_fusions=pending_f,
            boundary_records=boundary,
            budget={
                "pixels_used": self.scheduler.budget.pixels_used,
                "remaining": self.scheduler.budget.remaining,
                "tile_quota_hits": self.scheduler.budget.tile_quota_hits,
                "global_budget_hits": self.scheduler.budget.global_budget_hits,
                "fusion_pending": self.scheduler.budget.fusion_pending,
            },
            stages_committed=manifest_doc["stages"],
            manifest_doc=manifest_doc,
        )
