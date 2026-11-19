"""Incremental segmentation engine.

The engine never mutates cached artifacts in place. For every run it:

1. loads and validates the index (symmetric adjacency, unique row/col),
2. derives tile cache keys from current band bytes + index version + config,
3. recomputes only missing/invalid tiles within the recompute budget,
4. recomputes only fusion regions whose dependency fingerprint changed,
5. reassembles and post-processes the global grid,
6. atomically commits a new versioned snapshot.

A full rebuild is simply a run against an empty work directory, so by
construction the incremental and full paths share identical algorithms and
deterministic ordering.
"""

from __future__ import annotations

import json
import signal
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import SegmentationConfig
from .errors import BusinessError, ErrorCode, FailureRecord
from .features import classify_tile
from .fingerprint import (
    ResultStore,
    fusion_fingerprint,
    index_adjacency_version,
    tile_cache_key_with_version,
)
from .fusion import assemble_global, build_fusion_regions, postprocess
from .injection import Injection, non_finite_patch, truncated_png_bytes
from .runner_types import RunInputs, RunReport
from .snapshot import SnapshotManager
from .tiles import TileRecord, load_tile_bands

_SHUTDOWN = threading.Event()


def request_shutdown() -> None:
    _SHUTDOWN.set()


def install_signal_handlers() -> None:
    def _handler(signum, frame):  # pragma: no cover - signal timing dependent
        request_shutdown()

    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass


class IncrementalEngine:
    def __init__(self, inputs: RunInputs):
        self.inputs = inputs
        self.store = ResultStore(inputs.work_dir / "cache")
        self.snapshots = SnapshotManager(inputs.work_dir)
        self.report = RunReport(
            run_mode="incremental",
            config_hash=inputs.config.fingerprint(),
            index_version=inputs.index.band_order_version,
        )
        self._bands: dict[tuple[int, int], np.ndarray] = {}
        self._tile_keys: dict[tuple[int, int], str] = {}
        self._tile_results: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._failures: list[FailureRecord] = []
        self._lock = threading.Lock()
        self._adjacency_version = index_adjacency_version(inputs.index)

    # ------------------------------------------------------------------ index
    def _record_failure(self, record: FailureRecord) -> None:
        with self._lock:
            self._failures.append(record)
            if record.injection_id:
                self.report.hits.append(record.injection_id)

    def _injection_for_tile(self, key: tuple[int, int]) -> Injection | None:
        if self.inputs.injections is None:
            return None
        return self.inputs.injections.for_tile(key)

    # ------------------------------------------------------------------ tiles
    def _load_bands(self, tile: TileRecord) -> np.ndarray | None:
        injection = self._injection_for_tile(tile.key)
        patches_dir = self.inputs.patches_dir
        if injection is not None and injection.kind == "non_finite_band":
            patch_dir = self.inputs.work_dir / "injection_patches"
            patch_dir.mkdir(parents=True, exist_ok=True)
            non_finite_patch(
                patch_dir / f"{tile.name}.npy",
                (self.inputs.index.tile_size, self.inputs.index.tile_size,
                 self.inputs.index.band_count),
            )
            patches_dir = patch_dir
        if injection is not None and injection.kind == "truncated_png":
            staged = self.inputs.work_dir / "injection_tiles"
            staged.mkdir(parents=True, exist_ok=True)
            target = staged / tile.file
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes((self.inputs.sample_dir / tile.file).read_bytes())
                truncated_png_bytes(target, injection)
            sample_dir = staged
        else:
            sample_dir = self.inputs.sample_dir
        try:
            return load_tile_bands(
                tile, sample_dir, self.inputs.index,
                patches_dir=patches_dir, injection_id=injection.injection_id if injection else None,
            )
        except BusinessError as exc:
            self._record_failure(FailureRecord(
                code=exc.code,
                target=f"tile:{tile.row},{tile.col}",
                message=exc.message,
                sort_key=tile.key,
                injection_id=exc.injection_id,
                detail=exc.detail,
            ))
            return None

    # ------------------------------------------------------------------ run
    def run(self, *, force_full: bool = False) -> RunReport:
        if force_full:
            self.report.run_mode = "full_rebuild"
        cfg = self.inputs.config
        if self.inputs.injections is not None:
            override = self.inputs.injections.budget_override()
            if override is not None:
                cfg = SegmentationConfig.from_dict({**cfg.to_dict(), "recompute_budget": override})
        self.report.budget = cfg.recompute_budget
        budget_injection = None
        if self.inputs.injections is not None and self.inputs.injections.budget_override() is not None:
            budget_injection = next(
                (item.injection_id for item in self.inputs.injections.injections
                 if item.kind == "zero_budget"), None)
            if budget_injection:
                self.report.hits.append(budget_injection)
        self._budget_injection = budget_injection

        ordered_tiles = sorted(self.inputs.index.tiles, key=lambda tile: tile.key)
        self.report.requested_tiles = [tile.key for tile in ordered_tiles]
        if force_full:
            self._wipe_cache()

        # Phase 1: determine the invalid set in row-major order. A tile is
        # invalid when its derived cache key has no matching cached artifact.
        plan: list[tuple[TileRecord, str, np.ndarray]] = []
        invalid: list[tuple[int, int]] = []
        failed_tiles: list[tuple[int, int]] = []
        for tile in ordered_tiles:
            bands = self._load_bands(tile)
            if bands is None:
                failed_tiles.append(tile.key)
                continue
            cache_key = tile_cache_key_with_version(
                tile, bands, cfg, self.inputs.index.band_order_version
            )
            self._bands[tile.key] = bands
            self._tile_keys[tile.key] = cache_key
            if self.store.has_tile(tile.key, cache_key):
                self._tile_results[tile.key] = self.store.load_tile(tile.key, cache_key)
                self.report.reused_tiles.append(tile.key)
                self.store.prune_tile(tile.key, cache_key)
            else:
                invalid.append(tile.key)
                plan.append((tile, cache_key, bands))

        # Phase 2: honor the recompute budget in stable row-major order.
        if cfg.recompute_budget == 0:
            selected, deferred = [], list(plan)
        else:
            selected = plan[:cfg.recompute_budget]
            deferred = plan[cfg.recompute_budget:]
        pending_tiles = [tile.key for tile, _key, _bands in deferred]

        # Phase 3: execute the selected work; concurrency never changes order.
        def work(item: tuple[TileRecord, str, np.ndarray]) -> tuple[tuple[int, int], bool]:
            tile, cache_key, bands = item
            if _SHUTDOWN.is_set():
                return tile.key, False
            class_map, confidence, score_map = classify_tile(bands, cfg)
            self.store.save_tile(tile.key, cache_key, class_map, confidence, score_map)
            with self._lock:
                self._tile_results[tile.key] = {
                    "class_map": class_map,
                    "confidence": confidence,
                    "score_map": score_map,
                }
            self.store.prune_tile(tile.key, cache_key)
            return tile.key, True

        if cfg.workers == 1 or len(selected) <= 1:
            outcomes = [work(item) for item in selected]
        else:
            with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                futures = [pool.submit(work, item) for item in selected]
                outcomes = [future.result() for future in futures]
        for key, ok in outcomes:
            if ok:
                self.report.recomputed_tiles.append(key)
            else:
                pending_tiles.append(key)
        self.report.recomputed_tiles.sort()
        self.report.pending_tiles = sorted(set(pending_tiles + failed_tiles))
        self.report.budget_used = len(self.report.recomputed_tiles)

        fusion_state = self._run_fusions(cfg)
        self._finalize(cfg, fusion_state)
        self.report.failures = sorted(
            self._failures + self._budget_failures(self.report.pending_tiles, fusion_state),
            key=lambda record: record.sort_key,
        )
        self.report.signaled = _SHUTDOWN.is_set()
        return self.report

    def _budget_failures(self, pending_tiles, fusion_state) -> list[FailureRecord]:
        records: list[FailureRecord] = []
        injection_id = getattr(self, "_budget_injection", None)
        for key in pending_tiles:
            records.append(FailureRecord(
                code=ErrorCode.E_RECOMPUTE_BUDGET_PENDING,
                target=f"tile:{key[0]},{key[1]}",
                message="recompute deferred to next batch due to budget",
                sort_key=key,
                injection_id=injection_id,
            ))
        for name in fusion_state["pending_fusions"]:
            records.append(FailureRecord(
                code=ErrorCode.E_RECOMPUTE_BUDGET_PENDING,
                target=f"fusion:{name}",
                message="fusion recompute deferred to next batch due to budget",
                sort_key=(10**9, 10**9),
                injection_id=injection_id,
            ))
        return records

    def _run_fusions(self, cfg: SegmentationConfig) -> dict:
        regions = build_fusion_regions(self.inputs.index)
        pending_fusions: list[str] = []
        fusion_patches: dict[str, np.ndarray] = {}
        for region in regions:
            if region.a not in self._tile_results or region.b not in self._tile_results:
                pending_fusions.append(region.name)
                continue
            fingerprint = fusion_fingerprint(region, self._tile_keys, cfg, self._adjacency_version)
            if self.store.has_fusion(region, fingerprint):
                fusion_patches[region.name] = self.store.load_fusion(region, fingerprint)
                self.report.reused_fusions.append(region.name)
            else:
                patch = self._compute_fusion(region, cfg)
                if patch is None:
                    pending_fusions.append(region.name)
                    continue
                self.store.save_fusion(region, fingerprint, patch)
                fusion_patches[region.name] = patch
                self.report.recomputed_fusions.append(region.name)
            self.store.prune_fusion(region, fingerprint)
        return {"regions": regions, "patches": fusion_patches, "pending_fusions": pending_fusions}

    def _compute_fusion(self, region, cfg: SegmentationConfig) -> np.ndarray | None:
        from .fusion import _weighted_vote, _distance_weights

        map_a = self._tile_results[region.a]["class_map"]
        map_b = self._tile_results[region.b]["class_map"]
        ax0, ay0, ax1, ay1 = region.local_a_box
        bx0, by0, bx1, by1 = region.local_b_box
        patch_a = map_a[ay0:ay1, ax0:ax1]
        patch_b = map_b[by0:by1, bx0:bx1]
        if cfg.fusion_strategy == "first_wins":
            return np.where(region.a < region.b, patch_a, patch_b).astype(np.uint8)
        weight_a, weight_b = _distance_weights(region, self.inputs.index)
        return _weighted_vote(patch_a, patch_b, weight_a, weight_b, region.a, region.b)

    def _finalize(self, cfg: SegmentationConfig, fusion_state: dict) -> None:
        available = set(self._tile_results)
        if not available or set(self.inputs.index.ordered_keys()) != available:
            # Without every tile, do not publish a global grid; snapshot still
            # records the recovery state if at least one version exists.
            self.report.pending_fusions = fusion_state["pending_fusions"]
            previous = self.snapshots.latest()
            self.report.snapshot_version = previous.version if previous else None
            return

        class_maps = {key: value["class_map"] for key, value in self._tile_results.items()}
        raw_grid = assemble_global(self.inputs.index, class_maps, cfg)
        final_grid, changes = postprocess(raw_grid, cfg.min_component_area)

        previous = self.snapshots.latest()
        material_changes = bool(self.report.recomputed_tiles or
                                self.report.recomputed_fusions)
        if previous is not None and not material_changes:
            self.report.snapshot_version = previous.version
            self.report.pending_fusions = fusion_state["pending_fusions"]
            self.store.write_json("last_report.json", self.report.to_dict())
            return
        version = (previous.version + 1) if previous else 1
        tile_versions = {
            f"r{r},c{c}": key for (r, c), key in sorted(self._tile_keys.items())
        }
        fusion_versions = {}
        for region in fusion_state["regions"]:
            if region.name in fusion_state["patches"]:
                fusion_versions[region.name] = fusion_fingerprint(region, self._tile_keys, cfg, self._adjacency_version)
        info = self.snapshots.commit(
            version=version,
            config_hash=cfg.fingerprint(),
            index_version=self.inputs.index.band_order_version,
            tile_versions=tile_versions,
            fusion_versions=fusion_versions,
            global_grid=final_grid,
            postprocess_records=changes,
            pending=[list(k) for k in self.report.pending_tiles],
            failures=self.report.failure_dicts(),
        )
        self.report.snapshot_version = info.version
        self.report.pending_fusions = fusion_state["pending_fusions"]
        self.store.write_json("last_report.json", self.report.to_dict())

    def _wipe_cache(self) -> None:
        import shutil

        shutil.rmtree(self.inputs.work_dir / "cache", ignore_errors=True)
        self.store = ResultStore(self.inputs.work_dir / "cache")
        self._tile_results.clear()
        self._tile_keys.clear()
        self.report.reused_tiles.clear()
