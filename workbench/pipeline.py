"""Pipeline orchestration: staged, incremental, budgeted, recoverable runs."""
from __future__ import annotations

import hashlib
import json
import signal
from pathlib import Path

import numpy as np

from . import errors
from .classify import classify_tile
from .config import (CLASSES, Config, config_hash, load_config,
                     validate_config_dict, weights_fingerprint, canonical_json)
from .fingerprints import (band_content_fingerprint, fusion_fingerprint,
                           result_version, tile_cache_key)
from .fixedpoint import FixedPoint
from .fusion import assemble_global, fuse_pair, postprocess
from .inject import InjectionSet
from .loader import (BandLoadError, adjacency_pairs, load_manifest,
                     load_tile_bands, pair_key, pair_sort_key, validate_manifest)
from .scheduler import execute_parallel, make_plan
from .snapshot import SnapshotStore

EXIT_FORCED_STOP = 75


class ForcedStop(SystemExit):
    def __init__(self):
        super().__init__(EXIT_FORCED_STOP)


class Pipeline:
    def __init__(self, samples_dir, work_dir, config_path=None, inject_path=None,
                 full=False, stop_after=None):
        self.samples_dir = Path(samples_dir)
        self.work_dir = Path(work_dir)
        self.config_path = config_path or self.samples_dir / "config.json"
        self.injections = InjectionSet.load(inject_path)
        self.full = full
        self.stop_after = dict(stop_after or {})
        self.store = SnapshotStore(self.work_dir / "snapshot")
        self.failed = []
        self.ledger = {"tiles": {}, "fusions": {}}
        self._signal_stop = False

    # ------------------------------------------------------------------ run
    def _install_signal_handlers(self):
        def handler(signum, frame):
            self._signal_stop = True
        try:
            signal.signal(signal.SIGTERM, handler)
            signal.signal(signal.SIGINT, handler)
        except ValueError:
            pass  # not in main thread (e.g. embedded use)

    def run(self) -> dict:
        self._install_signal_handlers()
        self.pending_tiles = []
        self.retained_fusions = []
        self.budget_trace = []
        config = self._load_config()
        fp = FixedPoint(config.fixed_point_frac_bits, config.quantum)
        manifest = load_manifest(self.samples_dir)
        manifest = self.injections.patch_manifest(manifest)
        if manifest.tile_size % config.window_size != 0:
            self.failed.append({"tile_id": "*", "stage": "config",
                                "code": errors.E_CONFIG_INVALID,
                                "detail": "window_size does not divide tile_size",
                                "inj_id": None})
            raise RuntimeError("window_size does not divide tile_size")
        load_failed, valid_ids = validate_manifest(manifest,
                                                   config.declared_band_set_versions)
        for entry in load_failed:
            entry["inj_id"] = self.injections.inj_id_for(entry["tile_id"], "load")
        self.failed.extend(load_failed)
        self.store.write_json("stage_load.json",
                              {"failed": self.failed, "valid": valid_ids})
        self.store.stage_commit("load")

        bands_by_tile, valid_ids = self._load_bands(manifest, valid_ids)
        valid_ids = self._check_finite(valid_ids, bands_by_tile)

        weights_fp = weights_fingerprint(config)
        fp_params = fp.params_fingerprint()
        cache_keys = {}
        for tid in valid_ids:
            bands_fp = band_content_fingerprint(bands_by_tile[tid])
            cache_keys[tid] = tile_cache_key(
                bands_fp, manifest.tiles[tid].band_set_version, weights_fp, fp_params)

        state = None if self.full else self.store.load_manifest_state()
        prev_tiles = (state or {}).get("tiles", {})
        prev_fusions = (state or {}).get("fusions", {})

        dirty_tiles = [tid for tid in valid_ids
                       if self.full
                       or tid not in prev_tiles
                       or prev_tiles[tid]["cache_key"] != cache_keys[tid]
                       or self.store.tile_meta(tid) is None]
        dirty_tiles.sort(key=lambda t: manifest.tiles[t].sort_key())

        tile_results = self._classify_stage(config, fp, manifest, dirty_tiles,
                                            bands_by_tile, cache_keys, prev_tiles)
        self._dirty_tiles = set(dirty_tiles)

        pairs = adjacency_pairs(manifest, sorted(tile_results))
        fusion_results = self._fuse_stage(config, fp, manifest, pairs,
                                          tile_results, cache_keys, prev_fusions)

        global_class = assemble_global(manifest, sorted(tile_results),
                                       tile_results, fusion_results, pairs)
        cleaned, refills = postprocess(global_class, config.min_connected_pixels)
        self.store.stage_commit("postprocess")

        report = self._build_report(config, manifest, tile_results, fusion_results,
                                    pairs, cleaned, refills)
        self.store.save_global(cleaned, refills, report)
        self._commit_state(config, manifest, cache_keys, fusion_results, report)
        return report

    # -------------------------------------------------------------- stages
    def _load_config(self) -> Config:
        raw = json.loads(Path(self.config_path).read_text(encoding="utf-8"))
        raw = self.injections.patch_config_raw(raw)
        errs = validate_config_dict(raw)
        fatal = [e for e in errs if "min_connected_pixels" not in e[1]]
        for code, detail in errs:
            self.failed.append({"tile_id": "*", "stage": "config",
                                "code": code, "detail": detail,
                                "inj_id": self.injections.inj_id_for("*", "config")})
        if fatal:
            raise RuntimeError("fatal config errors: " +
                               "; ".join(d for _, d in fatal))
        if errs:
            raw["min_connected_pixels"] = max(0, int(raw["min_connected_pixels"]))
        tmp = self.work_dir / "_effective_config.json"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(raw), encoding="utf-8")
        return load_config(tmp)

    def _load_bands(self, manifest, valid_ids):
        bands_by_tile = {}
        still_valid = []
        for tid in valid_ids:
            try:
                bands = load_tile_bands(
                    self.samples_dir, tid, manifest.bands, manifest.tile_size,
                    band_reader=self.injections.band_reader(tid))
            except BandLoadError as exc:
                self.failed.append({
                    "tile_id": tid, "stage": "load", "code": exc.code,
                    "detail": exc.detail,
                    "inj_id": self.injections.inj_id_for(tid, "load")})
                continue
            bands = self.injections.patch_bands(tid, bands)
            bands_by_tile[tid] = bands
            still_valid.append(tid)
        return bands_by_tile, still_valid

    def _check_finite(self, valid_ids, bands_by_tile):
        still_valid = []
        for tid in valid_ids:
            finite = all(np.isfinite(arr.astype(np.float64)).all()
                         for arr in bands_by_tile[tid].values())
            if not finite:
                self.failed.append({
                    "tile_id": tid, "stage": "classify",
                    "code": errors.E_BAND_NONFINITE,
                    "detail": "band contains non-finite values",
                    "inj_id": self.injections.inj_id_for(tid, "classify")})
                continue
            still_valid.append(tid)
        return still_valid

    def _classify_stage(self, config, fp, manifest, dirty_tiles, bands_by_tile,
                        cache_keys, prev_tiles):
        tile_pixels = manifest.tile_size * manifest.tile_size
        plan = make_plan(dirty_tiles, tile_pixels, [],
                         config.global_pixel_budget, config.per_tile_quota,
                         config.merge_batch_limit)
        self.budget_trace = plan.budget_trace
        self.pending_tiles = list(plan.global_pending)
        for tid in plan.quota_aborted:
            self.failed.append({"tile_id": tid, "stage": "classify",
                                "code": errors.E_TILE_BUDGET_EXHAUSTED,
                                "detail": "tile pixels exceed per-tile quota",
                                "inj_id": None})
        for tid in plan.global_pending:
            self.failed.append({"tile_id": tid, "stage": "classify",
                                "code": errors.E_GLOBAL_BUDGET_EXHAUSTED,
                                "detail": "global pixel budget exhausted",
                                "inj_id": None})

        stop_limit = self.stop_after.get("tiles")
        exec_order = list(plan.classify_order)
        stopped = False
        if stop_limit is not None and stop_limit < len(exec_order):
            exec_order = exec_order[:stop_limit]
            stopped = True

        def stop_check():
            return self._signal_stop

        def work(tid):
            return classify_tile(bands_by_tile[tid], config, fp)

        raw = execute_parallel(exec_order, work,
                               config.thread_pool_size, stop_check=stop_check)
        stopped = stopped or bool(raw.pop("__stopped__", False))

        tile_results = {}
        completed = 0
        for tid in plan.classify_order:  # stable order for ledger + commits
            if tid not in raw:
                self.pending_tiles.append(tid)  # not reached before stop
                continue
            result = raw[tid]
            version = result_version(cache_keys[tid])
            meta = {"cache_key": cache_keys[tid], "result_version": version,
                    "score_summary": result["score_summary"],
                    "prob_summary": result["prob_summary"],
                    "band_set_version": manifest.tiles[tid].band_set_version}
            self.store.save_tile_result(tid, result, meta)
            tile_results[tid] = result
            self.ledger["tiles"][tid] = {"cache_key": cache_keys[tid],
                                         "version": version, "origin": "recomputed"}
            completed += 1
            if completed % max(1, config.snapshot_commit_interval) == 0:
                self._commit_partial(config, manifest, cache_keys)
        # reused tiles (only non-dirty ones may be reused)
        for tid in cache_keys:
            if tid in tile_results or tid in self.pending_tiles:
                continue
            if tid in set(dirty_tiles):
                continue
            stored = self.store.load_tile_result(tid)
            if stored is None:
                continue
            tile_results[tid] = stored
            meta = self.store.tile_meta(tid)
            self.ledger["tiles"][tid] = {"cache_key": cache_keys[tid],
                                         "version": meta["result_version"],
                                         "origin": "reused"}
        self.store.stage_commit("classify")
        if stopped:
            self._commit_partial(config, manifest, cache_keys)
            self._commit_queues()
            raise ForcedStop()
        return tile_results

    def _fuse_stage(self, config, fp, manifest, pairs, tile_results,
                    cache_keys, prev_fusions):
        fp_params = fp.params_fingerprint()
        pair_fps = {}
        for id_a, id_b in pairs:
            geom_desc = {"orientation": "horizontal"
                         if manifest.tiles[id_a].row == manifest.tiles[id_b].row
                         else "vertical",
                         "overlap_px": manifest.overlap_px,
                         "tile_size": manifest.tile_size}
            pair_fps[pair_key(manifest, id_a, id_b)] = fusion_fingerprint(
                cache_keys[id_a], cache_keys[id_b],
                tile_results[id_a]["prob_summary"]["hash"],
                tile_results[id_b]["prob_summary"]["hash"],
                geom_desc, config.fusion_strategy_version, fp_params)
        dirty_pairs = [pk for pk in (pair_key(manifest, a, b) for a, b in pairs)
                       if self.full
                       or pk not in prev_fusions
                       or prev_fusions[pk]["fingerprint"] != pair_fps[pk]
                       or self.store.fusion_meta(pk) is None]
        dirty_pairs.sort(key=lambda pk: self._pair_sort(manifest, pk))
        plan = make_plan([], 0, dirty_pairs, 0, 0, config.merge_batch_limit)
        self.retained_fusions = list(plan.fuse_pending)

        stop_limit = self.stop_after.get("fusions")
        exec_order = list(plan.fuse_order)
        stopped = False
        if stop_limit is not None and stop_limit < len(exec_order):
            exec_order = exec_order[:stop_limit]
            stopped = True

        def stop_check():
            return self._signal_stop

        def work(pk):
            id_a, id_b = pk.split("::")
            return fuse_pair(tile_results[id_a], tile_results[id_b],
                             manifest.tiles[id_a], manifest.tiles[id_b],
                             manifest, config, fp,
                             self.ledger["tiles"][id_a]["version"],
                             self.ledger["tiles"][id_b]["version"])

        raw = execute_parallel(exec_order, work, config.thread_pool_size,
                               stop_check=stop_check)
        stopped = stopped or bool(raw.pop("__stopped__", False))

        fusion_results = {}
        for pk in plan.fuse_order:
            if pk not in raw:
                self.retained_fusions.append(pk)
                continue
            result = raw[pk]
            version = pair_fps[pk][:16]
            meta = {"fingerprint": pair_fps[pk], "version": version,
                    "geometry": result["geometry"]}
            self.store.save_fusion_result(pk, result, meta)
            fusion_results[pk] = result
            self.ledger["fusions"][pk] = {"fingerprint": pair_fps[pk],
                                          "version": version,
                                          "origin": "recomputed"}
        dirty_set = set(dirty_pairs)
        for pk in pair_fps:
            if pk in fusion_results or pk in self.retained_fusions:
                continue
            if pk in dirty_set:
                continue
            stored = self.store.load_fusion_result(pk)
            if stored is None:
                continue
            fusion_results[pk] = stored
            meta = self.store.fusion_meta(pk)
            self.ledger["fusions"][pk] = {"fingerprint": pair_fps[pk],
                                          "version": meta["version"],
                                          "origin": "reused"}
        self.store.stage_commit("fuse")
        if stopped:
            self._commit_partial(config, manifest, cache_keys)
            self._commit_queues()
            raise ForcedStop()
        return fusion_results

    @staticmethod
    def _pair_sort(manifest, pk):
        id_a, id_b = pk.split("::")
        return pair_sort_key(manifest, id_a, id_b)

    # ------------------------------------------------------------- report
    def _build_report(self, config, manifest, tile_results, fusion_results,
                      pairs, cleaned, refills):
        boundary_events = []
        for pk in sorted(fusion_results):
            boundary_events.extend(fusion_results[pk]["boundary_events"])
        boundary_events.sort(key=lambda e: (e["global_row"], e["global_col"]))
        confusion = self._confusion(cleaned)
        fusion_stats = {}
        for pk in sorted(fusion_results):
            res = fusion_results[pk]
            fused = res["fused"]
            fusion_stats[pk] = {
                "pixels": int(fused.size),
                "class_counts": [int((fused == c).sum()) for c in range(len(CLASSES))],
                "sensitivity_events": len(res["boundary_events"]),
                "vote_hash": hashlib.sha256(
                    canonical_json(res["records"]).encode()).hexdigest()[:16],
            }
        per_tile_counts = {}
        per_tile_probs = {}
        for tid in sorted(tile_results):
            summary = tile_results[tid]["score_summary"]
            per_tile_counts[tid] = summary["class_pixel_counts"]
            per_tile_probs[tid] = tile_results[tid]["prob_summary"]
        global_version = hashlib.sha256(canonical_json({
            "config": config_hash(config),
            "tiles": {t: self.ledger["tiles"][t]["version"]
                      for t in sorted(self.ledger["tiles"])},
            "fusions": {p: self.ledger["fusions"][p]["version"]
                        for p in sorted(self.ledger["fusions"])},
        }).encode()).hexdigest()
        report = {
            "config_hash": config_hash(config),
            "fusion_strategy_version": config.fusion_strategy_version,
            "band_set_versions": {t: manifest.tiles[t].band_set_version
                                  for t in sorted(tile_results)},
            "global_version": global_version,
            "global_class_hash": hashlib.sha256(cleaned.tobytes()).hexdigest(),
            "global_shape": list(cleaned.shape),
            "per_tile_class_counts": per_tile_counts,
            "per_tile_prob_summaries": per_tile_probs,
            "fusion_stats": fusion_stats,
            "boundary_events": boundary_events,
            "refills": refills,
            "budget_trace": self.budget_trace,
            "failed": sorted(self.failed, key=lambda f: (f["stage"], f["tile_id"],
                                                         f["code"])),
            "pending_tiles": sorted(self.pending_tiles),
            "retained_fusions": sorted(self.retained_fusions),
            "invalidation_ledger": self.ledger,
            "confusion": confusion,
            "injections_hit": sorted(set(self.injections.hits)),
        }
        return report

    def _confusion(self, cleaned):
        gt_path = self.samples_dir / "ground_truth.npy"
        n = len(CLASSES)
        matrix = [[0] * n for _ in range(n)]
        if not gt_path.exists():
            return {"matrix": matrix, "correct": 0, "total": 0}
        gt = np.load(gt_path)
        rows = min(gt.shape[0], cleaned.shape[0])
        cols = min(gt.shape[1], cleaned.shape[1])
        gt = gt[:rows, :cols]
        pred = cleaned[:rows, :cols]
        for t in range(n):
            for p in range(n):
                matrix[t][p] = int(((gt == t) & (pred == p)).sum())
        correct = sum(matrix[i][i] for i in range(n))
        return {"matrix": matrix, "correct": int(correct), "total": int(gt.size)}

    # ------------------------------------------------------------- commit
    def _commit_partial(self, config, manifest, cache_keys):
        state = self.store.load_manifest_state() or {"tiles": {}, "fusions": {}}
        state["tiles"] = {k: v for k, v in state["tiles"].items() if k in cache_keys}
        state["fusions"] = {k: v for k, v in state["fusions"].items()
                            if all(part in cache_keys for part in k.split("::"))}
        for tid, entry in self.ledger["tiles"].items():
            state["tiles"][tid] = {"cache_key": entry["cache_key"],
                                   "result_version": entry["version"]}
        for pk, entry in self.ledger["fusions"].items():
            state["fusions"][pk] = {"fingerprint": entry["fingerprint"],
                                    "version": entry["version"]}
        state["config_hash"] = config_hash(config)
        state["fusion_strategy_version"] = config.fusion_strategy_version
        state["band_set_versions"] = {t: manifest.tiles[t].band_set_version
                                      for t in sorted(cache_keys)}
        self.store.commit_manifest_state(state)

    def _commit_state(self, config, manifest, cache_keys, fusion_results, report):
        self._commit_partial(config, manifest, cache_keys)
        self._commit_queues()

    def _commit_queues(self):
        self.store.write_json("queues.json", {
            "pending_tiles": sorted(self.pending_tiles),
            "retained_fusions": sorted(self.retained_fusions),
            "failed": self.failed,
            "budget_trace": self.budget_trace,
        })
