"""增量流水线：预算调度、内容指纹失效/复用、回归对齐、快照提交。"""
from __future__ import annotations

from collections import defaultdict

from .calibration import calibrate_tile_band, stable_key
from .errors import E_CALIB_BUDGET_PENDING, STAGE_CALIB, STAGE_REGRESS, StageError
from .executor import run_keyed
from .models import CalibRecord
from .regression import align_tile, model_id, solve_model
from .store import StateStore
from .fingerprint import content_fingerprint


class Pipeline:
    def __init__(self, store: StateStore, config: dict, table: dict[str, dict],
                 concurrency: int | None = None):
        self.store = store
        self.config = config
        self.table = table
        self.concurrency = int(config["concurrency"] if concurrency is None else concurrency)
        self.pending: list[tuple[str, int]] = []
        self.cumulative_pixels = 0
        self.total_grant = 0

    def _record_failure(self, failure: dict) -> None:
        """内容寻址失败清单：同一失败在任意批次序列中只保留一份，顺序确定。"""
        key = content_fingerprint(failure)
        for existing in self.store.failures:
            if existing.get("_failure_key") == key:
                return
        record = dict(failure)
        record["_failure_key"] = key
        self.store.failures.append(record)

    @staticmethod
    def _sort_failures(failures: list[dict]) -> list[dict]:
        return sorted(failures, key=lambda f: (
            f.get("stage", ""), f.get("tile_id", ""), f.get("band", -1),
            f.get("code", ""), f.get("injection_id", "")))

    # ------------------------------------------------------------------
    def run_batch(self, batch_id: int, tiles: list[dict]) -> dict:
        budget = int(self.config["pixel_budget"])
        self.total_grant += budget
        ledger = self.store.ledger
        failures = self.store.failures

        # 稳定顺序：瓦片 ID 升序、波段号升序；上一批待办优先
        index = {(t.tile_id, t.band): t for t in tiles}
        ordered = sorted(tiles, key=lambda t: stable_key(t.tile_id, t.band))
        work: list = []
        pending_carry = list(self.pending)
        self.pending = []
        seen_pairs: set[tuple[str, int]] = set()
        for pair in pending_carry + [(t.tile_id, t.band) for t in ordered]:
            if pair in seen_pairs or pair not in index:
                continue
            seen_pairs.add(pair)
            work.append(index[pair])

        # ---------- 阶段 1：逐瓦片波段定标（受预算约束） ----------
        reusable: list = []
        to_compute: list = []
        permanent_failures = {
            (f.get("tile_id"), f.get("band"))
            for f in self.store.failures
            if f.get("stage") == STAGE_CALIB and f.get("code") != E_CALIB_BUDGET_PENDING
        }
        for tile in work:
            pair = (tile.tile_id, tile.band)
            if pair in permanent_failures:
                # 已知永久失败：保留失败记录，不参与预算调度
                if pair in self.pending:
                    self.pending.remove(pair)
                continue
            probe = self._probe_calib(tile)
            if probe is not None:
                reusable.append((tile, probe))
            else:
                to_compute.append(tile)

        processed_tiles: list = []
        grant_left = budget
        for tile, existing in reusable:
            self.store.reuse_log.append({
                "stage": STAGE_CALIB, "tile_id": tile.tile_id, "band": tile.band,
                "key": existing["key"], "action": "REUSED",
            })
            ledger.append(self._ledger_row(batch_id, tile, 0, grant_left, "REUSED"))
            processed_tiles.append(tile)

        budget_tasks: list = []
        for tile in to_compute:
            pixels = len(tile.counts)
            if pixels > grant_left:
                self.pending.append((tile.tile_id, tile.band))
                self._record_failure(StageError(
                    E_CALIB_BUDGET_PENDING,
                    f"瓦片 {tile.tile_id} 波段 {tile.band} 需 {pixels} 像素，"
                    f"批次剩余预算 {grant_left}，转入待办",
                    tile_id=tile.tile_id, band=tile.band, stage=STAGE_CALIB,
                    sensor_id=tile.sensor_id).to_failure())
                ledger.append(self._ledger_row(batch_id, tile, 0, grant_left, "PENDING"))
                continue
            grant_left -= pixels
            self.cumulative_pixels += pixels
            budget_tasks.append((stable_key(tile.tile_id, tile.band), tile))

        results = run_keyed(budget_tasks, self._calib_safe, self.concurrency)
        failed_pairs: set[tuple[str, int]] = set()
        for _, outcome in results:
            tile, record, error = outcome
            if error is not None:
                self._record_failure(error)
                ledger.append(self._ledger_row(batch_id, tile, len(tile.counts),
                                               grant_left, "FAILED"))
                failed_pairs.add((tile.tile_id, tile.band))
                continue
            self.store.calib[record.key] = record.to_dict()
            ledger.append(self._ledger_row(batch_id, tile, record.pixel_count,
                                           grant_left, "PROCESSED"))
            processed_tiles.append(tile)

        # ---------- 阶段 2/3：回归模型与对齐（仅在无待办时运行，保证确定性） ----------
        # 永久失败项不得残留在待办队列，否则调度无法终止
        self.pending = [pair for pair in self.pending if pair not in failed_pairs]
        if not self.pending:
            self._run_regression(ordered, failures)
            self._gc_artifacts(ordered)
        self._prune_processing_failures(ordered)
        completed = sorted({t.tile_id for t in ordered if (t.tile_id, t.band)
                            not in self.pending})
        return {
            "batch_id": batch_id,
            "budget_grant": budget,
            "budget_remaining": grant_left,
            "cumulative_processed_pixels": self.cumulative_pixels,
            "pending": sorted(self.pending),
            "completed_tiles": completed,
        }

    # ------------------------------------------------------------------
    def _calib_safe(self, tile):
        try:
            return tile, calibrate_tile_band(tile, self.table, self.config), None
        except StageError as exc:
            return tile, None, exc.to_failure()
        except Exception as exc:  # 防御性隔离
            return tile, None, StageError(
                "E_CALIB_FAILED", str(exc), tile_id=tile.tile_id, band=tile.band,
                stage=STAGE_CALIB, sensor_id=tile.sensor_id).to_failure()

    def _probe_calib(self, tile):
        from .calibration import calibrate_tile_band
        from .fingerprint import content_fingerprint, derive_cache_key, sensor_coeff_fingerprint
        sensor = self.table[tile.sensor_id]
        override = tile.fp_override or {}
        upstream = {
            "tile": tile.to_dict(),
            "sensor_fingerprint": sensor_coeff_fingerprint(sensor),
            "fixed_point_bits": int(override.get("fixed_point_bits", self.config["fixed_point_bits"])),
            "fixed_point_frac_bits": int(override.get(
                "fixed_point_frac_bits", self.config["fixed_point_frac_bits"])),
            "quantization_levels": int(self.config["quantization_levels"]),
        }
        key = derive_cache_key("calib", content_fingerprint(upstream))
        return self.store.calib.get(key)

    def _run_regression(self, tiles, failures):
        ref_sensor = self.config["reference_sensor"]
        whitelist = [s for s in self.config["source_whitelist"] if s in self.table and s != ref_sensor]
        calib_index = {(rec["tile_id"], rec["band"]): rec for rec in self.store.calib.values()}
        by_scene_band: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
        for tile in tiles:
            by_scene_band[(tile.scene_id, tile.band)][tile.sensor_id] = tile.tile_id

        scene_pairs_by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for (scene, band), sensor_tiles in sorted(by_scene_band.items()):
            if ref_sensor not in sensor_tiles:
                continue
            for source in whitelist:
                if source in sensor_tiles:
                    scene_pairs_by_source[source].append(
                        (sensor_tiles[source], sensor_tiles[ref_sensor]))

        bands = sorted({t.band for t in tiles})
        live_model_keys: set[str] = set()
        models_by_id: dict[str, dict] = {}
        for source in sorted(whitelist):
            for band in bands:
                try:
                    model = solve_model(source, band, self.config, calib_index,
                                       self.table, sorted(scene_pairs_by_source[source]))
                except StageError as exc:
                    self._record_failure(exc.to_failure())
                    continue
                mid = model.model_id
                existing = next((m for m in self.store.models.values()
                                 if m["model_id"] == mid), None)
                if existing and existing["key"] == model.key:
                    self.store.reuse_log.append({
                        "stage": STAGE_REGRESS, "model_id": mid, "band": band,
                        "key": model.key, "action": "REUSED",
                    })
                    models_by_id[mid] = existing
                else:
                    if existing:
                        # 旧键模型已被 GC 或键已变更：记录失效重算并移除旧键
                        self.store.reuse_log.append({
                            "stage": STAGE_REGRESS, "model_id": mid, "band": band,
                            "key": model.key, "action": "RECOMPUTED",
                            "old_key": existing["key"],
                        })
                        self.store.models.pop(existing["key"], None)
                    models_by_id[mid] = model.to_dict()
                    self.store.models[model.key] = models_by_id[mid]
                live_model_keys.add(models_by_id[mid]["key"])

        # 对齐：每个非参考传感器瓦片，按稳定顺序
        align_tasks = sorted(
            [t for t in tiles if t.sensor_id != ref_sensor],
            key=lambda t: stable_key(t.tile_id, t.band))
        ref_records = {(rec["scene_id"], rec["band"]): rec
                       for rec in self.store.calib.values()
                       if rec["sensor_id"] == ref_sensor}
        live_align_keys: set[str] = set()
        for tile in align_tasks:
            calib = calib_index.get((tile.tile_id, tile.band))
            mid = model_id(tile.sensor_id, tile.band)
            model = models_by_id.get(mid)
            if calib is None or model is None:
                continue
            record = CalibRecord(**calib)
            try:
                alignment = align_tile(record, _model_from_dict(model), self.config, ref_records)
            except StageError as exc:
                self._record_failure(exc.to_failure())
                continue
            existing = self.store.alignments.get(alignment.key)
            if existing:
                self.store.reuse_log.append({
                    "stage": "alignment", "tile_id": tile.tile_id, "band": tile.band,
                    "key": alignment.key, "action": "REUSED",
                })
            self.store.alignments[alignment.key] = alignment.to_dict()
            live_align_keys.add(alignment.key)

        # GC：仅清理已被同 model_id 新版本替代的旧模型；对齐按内容键清理
        live_model_ids = {m["model_id"] for m in models_by_id.values()}
        stale_model_keys = [mk for mk in self.store.models
                            if self.store.models[mk]["model_id"] not in live_model_ids]
        for stale_key in stale_model_keys:
            del self.store.models[stale_key]
        stale_align_keys = [ak for ak in self.store.alignments if ak not in live_align_keys]
        for stale_key in stale_align_keys:
            del self.store.alignments[stale_key]

    def _gc_artifacts(self, tiles):
        live_calib = set()
        for tile in tiles:
            probe = self._probe_calib(tile)
            if probe is not None:
                live_calib.add(probe["key"])
        for key in [k for k in self.store.calib if k not in live_calib]:
            del self.store.calib[key]

    def _prune_processing_failures(self, ordered):
        live_tiles = {(t.tile_id, t.band) for t in ordered}
        calib_pairs = {(rec["tile_id"], rec["band"]) for rec in self.store.calib.values()}
        model_pairs = {(m["source_sensor"], m["band"]) for m in self.store.models.values()}
        aligned_pairs = {(rec["tile_id"], rec["band"]) for rec in self.store.alignments.values()}
        kept = []
        for failure in self.store.failures:
            stage = failure.get("stage")
            if stage not in (STAGE_CALIB, STAGE_REGRESS):
                kept.append(failure)
                continue
            pair = (failure.get("tile_id"), failure.get("band"))
            code = failure.get("code")
            if code == E_CALIB_BUDGET_PENDING:
                if pair in set(self.pending):
                    kept.append(failure)
                continue
            if stage == STAGE_CALIB:
                if pair in live_tiles and pair not in calib_pairs:
                    kept.append(failure)
                continue
            if stage == STAGE_REGRESS:
                source = failure.get("sensor_id")
                if failure.get("tile_id") is None:
                    if (source, failure.get("band")) not in model_pairs:
                        kept.append(failure)
                elif pair not in aligned_pairs:
                    kept.append(failure)
        self.store.failures = kept

    def _ledger_row(self, batch_id, tile, pixels, remaining, status):
        return {
            "batch_id": batch_id,
            "tile_id": tile.tile_id,
            "band": tile.band,
            "pixels": pixels,
            "cumulative_pixels": self.cumulative_pixels,
            "budget_grant_remaining": remaining,
            "status": status,
            "sort_key": stable_key(tile.tile_id, tile.band),
        }


def _model_from_dict(data: dict):
    from .models import RegressionModel
    return RegressionModel(**data)
