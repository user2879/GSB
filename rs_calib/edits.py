"""增量编辑计划应用：瓦片追加、计数回填修正、定标系数/回归配置变更。"""
from __future__ import annotations

import copy
import json

from .errors import E_BACKFILL_TIMESTAMP_ORDER, STAGE_LOAD, StageError
from .models import Tile


class EditEngine:
    def __init__(self, table: dict[str, dict], config: dict, tiles: list[Tile]):
        self.table = table
        self.config = config
        self.tiles: dict[str, Tile] = {t.tile_id: t for t in tiles}
        # 每个传感器已“完成”过的最大时间戳，用于回填时序判定（确定性、数据驱动）
        self.seen_max_ts: dict[str, int] = {}
        for tile in sorted(tiles, key=lambda t: (t.sensor_id, t.timestamp_ms, t.tile_id)):
            self.seen_max_ts[tile.sensor_id] = max(
                self.seen_max_ts.get(tile.sensor_id, tile.timestamp_ms), tile.timestamp_ms)

    def tile_list(self) -> list[Tile]:
        return [self.tiles[k] for k in sorted(self.tiles)]

    def mark_completed(self, tiles: list[Tile]) -> None:
        for tile in tiles:
            self.seen_max_ts[tile.sensor_id] = max(
                self.seen_max_ts.get(tile.sensor_id, tile.timestamp_ms), tile.timestamp_ms)

    def apply_batch(self, batch: dict) -> list[dict]:
        failures: list[dict] = []
        for action in batch.get("actions", []):
            kind = action["type"]
            if kind == "add_tiles":
                for rec in action["tiles"]:
                    tile = _tile_from_rec(rec)
                    self.tiles[tile.tile_id] = tile
                    self.seen_max_ts[tile.sensor_id] = max(
                        self.seen_max_ts.get(tile.sensor_id, tile.timestamp_ms), tile.timestamp_ms)
            elif kind == "backfill_counts":
                tile_id = action["tile_id"]
                if tile_id not in self.tiles:
                    failures.append(StageError(
                        "E_BACKFILL_TILE_MISSING", f"回填目标瓦片不存在: {tile_id}",
                        tile_id=tile_id, stage=STAGE_LOAD).to_failure())
                    continue
                old = self.tiles[tile_id]
                new_ts = int(action.get("timestamp_ms", old.timestamp_ms))
                if new_ts < old.timestamp_ms:
                    failures.append(StageError(
                        E_BACKFILL_TIMESTAMP_ORDER,
                        f"回填瓦片 {tile_id} 时间戳 {new_ts} 早于该瓦片已完成游标 "
                        f"{old.timestamp_ms}",
                        tile_id=tile_id, band=old.band, stage=STAGE_LOAD,
                        sensor_id=old.sensor_id).to_failure())
                updated = Tile(
                    tile_id=old.tile_id, scene_id=old.scene_id, sensor_id=old.sensor_id,
                    band=old.band, timestamp_ms=new_ts,
                    counts=tuple(int(v) for v in action["counts"]),
                    rows=old.rows, cols=old.cols,
                    saturated_flags=tuple(bool(v) for v in action.get(
                        "saturated_flags", old.saturated_flags)),
                    fp_override=old.fp_override,
                )
                self.tiles[tile_id] = updated
            elif kind == "update_sensor_coeff":
                sid = action["sensor_id"]
                entry = self.table[sid]
                entry = copy.deepcopy(entry)
                if "saturation_limit" in action:
                    entry["saturation_limit"] = int(action["saturation_limit"])
                if "bit_width" in action:
                    entry["bit_width"] = int(action["bit_width"])
                if "reference_time_ms" in action:
                    entry["reference_time_ms"] = int(action["reference_time_ms"])
                for band_rec in action.get("bands", []):
                    band = int(band_rec["band"])
                    entry["bands"][band] = {
                        "band": band,
                        "gain": float(band_rec["gain"]),
                        "offset": float(band_rec["offset"]),
                    }
                self.table[sid] = entry
            elif kind == "update_regression_config":
                for key, value in action.get("values", {}).items():
                    self.config[key] = value
            else:
                failures.append(StageError(
                    "E_EDIT_UNKNOWN", f"未知编辑类型: {kind}", stage=STAGE_LOAD).to_failure())
        return failures


def _tile_from_rec(rec: dict) -> Tile:
    counts = [int(v) for v in rec["counts"]]
    flags = [bool(v) for v in rec.get("saturated_flags", [False] * len(counts))]
    return Tile(
        tile_id=str(rec["tile_id"]),
        scene_id=str(rec.get("scene_id", str(rec["tile_id"]).split("-t")[0])),
        sensor_id=str(rec["sensor_id"]),
        band=int(rec["band"]),
        timestamp_ms=int(rec["timestamp_ms"]),
        counts=tuple(counts),
        rows=int(rec.get("rows", len(counts))),
        cols=int(rec.get("cols", 1)),
        saturated_flags=tuple(flags),
        fp_override=rec.get("_fp_override"),
    )


def load_edit_plan(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        plan = json.load(fh)
    return sorted(plan["batches"], key=lambda b: int(b["batch_id"]))
