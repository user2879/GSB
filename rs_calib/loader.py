"""瓦片包 / 定标表 / 回归配置的加载与逐项校验层。

任何单项校验失败都以业务码写入失败清单并继续，不抛出中断整体加载。
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

from .errors import (
    E_GAIN_INVALID,
    E_OFFSET_INVALID,
    E_PIXEL_NEGATIVE,
    E_PIXEL_NOT_INTEGER,
    E_PIXEL_OVERFLOW,
    E_REG_SOURCE_EMPTY,
    E_REG_SOURCE_UNKNOWN,
    E_TILE_BAND_UNDECLARED,
    E_TILE_DUPLICATE,
    E_TILE_SENSOR_UNKNOWN,
    E_TIMESTAMP_NOT_ASCENDING,
    STAGE_LOAD,
    StageError,
)
from .models import Tile
from . import yaml_lite


def _failure(code: str, message: str, **kw) -> dict:
    return StageError(code, message, stage=STAGE_LOAD, **kw).to_failure()


def load_calibration_table(path: str) -> tuple[dict[str, dict], list[dict]]:
    failures: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    table: dict[str, dict] = {}
    for entry in raw.get("sensors", []):
        sid = entry["sensor_id"]
        bands = {}
        for band_entry in entry.get("bands", []):
            band = int(band_entry["band"])
            gain = band_entry.get("gain")
            offset = band_entry.get("offset")
            if not isinstance(gain, (int, float)) or isinstance(gain, bool) or not math.isfinite(gain) or gain <= 0:
                failures.append(_failure(E_GAIN_INVALID,
                                         f"传感器 {sid} 波段 {band} 增益非法: {gain!r}",
                                         sensor_id=sid, band=band))
                continue
            if not isinstance(offset, (int, float)) or isinstance(offset, bool) or not math.isfinite(offset):
                failures.append(_failure(E_OFFSET_INVALID,
                                         f"传感器 {sid} 波段 {band} 偏移非法: {offset!r}",
                                         sensor_id=sid, band=band))
                continue
            bands[band] = {"band": band, "gain": float(gain), "offset": float(offset)}
        bit_width = int(entry["bit_width"])
        saturation_limit = int(entry["saturation_limit"])
        table[sid] = {
            "sensor_id": sid,
            "bands": dict(sorted(bands.items())),
            "bit_width": bit_width,
            "saturation_limit": saturation_limit,
            "reference_time_ms": int(entry["reference_time_ms"]),
        }
    return table, failures


def load_regression_config(path: str, table: dict[str, dict]) -> tuple[dict, list[dict]]:
    failures: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml_lite.loads(fh.read())
    whitelist = config.get("source_whitelist") or []
    if not whitelist:
        failures.append(_failure(E_REG_SOURCE_EMPTY, "回归源传感器白名单为空"))
    for sid in whitelist:
        if sid not in table:
            failures.append(_failure(E_REG_SOURCE_UNKNOWN,
                                     f"回归源传感器 {sid} 不在定标表内", sensor_id=sid))
    ref = config.get("reference_sensor")
    if ref not in table:
        failures.append(_failure(E_REG_SOURCE_UNKNOWN,
                                 f"参考传感器 {ref} 不在定标表内", sensor_id=ref))
    # 默认值补齐，保证后续阶段字段稳定
    config.setdefault("feature_orders", [1])
    config.setdefault("initial_weights", 0.0)
    config.setdefault("regularization", 1e-6)
    config.setdefault("near_singular_rcond", 1e-10)
    config.setdefault("fixed_point_bits", 32)
    config.setdefault("fixed_point_frac_bits", 12)
    config.setdefault("quantization_levels", 4096)
    config.setdefault("pixel_budget", 10_000_000)
    config.setdefault("concurrency", 1)
    config.setdefault("snapshot_commit_interval", 1)
    return config, failures


def load_tile_pack(paths: list[str], table: dict[str, dict]) -> tuple[list[Tile], list[dict]]:
    """逐行解析瓦片包 JSONL 并执行全部加载期校验。"""
    failures: list[dict] = []
    tiles: list[Tile] = []
    seen_ids: set[str] = set()
    last_ts_by_sensor: dict[str, int] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    failures.append(_failure("E_TILE_JSON_INVALID",
                                             f"{os.path.basename(path)}:{lineno} JSON 解析失败: {exc}"))
                    continue
                tile_id = str(rec["tile_id"])
                sensor_id = str(rec["sensor_id"])
                band = int(rec["band"])
                counts = rec.get("counts", [])
                flags = rec.get("saturated_flags", [False] * len(counts))
                valid = True

                if tile_id in seen_ids:
                    failures.append(_failure(E_TILE_DUPLICATE,
                                             f"瓦片 ID 重复: {tile_id}", tile_id=tile_id,
                                             sensor_id=sensor_id, band=band))
                    valid = False
                seen_ids.add(tile_id)

                if sensor_id not in table:
                    failures.append(_failure(E_TILE_SENSOR_UNKNOWN,
                                             f"传感器 {sensor_id} 不在定标表内",
                                             tile_id=tile_id, sensor_id=sensor_id, band=band))
                    valid = False
                elif band not in table[sensor_id]["bands"]:
                    failures.append(_failure(E_TILE_BAND_UNDECLARED,
                                             f"波段 {band} 未在传感器 {sensor_id} 声明",
                                             tile_id=tile_id, sensor_id=sensor_id, band=band))
                    valid = False

                clean_counts: list[int] = []
                for idx, value in enumerate(counts):
                    if isinstance(value, bool) or not isinstance(value, int):
                        failures.append(_failure(E_PIXEL_NOT_INTEGER,
                                                 f"像素 {idx} 非整数: {value!r}",
                                                 tile_id=tile_id, band=band, sensor_id=sensor_id))
                        valid = False
                        continue
                    if value < 0:
                        failures.append(_failure(E_PIXEL_NEGATIVE,
                                                 f"像素 {idx} 计数为负: {value}",
                                                 tile_id=tile_id, band=band, sensor_id=sensor_id))
                        valid = False
                    elif sensor_id in table:
                        limit = (1 << table[sensor_id]["bit_width"]) - 1
                        if value > limit:
                            failures.append(_failure(E_PIXEL_OVERFLOW,
                                                     f"像素 {idx} 计数 {value} 超过位宽上限 {limit}",
                                                     tile_id=tile_id, band=band, sensor_id=sensor_id))
                            valid = False
                    clean_counts.append(value if isinstance(value, int) and not isinstance(value, bool) else 0)

                ts = int(rec["timestamp_ms"])
                if sensor_id in last_ts_by_sensor and ts <= last_ts_by_sensor[sensor_id]:
                    failures.append(_failure(E_TIMESTAMP_NOT_ASCENDING,
                                             f"传感器 {sensor_id} 时间戳 {ts} 未严格递增",
                                             tile_id=tile_id, band=band, sensor_id=sensor_id))
                    valid = False
                last_ts_by_sensor.setdefault(sensor_id, ts)
                if sensor_id not in last_ts_by_sensor or ts > last_ts_by_sensor[sensor_id]:
                    last_ts_by_sensor[sensor_id] = ts

                if not valid:
                    continue
                rows = int(rec.get("rows", 0)) or len(clean_counts)
                cols = int(rec.get("cols", 0)) or 1
                tiles.append(Tile(
                    tile_id=tile_id,
                    scene_id=str(rec.get("scene_id", tile_id.split("-t")[0])),
                    sensor_id=sensor_id,
                    band=band,
                    timestamp_ms=ts,
                    counts=tuple(clean_counts),
                    rows=rows,
                    cols=cols,
                    saturated_flags=tuple(bool(x) for x in flags),
                    fp_override=rec.get("_fp_override"),
                ))
    return tiles, failures
