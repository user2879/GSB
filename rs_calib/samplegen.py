"""固定种子确定性样本生成器：瓦片包 JSONL / 定标表 / 回归 YAML / 编辑计划 / 注入配置。"""
from __future__ import annotations

import json
import os

import numpy as np


SENSORS = ["S1", "S2", "S3"]
REFERENCE_SENSOR = "S1"
BANDS = [1, 2, 3]
ROWS, COLS = 4, 4
N_SCENES = 6
BASE_TS = 1_700_000_000_000


def _calibration_table(rng: np.random.Generator) -> dict:
    sensors = []
    for idx, sid in enumerate(SENSORS):
        bit_width = 12 if sid != "S3" else 10
        bands = []
        for band in BANDS:
            gain = float(0.02 + 0.004 * idx + 0.001 * band)
            offset = float(-0.5 + 0.1 * idx + 0.02 * band)
            bands.append({"band": band, "gain": round(gain, 6),
                          "offset": round(offset, 6)})
        sensors.append({
            "sensor_id": sid,
            "bands": bands,
            "bit_width": bit_width,
            "saturation_limit": (1 << bit_width) - 1,
            "reference_time_ms": BASE_TS,
        })
    return {"schema_version": 1, "sensors": sensors}


def _true_radiance(rng: np.random.Generator, scene: int, band: int) -> np.ndarray:
    grid = np.arange(ROWS * COLS, dtype=np.float64).reshape(ROWS, COLS)
    base = 20.0 + 4.0 * scene + 6.0 * band + grid * (0.3 + 0.05 * band)
    return base + rng.normal(0.0, 0.4, size=(ROWS, COLS))


def generate_sample_dataset(out_dir: str, seed: int = 20261122) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    table = _calibration_table(rng)
    with open(os.path.join(out_dir, "calibration_table.json"), "w", encoding="utf-8") as fh:
        json.dump(table, fh, ensure_ascii=False, indent=2, sort_keys=True)

    coeff = {s["sensor_id"]: s for s in table["sensors"]}
    packs: dict[str, list[dict]] = {"tiles_batch0.jsonl": []}
    tile_counter = 0
    tile_index: dict[str, dict] = {}

    for scene in range(N_SCENES):
        truth = {band: _true_radiance(rng, scene, band) for band in BANDS}
        for sid in SENSORS:
            sensor = coeff[sid]
            for band in BANDS:
                tile_counter += 1
                tile_id = f"tile-{tile_counter:04d}"
                gain = sensor["bands"][band - 1]["gain"]
                offset = sensor["bands"][band - 1]["offset"]
                physical = truth[band] * (1.0 if sid == REFERENCE_SENSOR
                                          else 1.0 + 0.15 * SENSORS.index(sid))
                physical += rng.normal(0.0, 0.25, size=(ROWS, COLS))
                counts = np.rint((physical - offset) / gain).astype(np.int64)
                limit = sensor["saturation_limit"]
                counts = np.clip(counts, 0, limit)
                # 确定性注入两个饱和像素
                saturated = np.zeros((ROWS, COLS), dtype=bool)
                saturated[0, 0] = True
                saturated[ROWS - 1, COLS - 1] = True
                counts[0, 0] = limit
                counts[ROWS - 1, COLS - 1] = limit
                ts = BASE_TS + scene * 60_000 + SENSORS.index(sid) * 1_000 + band
                record = {
                    "tile_id": tile_id,
                    "scene_id": f"scene-{scene:02d}",
                    "sensor_id": sid,
                    "band": band,
                    "rows": ROWS,
                    "cols": COLS,
                    "timestamp_ms": int(ts),
                    "counts": [int(v) for v in counts.flatten().tolist()],
                    "saturated_flags": [bool(v) for v in saturated.flatten().tolist()],
                }
                packs["tiles_batch0.jsonl"].append(record)
                tile_index[tile_id] = record

    # 批次 1/2 的新瓦片（场景 6、7）
    extra_scenes = [N_SCENES, N_SCENES + 1]
    next_counter = tile_counter
    plan_batches = []
    for batch_no, scene in enumerate(extra_scenes, start=1):
        pack_name = f"tiles_batch{batch_no}.jsonl"
        packs[pack_name] = []
        add_records = []
        truth = {band: _true_radiance(rng, scene, band) for band in BANDS}
        for sid in SENSORS:
            sensor = coeff[sid]
            for band in BANDS:
                next_counter += 1
                tile_id = f"tile-{next_counter:04d}"
                gain = sensor["bands"][band - 1]["gain"]
                offset = sensor["bands"][band - 1]["offset"]
                physical = truth[band] * (1.0 if sid == REFERENCE_SENSOR
                                          else 1.0 + 0.15 * SENSORS.index(sid))
                counts = np.rint((physical - offset) / gain).astype(np.int64)
                limit = sensor["saturation_limit"]
                counts = np.clip(counts, 0, limit)
                saturated = np.zeros((ROWS, COLS), dtype=bool)
                saturated[1, 1] = True
                counts[1, 1] = limit
                ts = BASE_TS + scene * 60_000 + SENSORS.index(sid) * 1_000 + band
                record = {
                    "tile_id": tile_id,
                    "scene_id": f"scene-{scene:02d}",
                    "sensor_id": sid,
                    "band": band,
                    "rows": ROWS, "cols": COLS,
                    "timestamp_ms": int(ts),
                    "counts": [int(v) for v in counts.flatten().tolist()],
                    "saturated_flags": [bool(v) for v in saturated.flatten().tolist()],
                }
                packs[pack_name].append(record)
                add_records.append(record)

        actions = [{"type": "add_tiles", "tiles": add_records}]
        if batch_no == 1:
            # 回填修正第一个 S2 瓦片的计数
            target = next(rec for rec in packs["tiles_batch0.jsonl"]
                          if rec["sensor_id"] == "S2" and rec["band"] == 1)
            revised = list(target["counts"])
            revised[5] = max(0, revised[5] - 12)
            actions.append({
                "type": "backfill_counts",
                "tile_id": target["tile_id"],
                "timestamp_ms": target["timestamp_ms"] + 500,
                "counts": revised,
                "saturated_flags": target["saturated_flags"],
            })
            # S2 系数变更 -> 精确失效
            s2 = coeff["S2"]
            actions.append({
                "type": "update_sensor_coeff",
                "sensor_id": "S2",
                "bands": [
                    {"band": 1,
                     "gain": round(s2["bands"][0]["gain"] * 1.08, 6),
                     "offset": round(s2["bands"][0]["offset"] - 0.03, 6)},
                    {"band": 2,
                     "gain": s2["bands"][1]["gain"],
                     "offset": s2["bands"][1]["offset"]},
                    {"band": 3,
                     "gain": s2["bands"][2]["gain"],
                     "offset": s2["bands"][2]["offset"]},
                ],
            })
        if batch_no == 2:
            actions.append({
                "type": "update_regression_config",
                "values": {"regularization": 2.5e-6},
            })
        plan_batches.append({"batch_id": batch_no, "actions": actions})

    for name, rows in packs.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    config = {
        "reference_sensor": REFERENCE_SENSOR,
        "source_whitelist": ["S2", "S3"],
        "feature_orders": [0, 1, 2],
        "initial_weights": 0.0,
        "regularization": 1.0e-6,
        "near_singular_rcond": 1.0e-10,
        "fixed_point_bits": 32,
        "fixed_point_frac_bits": 12,
        "quantization_levels": 4096,
        "pixel_budget": 800,
        "concurrency": 3,
        "snapshot_commit_interval": 1,
    }
    with open(os.path.join(out_dir, "regression_config.yaml"), "w", encoding="utf-8") as fh:
        fh.write(_dump_yaml(config))

    with open(os.path.join(out_dir, "edit_plan.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema_version": 1, "batches": plan_batches}, fh,
                  ensure_ascii=False, indent=2, sort_keys=True)

    backfill_target = next(rec for rec in packs["tiles_batch0.jsonl"]
                           if rec["sensor_id"] == "S2" and rec["band"] == 1)["tile_id"]
    overflow_target = packs["tiles_batch0.jsonl"][0]["tile_id"]
    unknown_band_target = packs["tiles_batch0.jsonl"][1]["tile_id"]
    injections = {
        "schema_version": 1,
        "injections": [
            {"injection_id": "INJ-001", "kind": "count_overflow",
             "target": {"pack": "tiles_batch0.jsonl", "tile_id": overflow_target, "delta": 5}},
            {"injection_id": "INJ-002", "kind": "unknown_band",
             "target": {"pack": "tiles_batch0.jsonl", "tile_id": unknown_band_target, "band": 99}},
            {"injection_id": "INJ-003", "kind": "backfill_timestamp_regress",
             "target": {"batch_id": 1, "tile_id": backfill_target,
                        "timestamp_ms": 1}},
            {"injection_id": "INJ-004", "kind": "regression_source_missing",
             "target": {"sensor_id": "S9"}},
            {"injection_id": "INJ-005", "kind": "fixedpoint_overflow",
             "target": {"pack": "tiles_batch0.jsonl", "tile_id": "tile-0010",
                        "fixed_point_bits": 8, "fixed_point_frac_bits": 4}},
        ],
    }
    with open(os.path.join(out_dir, "injection_config.json"), "w", encoding="utf-8") as fh:
        json.dump(injections, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return {"out_dir": out_dir, "tile_count": next_counter, "scenes": N_SCENES + 2}


def _dump_yaml(config: dict) -> str:
    lines = ["# 跨传感器回归对齐配置（固定种子生成）"]
    for key in sorted(config):
        value = config[key]
        lines.append(f"{key}: {_scalar(value)}")
    return "\n".join(lines) + "\n"


def _scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(v) for v in value) + "]"
    return str(value)
