"""异常注入：在加载/编辑前对样本做确定性变异，注入项编号进入诊断报告。"""
from __future__ import annotations

import copy
import json

from .errors import (
    E_FIXEDPOINT_OVERFLOW,
    E_PIXEL_OVERFLOW,
    E_REG_SOURCE_UNKNOWN,
    E_TILE_BAND_UNDECLARED,
)

INJECTION_CODE_MAP = {
    "count_overflow": E_PIXEL_OVERFLOW,
    "unknown_band": E_TILE_BAND_UNDECLARED,
    "backfill_timestamp_regress": "E_BACKFILL_TIMESTAMP_ORDER",
    "regression_source_missing": E_REG_SOURCE_UNKNOWN,
    "fixedpoint_overflow": E_FIXEDPOINT_OVERFLOW,
}


def load_injection_config(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["injections"]


def apply_injections(raw_packs: dict[str, list[dict]], table: dict, config: dict,
                     plan_batches: list[dict], injections: list[dict]):
    """返回变异后的副本，并回填每个注入项预期命中的业务码。"""
    packs = copy.deepcopy(raw_packs)
    table = copy.deepcopy(table)
    config = copy.deepcopy(config)
    plan_batches = copy.deepcopy(plan_batches)
    hits = []
    for item in injections:
        iid = item["injection_id"]
        kind = item["kind"]
        target = item.get("target", {})
        expected = INJECTION_CODE_MAP[kind]
        rec = {"injection_id": iid, "kind": kind, "expected_code": expected, "hit": False}
        if kind == "count_overflow":
            line = _find_line(packs, target["pack"], target["tile_id"])
            limit = (1 << table[line["sensor_id"]]["bit_width"]) - 1
            line["counts"][0] = limit + int(target.get("delta", 1))
        elif kind == "unknown_band":
            line = _find_line(packs, target["pack"], target["tile_id"])
            line["band"] = int(target.get("band", 999))
        elif kind == "backfill_timestamp_regress":
            batch = next(b for b in plan_batches if int(b["batch_id"]) == int(target["batch_id"]))
            for action in batch["actions"]:
                if action.get("type") == "backfill_counts" and action.get("tile_id") == target["tile_id"]:
                    action["timestamp_ms"] = int(target["timestamp_ms"])
        elif kind == "regression_source_missing":
            config["source_whitelist"] = config["source_whitelist"] + [target["sensor_id"]]
        elif kind == "fixedpoint_overflow":
            line = _find_line(packs, target["pack"], target["tile_id"])
            # 仅对目标瓦片施加窄定点位宽，保证正常瓦片路径不受影响
            line["_fp_override"] = {
                "fixed_point_bits": int(target.get("fixed_point_bits", 8)),
                "fixed_point_frac_bits": int(target.get("fixed_point_frac_bits", 4)),
            }
        hits.append(rec)
    return packs, table, config, plan_batches, hits


def mark_hits(hits: list[dict], failures: list[dict]) -> list[dict]:
    by_code: dict[str, list[dict]] = {}
    for failure in failures:
        by_code.setdefault(failure["code"], []).append(failure)
    for hit in hits:
        matched = by_code.get(hit["expected_code"])
        if matched:
            hit["hit"] = True
            hit["matched_failures"] = sorted(
                f"{f.get('tile_id', '-')}#b{f.get('band', '-')}#{f.get('code')}" for f in matched)
    return hits


def _find_line(packs: dict[str, list[dict]], pack_name: str, tile_id: str) -> dict:
    for line in packs[pack_name]:
        if line["tile_id"] == tile_id:
            return line
    raise KeyError(f"注入目标不存在: {pack_name}/{tile_id}")
