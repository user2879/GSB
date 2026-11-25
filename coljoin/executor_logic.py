"""执行器侧的确定性连接、过滤与分组聚合运算。

核心不变量：``index_map[bg]`` 对连接链上每个块组保存与当前中间结果
等长的物理行号数组，第 i 个位置共同描述中间结果的第 i 行。
新块组通过等值谓词把现有中间结果展开（一对多），再对所有块组按位置对齐。
聚合为整数算式，avg 在合并阶段 floor 除法，跨分片可唯一复核。
"""

from typing import Any, Dict, List

import numpy as np

from .errors import SpecError
from .model import QuerySpec
from .storage import load_generated_specs, read_column

_FILTER_CODE = {
    "eq": np.equal,
    "ne": np.not_equal,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
}


def _group_spec(data_dir: str, bg_id: str):
    for group in load_generated_specs(data_dir):
        if group.bg_id == bg_id:
            return group
    raise SpecError("执行器找不到块组", object_id=bg_id, field="bg")


def scan_indexes(data_dir: str, bg_id: str, filters: List[Dict[str, Any]]) -> np.ndarray:
    group = _group_spec(data_dir, bg_id)
    mask = np.ones(group.rows, dtype=bool)
    for pred in filters:
        values, bitmap = read_column(data_dir, bg_id, pred["column"])
        if pred["op"] == "is_null":
            condition = bitmap == 1
        elif pred["op"] == "not_null":
            condition = bitmap == 0
        else:
            condition = _FILTER_CODE[pred["op"]](values, pred["value"])
            condition = condition & (bitmap == 0)
        mask = mask & condition
    return np.nonzero(mask)[0].astype(np.int64)


def _predicate_columns(pred, right_bg):
    if pred["right_bg"] == right_bg:
        return pred["left_bg"], pred["left_col"], pred["right_col"]
    if pred["left_bg"] == right_bg:
        return pred["right_bg"], pred["right_col"], pred["left_col"]
    raise SpecError("连接谓词未覆盖新加入块组", object_id=right_bg, field="join")


def execute_plan_on_batch(
    data_dir: str,
    plan: Dict[str, Any],
    query: QuerySpec,
    batch: Dict[str, Any],
) -> Dict[str, Any]:
    steps = plan["body"]["steps"]
    scan_filters = {
        step["bg"]: step["filters"] for step in steps if step["op"] == "scan"
    }
    order = plan["body"]["order"]
    driver = order[0]
    start, end = int(batch["start"]), int(batch["end"])
    driver_rows = scan_indexes(data_dir, driver, scan_filters[driver])
    driver_rows = driver_rows[(driver_rows >= start) & (driver_rows < end)]
    index_map: Dict[str, np.ndarray] = {driver: driver_rows}

    for step in steps:
        if step["op"] != "join":
            continue
        new_bg = step["new_bg"]
        right_scan = scan_indexes(data_dir, new_bg, scan_filters[new_bg])
        result_positions: List[int] = []
        result_right_rows: List[int] = []
        intermediate_size = len(next(iter(index_map.values())))
        for position in range(intermediate_size):
            base_matches = [int(row) for row in right_scan]
            satisfied = True
            for pred in step["predicates"]:
                left_bg, lcol, rcol = _predicate_columns(pred, new_bg)
                left_values, _ = read_column(data_dir, left_bg, lcol)
                right_values, _ = read_column(data_dir, new_bg, rcol)
                key = left_values[int(index_map[left_bg][position])].item()
                base_matches = [
                    row
                    for row in base_matches
                    if right_values[int(row)].item() == key
                ]
                if not base_matches:
                    satisfied = False
                    break
            if satisfied:
                for right_row in base_matches:
                    result_positions.append(position)
                    result_right_rows.append(right_row)
        positions = np.array(result_positions, dtype=np.int64)
        aligned = {
            bg_id: rows[positions] if positions.size else rows[np.array([], dtype=np.int64)]
            for bg_id, rows in index_map.items()
        }
        aligned[new_bg] = np.array(result_right_rows, dtype=np.int64)
        index_map = aligned

    partial = _aggregate(data_dir, query, index_map)
    return merge_partial_aggregates([partial], query)


def _aggregate(data_dir, query: QuerySpec, index_map: Dict[str, np.ndarray]):
    lengths = {bg_id: len(rows) for bg_id, rows in index_map.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"连接链索引未对齐: {lengths}")
    row_count = len(next(iter(index_map.values())))
    if query.group_by:
        ref = query.group_by[0]
        values, _ = read_column(data_dir, ref["bg"], ref["column"])
        group_keys = np.array(
            [str(value) for value in values[index_map[ref["bg"]]]], dtype=object
        )
    else:
        group_keys = np.array(["*"] * row_count, dtype=object)

    result: Dict[str, Dict[str, Any]] = {}
    for key in np.unique(group_keys):
        member = group_keys == key
        bucket: Dict[str, Any] = {"count": int(member.sum())}
        for agg in query.aggregates:
            rows = index_map[agg.bg_id][member]
            if agg.func == "count":
                bucket[agg.alias] = int(len(rows))
                continue
            values, bitmap = read_column(data_dir, agg.bg_id, agg.column)
            valid = values[rows][bitmap[rows] == 0]
            if agg.func == "sum":
                bucket[agg.alias] = int(valid.sum()) if valid.size else 0
            elif agg.func == "avg":
                bucket[agg.alias + "_sum"] = int(valid.sum()) if valid.size else 0
                bucket[agg.alias + "_cnt"] = int(valid.size)
            elif agg.func == "min":
                bucket[agg.alias] = int(valid.min()) if valid.size else None
            elif agg.func == "max":
                bucket[agg.alias] = int(valid.max()) if valid.size else None
        result[str(key)] = bucket
    return {"groups": result}


def merge_partial_aggregates(partials: List[Dict[str, Any]], query: QuerySpec):
    merged: Dict[str, Dict[str, Any]] = {}
    for partial in partials:
        for key, bucket in partial["groups"].items():
            target = merged.setdefault(key, {"count": 0})
            target["count"] += bucket["count"]
            for agg in query.aggregates:
                if agg.func == "count":
                    target[agg.alias] = target.get(agg.alias, 0) + bucket.get(agg.alias, 0)
                elif agg.func == "sum":
                    target[agg.alias] = target.get(agg.alias, 0) + bucket.get(agg.alias, 0)
                elif agg.func == "avg":
                    target[agg.alias + "_sum"] = (
                        target.get(agg.alias + "_sum", 0)
                        + bucket.get(agg.alias + "_sum", 0)
                    )
                    target[agg.alias + "_cnt"] = (
                        target.get(agg.alias + "_cnt", 0)
                        + bucket.get(agg.alias + "_cnt", 0)
                    )
                elif agg.func in ("min", "max"):
                    value = bucket.get(agg.alias)
                    if value is None:
                        continue
                    current = target.get(agg.alias)
                    if current is None:
                        target[agg.alias] = value
                    elif agg.func == "min":
                        target[agg.alias] = min(current, value)
                    else:
                        target[agg.alias] = max(current, value)
    for agg in query.aggregates:
        if agg.func != "avg":
            continue
        for bucket in merged.values():
            denominator = bucket.get(agg.alias + "_cnt", 0)
            numerator = bucket.get(agg.alias + "_sum", 0)
            bucket[agg.alias] = numerator // denominator if denominator else 0
    return {"groups": merged}
