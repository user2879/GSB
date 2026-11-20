"""确定性样本段定义与查询文件生成（使用 NumPy 构造与校验）。

生成两个段 ``seg_a``、``seg_b``，每段 3 个行组、4 列，块级混合使用
RAW/DICT/RLE 三种编码与压缩标记；随机数固定种子，重复生成逐字节一致。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .codecs import DICT, RAW, RLE
from .schema import SegmentSpec, write_segment_spec

ROW_GROUP_SIZE = 12
ROW_GROUP_COUNTS = (ROW_GROUP_SIZE, ROW_GROUP_SIZE, ROW_GROUP_SIZE)

ENCODING_PLAN = {
    # (段, 列): 块级编码，按行组顺序
    ("seg_a", "order_id"): [RAW, RAW, RAW],
    ("seg_a", "region"): [DICT, DICT, RLE],
    ("seg_a", "amount"): [RAW, DICT, RAW],
    ("seg_a", "priority"): [RLE, RLE, DICT],
    ("seg_b", "order_id"): [RAW, RAW, RAW],
    ("seg_b", "region"): [DICT, RLE, DICT],
    ("seg_b", "amount"): [DICT, RAW, RAW],
    ("seg_b", "priority"): [RLE, DICT, RLE],
}

REGIONS = ("north", "south", "east", "west")
PRIORITIES = (1, 2, 3)


def _numpy_columns(segment_id: str, base: int) -> dict[str, list[list]]:
    rng = np.random.default_rng(20261121 + base)
    columns: dict[str, list[list]] = {}
    offset = 0
    for group_index, count in enumerate(ROW_GROUP_COUNTS):
        order_ids = (np.arange(count, dtype=np.int64)
                     + base * 1000 + offset).tolist()
        region_idx = rng.integers(0, len(REGIONS), size=count)
        # 让部分行组出现常量区域，便于 RLE 与区域图跳过演示
        if group_index == 1:
            region_idx = np.full(count, (base + group_index) % len(REGIONS))
        regions = [REGIONS[int(i)] for i in region_idx]
        amounts = (rng.integers(10, 200, size=count, dtype=np.int64)
                   + base * 7).tolist()
        if group_index == 2:
            priorities = np.full(count, 2, dtype=np.int64).tolist()
        else:
            priorities = rng.integers(1, 4, size=count,
                                      dtype=np.int64).tolist()
        for name, values in (("order_id", order_ids), ("region", regions),
                             ("amount", amounts),
                             ("priority", priorities)):
            columns.setdefault(name, []).append(values)
        offset += count
    return columns


def build_spec(segment_id: str, base: int) -> SegmentSpec:
    raw_columns = _numpy_columns(segment_id, base)
    column_meta = [
        {"name": "order_id", "type": "int64", "encoding": RAW,
         "compressed": False},
        {"name": "region", "type": "string", "encoding": DICT,
         "compressed": True},
        {"name": "amount", "type": "int64", "encoding": RAW,
         "compressed": True},
        {"name": "priority", "type": "int64", "encoding": RLE,
         "compressed": False},
    ]
    row_groups = []
    start = 0
    blocks = []
    for group_index, count in enumerate(ROW_GROUP_COUNTS):
        row_groups.append({"index": group_index, "start": start,
                           "end": start + count})
        for meta in column_meta:
            name = meta["name"]
            plan_key = (segment_id, name)
            encoding = ENCODING_PLAN.get(
                plan_key, ENCODING_PLAN[("seg_a", name)])[group_index]
            blocks.append({
                "row_group": group_index, "column": name,
                "block_index": 0, "encoding": encoding,
                "compressed": meta["compressed"],
                "values": raw_columns[name][group_index],
            })
        start += count
    raw = {"segment_id": segment_id, "format_version": 1,
           "columns": column_meta, "row_groups": row_groups,
           "blocks": blocks}
    from .schema import parse_segment_spec
    return parse_segment_spec(raw)


def generate(sample_dir: str | Path) -> list[Path]:
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ordinal, segment_id in enumerate(("seg_a", "seg_b")):
        spec = build_spec(segment_id, ordinal + 1)
        target = sample_dir / f"segment_{segment_id}.json"
        write_segment_spec(spec, target)
        paths.append(target)

    queries = {
        "q_full.json": {
            "name": "全量扫描（仅列裁剪）",
            "sql": "SELECT order_id, region, amount, priority FROM orders"},
        "q_eq_in.json": {
            "name": "等值 + IN 下推",
            "sql": "SELECT order_id, amount FROM orders "
                   "WHERE region = 'north' AND priority IN (1, 2)"},
        "q_range.json": {
            "name": "范围 AND 组合下推",
            "sql": "SELECT order_id, region FROM orders "
                   "WHERE amount >= 50 AND amount <= 150 AND priority = 2"},
        "q_paren.json": {
            "name": "括号/空白等价性查询",
            "sql": "SELECT order_id, region FROM orders "
                   "WHERE  ( ( amount >= 50 ) AND ( amount <= 150 ) ) "
                   "AND priority = 2"},
    }
    for name, payload in queries.items():
        (sample_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8")
    return paths
