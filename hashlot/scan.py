"""列式扫描、稳定哈希分区映射与热点子分区再分发规划。

扫描输出按“源表原始行号升序”排列，保证不依赖任何字典/集合遍历顺序。
"""

from dataclasses import dataclass

import numpy as np

from .hashing import (
    RULE_NAME,
    encode_group_columns,
    hash_group_key,
    map_partition,
    map_subpartition,
)

_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}


@dataclass(frozen=True)
class ScannedRow:
    row_index: int
    key_encoded: tuple
    key_hash: int
    partition: int
    subpartition: int = -1
    values: tuple = ()


@dataclass
class ScanPlan:
    rows: list
    key_to_partition: dict
    partition_row_counts: list
    hot_partitions: tuple
    subpartition_assignments: dict
    partition_key_row_counts: dict


def _column_array(column):
    if column.dtype == "int64":
        return np.array(column.values, dtype=np.int64)
    return np.array(column.values, dtype=np.float64)


def apply_filters(spec):
    source = next(t for t in spec.tables if t.name == spec.query.source_table)
    n = source.row_count
    alive = np.ones(n, dtype=bool)
    for flt in spec.query.filters:
        table = next(t for t in spec.tables if t.name == flt.table)
        column = table.columns[flt.column]
        arr = _column_array(column)
        present = np.array(column.bitmap, dtype=bool)
        keep = _OPS[flt.op](arr, flt.value)
        alive = alive & present & keep
    return source, [int(i) for i in np.nonzero(alive)[0]]


def build_scan_plan(spec, redistribute: bool = True):
    source, alive_rows = apply_filters(spec)
    pc = spec.topology.partition_count
    rows = []
    key_to_partition = {}
    counts = [0] * pc
    key_rows = {p: {} for p in range(pc)}
    for row_index in alive_rows:
        encoded = encode_group_columns(source.columns, row_index,
                                       spec.query.group_keys, spec.scale)
        key_hash = hash_group_key(encoded)
        partition = map_partition(key_hash, pc)
        values = tuple(
            None if agg[0] == "count"
            else (None if source.columns[agg[1]].bitmap[row_index] == 0
                  else source.columns[agg[1]].values[row_index])
            for agg in spec.query.aggregates
        )
        rows.append(ScannedRow(row_index, encoded, key_hash, partition, -1, values))
        key_to_partition.setdefault(encoded, partition)
        counts[partition] += 1
        key_rows[partition].setdefault(encoded, 0)
        key_rows[partition][encoded] += 1

    hot = []
    sub_assignments = {}
    if spec.hotspot.enabled and redistribute:
        sub_count = spec.hotspot.subpartition_count
        for partition in range(pc):
            if counts[partition] >= spec.hotspot.row_threshold:
                hot.append(partition)
                sub_assignments[partition] = {}
                for key, row_total in key_rows[partition].items():
                    key_hash = hash_group_key(key)
                    sub = map_subpartition(key_hash, sub_count)
                    sub_assignments[partition][key] = sub
        for row in rows:
            if row.partition in sub_assignments:
                object.__setattr__(
                    row, "subpartition",
                    sub_assignments[row.partition][row.key_encoded],
                )

    return ScanPlan(
        rows=rows,
        key_to_partition=key_to_partition,
        partition_row_counts=counts,
        hot_partitions=tuple(hot),
        subpartition_assignments=sub_assignments,
        partition_key_row_counts=key_rows,
    )


def mapping_summary(plan, spec):
    """生成与字典遍历顺序无关的分组键 -> 分区映射摘要。"""
    items = sorted(plan.key_to_partition.items(),
                   key=lambda kv: (kv[1], kv[0]))
    return {
        "hash_rule": RULE_NAME,
        "partition_count": spec.topology.partition_count,
        "group_key_columns": list(spec.query.group_keys),
        "entries": [
            {
                "key_encoded_hex": b"".join(
                    bytes((tag,)) + payload for tag, payload in key
                ).hex(),
                "partition": partition,
            }
            for key, partition in items
        ],
        "partition_row_distribution": list(plan.partition_row_counts),
        "hot_partitions": list(plan.hot_partitions),
    }
