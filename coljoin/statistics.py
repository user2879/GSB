"""统计口径版本、选择率与基数估算（纯整数/定点可复核）。

统计口径版本由全部块组直方图与 NDV 的规范化摘要唯一确定，
直方图或块组数据变更会改变该摘要，从而驱动计划版本失效。
"""

from typing import Dict, List

from .canonical import digest
from .fixedpoint import SAT_MAX, ceil_div
from .model import BlockGroupSpec, FilterPredicate, HistogramBucket

FILTER_SELECTIVITY = {
    "eq": (1, 8),
    "ne": (7, 8),
    "lt": (1, 3),
    "le": (1, 2),
    "gt": (1, 3),
    "ge": (1, 2),
    "is_null": (1, 16),
    "not_null": (15, 16),
}


def stats_fingerprint(groups: List[BlockGroupSpec]) -> Dict[str, object]:
    """返回统计口径的确定性指纹结构（不含任何遍历顺序噪声）。"""
    payload = []
    for group in sorted(groups, key=lambda item: item.bg_id):
        cols = []
        for column in sorted(group.columns, key=lambda item: item.name):
            cols.append(
                {
                    "name": column.name,
                    "ndv": column.ndv,
                    "null_count": column.null_count,
                    "histogram": [
                        {"low": b.low, "high": b.high, "count": b.count}
                        for b in column.histogram
                    ],
                }
            )
        payload.append({"bg_id": group.bg_id, "rows": group.rows, "columns": cols})
    return {"stats": payload}


def stats_digest(groups: List[BlockGroupSpec]) -> str:
    return digest(stats_fingerprint(groups))


def effective_rows(group: BlockGroupSpec) -> int:
    """非空行数：以任意一列的 null_count 为准（位图全列共享）。"""
    if not group.columns:
        return group.rows
    return max(0, group.rows - group.columns[0].null_count)


def histogram_total(buckets: List[HistogramBucket]) -> int:
    return sum(bucket.count for bucket in buckets)


def filter_selectivity(pred: FilterPredicate):
    return FILTER_SELECTIVITY[pred.op]


def apply_filters(rows: int, filters: List[FilterPredicate]) -> int:
    """以整数比例连乘选择率，全部 floor 截断，夹在 [0, rows]。"""
    current = rows
    for pred in filters:
        op = pred.op if hasattr(pred, "op") else pred["op"]
        numerator, denominator = FILTER_SELECTIVITY[op]
        current = (current * numerator) // denominator
    return max(0, min(rows, current))


def ndv_of(group: BlockGroupSpec, column: str) -> int:
    spec = group.column(column)
    if spec is None or spec.ndv <= 0:
        return max(1, effective_rows(group))
    return max(1, spec.ndv)


def join_cardinality(
    left_rows: int,
    right_rows: int,
    left_ndv: int,
    right_ndv: int,
) -> int:
    """等值连接基数：max(l,r) * (l/ndv_l) * (r/ndv_r)，floor 截断并饱和。"""
    denominator = max(1, left_ndv * right_ndv)
    estimate = (max(left_rows, right_rows) * left_rows * right_rows) // denominator
    return min(SAT_MAX, max(0, estimate))


def blocks_for(rows: int, granularity: int) -> int:
    return ceil_div(max(0, rows), granularity)
