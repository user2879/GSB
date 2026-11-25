"""本地 JSON 声明加载与强校验。

校验严格按声明顺序进行，遇到首个非法对象/字段即抛 :class:`SpecError`，
错误码统一为 E_SPEC，并携带 object/field 定位信息。
"""

import json
import os
from typing import Any, Dict, List

from .errors import SpecError
from .model import (
    Aggregate,
    BlockGroupSpec,
    ColumnSpec,
    CostWeights,
    FilterPredicate,
    HistogramBucket,
    JoinPredicate,
    QuerySpec,
    SessionSpec,
)

VALID_DTYPES = ("integer", "float")
VALID_ENCODINGS = ("plain", "dictionary", "rle")
VALID_FILTER_OPS = ("eq", "ne", "lt", "le", "gt", "ge", "is_null", "not_null")
VALID_AGG_FUNCS = ("count", "sum", "min", "max", "avg")
KNOWN_WEIGHT_KEYS = ("scan", "compare", "build", "probe", "memory")


def _read_json(path: str) -> Any:
    if not os.path.exists(path):
        raise SpecError(f"声明文件不存在: {path}", field="file")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"JSON 解析失败: {exc.msg}", field="json", line=getattr(exc, "lineno", None)
        ) from exc


def _require(obj: Dict[str, Any], key: str, object_id: str, dtype=None):
    if key not in obj:
        raise SpecError("缺少必需字段", object_id=object_id, field=key)
    if dtype is not None and not isinstance(obj[key], dtype):
        raise SpecError(
            f"字段类型应为 {dtype.__name__}", object_id=object_id, field=key
        )
    return obj[key]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
        value, float
    )


def _validate_histogram(
    buckets_raw: Any, rows: int, object_id: str, col_name: str
) -> List[HistogramBucket]:
    field_name = f"columns.{col_name}.histogram"
    if not isinstance(buckets_raw, list) or not buckets_raw:
        raise SpecError("直方图必须为非空区间数组", object_id=object_id, field=field_name)
    buckets: List[HistogramBucket] = []
    prev_high = None
    total = 0
    for index, raw in enumerate(buckets_raw):
        bfield = f"{field_name}[{index}]"
        if not isinstance(raw, dict):
            raise SpecError("直方图区间必须为对象", object_id=object_id, field=bfield)
        low = _require(raw, "low", object_id)
        high = _require(raw, "high", object_id)
        count = _require(raw, "count", object_id)
        if not _is_number(low) or not _is_number(high):
            raise SpecError("区间端点必须为数值", object_id=object_id, field=bfield)
        if not _is_int(count) or count < 0:
            raise SpecError("区间计数必须为非负整数", object_id=object_id, field=bfield)
        if float(high) < float(low):
            raise SpecError("区间 high 不得小于 low", object_id=object_id, field=bfield)
        if prev_high is not None:
            if float(low) < float(prev_high):
                raise SpecError(
                    "直方图区间重叠：low 小于上一区间 high",
                    object_id=object_id,
                    field=bfield,
                )
            if float(low) != float(prev_high):
                raise SpecError(
                    "直方图区间未首尾相接：相邻端点必须相等",
                    object_id=object_id,
                    field=bfield,
                )
        buckets.append(HistogramBucket(float(low), float(high), count))
        total += count
        prev_high = high
    if total != rows:
        raise SpecError(
            f"直方图计数之和 {total} 必须等于行数 {rows}",
            object_id=object_id,
            field=field_name,
        )
    return buckets


def _validate_block_group(raw: Any, seen_ids: set) -> BlockGroupSpec:
    if not isinstance(raw, dict):
        raise SpecError("块组声明必须为对象", field="block_groups[]")
    bg_id = _require(raw, "bg_id", "block_groups[]")
    if not isinstance(bg_id, str) or not bg_id:
        raise SpecError("块组标识必须为非空字符串", field="block_groups[].bg_id")
    if bg_id in seen_ids:
        raise SpecError("块组标识重复", object_id=bg_id, field="bg_id")
    seen_ids.add(bg_id)

    rows = _require(raw, "rows", bg_id)
    if not _is_int(rows) or rows < 0:
        raise SpecError("行数必须为非负整数", object_id=bg_id, field="rows")
    seed = _require(raw, "seed", bg_id)
    if not _is_int(seed) or seed < 0:
        raise SpecError("种子必须为非负整数", object_id=bg_id, field="seed")

    null_bitmap = _require(raw, "null_bitmap", bg_id)
    if not isinstance(null_bitmap, list) or any(
        bit not in (0, 1) for bit in null_bitmap
    ):
        raise SpecError(
            "空值位图必须为 0/1 数组", object_id=bg_id, field="null_bitmap"
        )
    if len(null_bitmap) != rows:
        raise SpecError(
            f"空值位图长度 {len(null_bitmap)} 必须等于行数 {rows}",
            object_id=bg_id,
            field="null_bitmap",
        )

    columns_raw = _require(raw, "columns", bg_id, list)
    if not columns_raw:
        raise SpecError("块组至少声明一列", object_id=bg_id, field="columns")
    seen_cols = set()
    columns: List[ColumnSpec] = []
    for col_raw in columns_raw:
        if not isinstance(col_raw, dict):
            raise SpecError("列声明必须为对象", object_id=bg_id, field="columns[]")
        name = _require(col_raw, "name", bg_id)
        if not isinstance(name, str) or not name:
            raise SpecError("列名必须为非空字符串", object_id=bg_id, field="columns[].name")
        if name in seen_cols:
            raise SpecError("列名重复", object_id=bg_id, field=f"columns.{name}.name")
        seen_cols.add(name)
        dtype = _require(col_raw, "dtype", bg_id)
        if dtype not in VALID_DTYPES:
            raise SpecError(
                f"列类型必须属于 {VALID_DTYPES}",
                object_id=bg_id,
                field=f"columns.{name}.dtype",
            )
        encoding = _require(col_raw, "encoding", bg_id)
        if encoding not in VALID_ENCODINGS:
            raise SpecError(
                f"编码方式必须属于 {VALID_ENCODINGS}",
                object_id=bg_id,
                field=f"columns.{name}.encoding",
            )
        histogram_raw = col_raw.get("histogram")
        if histogram_raw is None:
            histogram: List[HistogramBucket] = []
        else:
            histogram = _validate_histogram(histogram_raw, rows, bg_id, name)
        null_count = col_raw.get("null_count", sum(null_bitmap))
        if not _is_int(null_count) or null_count < 0 or null_count > rows:
            raise SpecError(
                "null_count 必须为 [0, rows] 内整数",
                object_id=bg_id,
                field=f"columns.{name}.null_count",
            )
        ndv = col_raw.get("ndv", 0)
        if not _is_int(ndv) or ndv < 0:
            raise SpecError(
                "ndv 必须为非负整数", object_id=bg_id, field=f"columns.{name}.ndv"
            )
        columns.append(ColumnSpec(name, dtype, encoding, histogram, null_count, ndv))

    value_ranges = raw.get("value_ranges", {})
    if not isinstance(value_ranges, dict):
        raise SpecError("value_ranges 必须为对象", object_id=bg_id, field="value_ranges")
    for col_name, rng in value_ranges.items():
        if col_name not in seen_cols:
            raise SpecError(
                "value_ranges 引用了未声明列",
                object_id=bg_id,
                field=f"value_ranges.{col_name}",
            )
        if not isinstance(rng, dict) or "low" not in rng or "high" not in rng:
            raise SpecError(
                "value_ranges 项必须含 low/high",
                object_id=bg_id,
                field=f"value_ranges.{col_name}",
            )
    return BlockGroupSpec(bg_id, rows, seed, list(null_bitmap), columns, value_ranges)


def load_block_groups(path: str) -> List[BlockGroupSpec]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or "block_groups" not in payload:
        raise SpecError("块组文件根对象必须含 block_groups", field="block_groups")
    raw_groups = payload["block_groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        raise SpecError("block_groups 必须为非空数组", field="block_groups")
    seen: set = set()
    groups = [_validate_block_group(raw, seen) for raw in raw_groups]
    return groups


def _column_exists(groups: Dict[str, BlockGroupSpec], bg_id: str, col: str, where: str):
    if bg_id not in groups:
        raise SpecError("谓词引用了不存在的块组", object_id=bg_id, field=where)
    if groups[bg_id].column(col) is None:
        raise SpecError(
            "谓词引用了不存在的列", object_id=bg_id, field=f"{where}.{col}"
        )


def _validate_query(raw: Any, groups: Dict[str, BlockGroupSpec]) -> QuerySpec:
    if not isinstance(raw, dict):
        raise SpecError("查询声明必须为对象", field="query")
    query_id = _require(raw, "query_id", "query")
    if not isinstance(query_id, str) or not query_id:
        raise SpecError("query_id 必须为非空字符串", field="query.query_id")

    joins_raw = _require(raw, "joins", query_id, list)
    if not joins_raw:
        raise SpecError("多表连接至少需要一个连接谓词", object_id=query_id, field="joins")
    joins: List[JoinPredicate] = []
    for index, jr in enumerate(joins_raw):
        if not isinstance(jr, dict):
            raise SpecError("连接谓词必须为对象", object_id=query_id, field=f"joins[{index}]")
        left_bg = _require(jr, "left_bg", query_id)
        left_col = _require(jr, "left_col", query_id)
        right_bg = _require(jr, "right_bg", query_id)
        right_col = _require(jr, "right_col", query_id)
        _column_exists(groups, left_bg, left_col, f"joins[{index}].left")
        _column_exists(groups, right_bg, right_col, f"joins[{index}].right")
        if left_bg == right_bg:
            raise SpecError(
                "自连接不支持：连接谓词两端块组相同",
                object_id=query_id,
                field=f"joins[{index}]",
            )
        joins.append(JoinPredicate(left_bg, left_col, right_bg, right_col))

    filters: List[FilterPredicate] = []
    for index, fr in enumerate(raw.get("filters", [])):
        if not isinstance(fr, dict):
            raise SpecError("过滤谓词必须为对象", object_id=query_id, field=f"filters[{index}]")
        bg_id = _require(fr, "bg", query_id)
        column = _require(fr, "column", query_id)
        op = _require(fr, "op", query_id)
        if op not in VALID_FILTER_OPS:
            raise SpecError(
                f"过滤算子必须属于 {VALID_FILTER_OPS}",
                object_id=query_id,
                field=f"filters[{index}].op",
            )
        _column_exists(groups, bg_id, column, f"filters[{index}]")
        if op not in ("is_null", "not_null") and "value" not in fr:
            raise SpecError(
                "比较型过滤谓词必须提供 value",
                object_id=query_id,
                field=f"filters[{index}].value",
            )
        filters.append(FilterPredicate(bg_id, column, op, fr.get("value")))

    projections = raw.get("projections", [])
    group_by = raw.get("group_by", [])
    for label, refs in (("projections", projections), ("group_by", group_by)):
        if not isinstance(refs, list):
            raise SpecError(f"{label} 必须为数组", object_id=query_id, field=label)
        for index, ref in enumerate(refs):
            if not isinstance(ref, dict) or "bg" not in ref or "column" not in ref:
                raise SpecError(
                    f"{label} 项必须含 bg/column",
                    object_id=query_id,
                    field=f"{label}[{index}]",
                )
            _column_exists(groups, ref["bg"], ref["column"], f"{label}[{index}]")

    aggregates: List[Aggregate] = []
    for index, ar in enumerate(raw.get("aggregates", [])):
        if not isinstance(ar, dict):
            raise SpecError("聚合必须为对象", object_id=query_id, field=f"aggregates[{index}]")
        func = _require(ar, "func", query_id)
        if func not in VALID_AGG_FUNCS:
            raise SpecError(
                f"聚合函数必须属于 {VALID_AGG_FUNCS}",
                object_id=query_id,
                field=f"aggregates[{index}].func",
            )
        bg_id = _require(ar, "bg", query_id)
        column = ar.get("column")
        if func != "count" and column is None:
            raise SpecError(
                "非 count 聚合必须指定 column",
                object_id=query_id,
                field=f"aggregates[{index}].column",
            )
        if column is not None:
            _column_exists(groups, bg_id, column, f"aggregates[{index}]")
        elif bg_id not in groups:
            raise SpecError("聚合引用了不存在的块组", object_id=bg_id, field=f"aggregates[{index}]")
        alias = ar.get("alias", f"{func}_{bg_id}_{column or '*'}")
        aggregates.append(Aggregate(func, bg_id, column, alias))

    referenced = QuerySpec(query_id, joins, filters, projections, group_by, aggregates)
    unknown = referenced.bg_ids() - set(groups.keys())
    if unknown:
        raise SpecError(
            f"查询引用了未声明块组: {sorted(unknown)[0]}",
            object_id=query_id,
            field="bg",
        )
    return referenced


def load_query(path: str, groups: List[BlockGroupSpec]) -> QuerySpec:
    return _validate_query(_read_json(path), {g.bg_id: g for g in groups})


def load_weights(path: str) -> CostWeights:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise SpecError("权重声明必须为对象", field="weights")
    object_id = "weights"
    for key in KNOWN_WEIGHT_KEYS:
        if key not in raw:
            raise SpecError("缺少代价权重", object_id=object_id, field=key)
        if not _is_int(raw[key]) or raw[key] < 0:
            raise SpecError(
                "代价权重必须为非负整数", object_id=object_id, field=key
            )
    lo = raw.get("weight_sum_min", 1)
    hi = raw.get("weight_sum_max", 1_000_000_000)
    if not _is_int(lo) or not _is_int(hi) or lo < 0 or hi < lo:
        raise SpecError(
            "权重和合法区间非法", object_id=object_id, field="weight_sum_min/max"
        )
    total = sum(raw[key] for key in KNOWN_WEIGHT_KEYS)
    if not lo <= total <= hi:
        raise SpecError(
            f"代价权重之和 {total} 未落在合法区间 [{lo}, {hi}]",
            object_id=object_id,
            field="weight_sum",
        )
    return CostWeights(
        raw["scan"], raw["compare"], raw["build"], raw["probe"], raw["memory"], lo, hi
    )


def load_session(path: str) -> SessionSpec:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise SpecError("会话声明必须为对象", field="session")
    scale = raw.get("scale", 1000)
    if not _is_int(scale) or scale <= 0:
        raise SpecError("scale 必须为正整数", object_id="session", field="scale")
    granularity = raw.get("block_granularity", 1024)
    if not _is_int(granularity) or granularity <= 0:
        raise SpecError(
            "block_granularity 必须为正整数", object_id="session", field="block_granularity"
        )
    stats_version = raw.get("stats_version", "stats-v1")
    if not isinstance(stats_version, str) or not stats_version:
        raise SpecError(
            "stats_version 必须为非空字符串", object_id="session", field="stats_version"
        )
    return SessionSpec(scale, granularity, stats_version)
