from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_json, digest_json, normalize_float
from .errors import fail


@dataclass(frozen=True)
class Bucket:
    index: int
    lower_bound: int | float
    upper_bound: int | float
    lower_inclusive: bool
    upper_inclusive: bool
    row_count: int
    null_count: int
    unique_count: int
    top_key: int | float | None
    top_count: int

    def as_dict(self) -> dict[str, Any]:
        lower_brace = "[" if self.lower_inclusive else "("
        upper_brace = "]" if self.upper_inclusive else ")"
        return {
            "index": self.index,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "lower_inclusive": self.lower_inclusive,
            "upper_inclusive": self.upper_inclusive,
            "interval": f"{lower_brace}{self.lower_bound}, {self.upper_bound}{upper_brace}",
            "row_count": self.row_count,
            "null_count": self.null_count,
            "unique_count": self.unique_count,
            "top_key": self.top_key,
            "top_count": self.top_count,
        }


@dataclass
class Histogram:
    table: str
    key: str
    key_type: str
    bucket_count: int
    quantile_policy: str
    boundary_rule: str
    total_rows: int
    non_null_rows: int
    null_count: int
    buckets: tuple[Bucket, ...]
    internal_boundaries: tuple[int | float, ...]
    digest: str

    def as_dict(self, include_buckets: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "table": self.table,
            "key": self.key,
            "key_type": self.key_type,
            "bucket_count": self.bucket_count,
            "quantile_policy": self.quantile_policy,
            "boundary_rule": self.boundary_rule,
            "total_rows": self.total_rows,
            "non_null_rows": self.non_null_rows,
            "null_count": self.null_count,
            "digest": self.digest,
        }
        if include_buckets:
            result["buckets"] = [bucket.as_dict() for bucket in self.buckets]
        return result

    def bucket_index(self, value: int | float) -> int:
        if value is None:
            raise ValueError("null keys do not belong to a bucket")
        if self.key_type == "integer":
            return min(len(self.buckets) - 1, bisect.bisect_right(self.internal_boundaries, value))
        return min(len(self.buckets) - 1, bisect.bisect_left(self.internal_boundaries, value))


def _boundary_rule(key_type: str) -> str:
    if key_type == "integer":
        return "整数键: 所有桶 [lower, upper)，末桶 upper=max+1 作为排他哨兵"
    return "浮点键: 首桶 [lower, upper]，后续桶 (lower, upper]"


def build_histogram(
    table: str,
    key: str,
    key_type: str,
    rows: Iterable[dict[str, Any]],
    bucket_count: int,
    quantile_policy: str = "nearest_rank",
) -> Histogram:
    if not isinstance(bucket_count, int) or isinstance(bucket_count, bool) or bucket_count <= 0:
        fail("E_HIST", "分桶数必须是正整数", "params.bucket_count", value=bucket_count)
    if key_type not in {"integer", "float"}:
        fail("E_TYPE", "直方图仅支持数值联接键", "join", key=key, key_type=key_type)
    if quantile_policy not in {"nearest_rank", "lower_rank"}:
        fail("E_HIST", "未知分位口径", "params.quantile_policy", value=quantile_policy)
    row_list = list(rows)
    values = sorted(float(row[key]) if key_type == "float" else int(row[key]) for row in row_list if row.get(key) is not None)
    null_count = len(row_list) - len(values)
    non_null_rows = len(values)
    if bucket_count > len(row_list):
        fail(
            "E_HIST",
            "分桶数超过表行数",
            "params.bucket_count",
            bucket_count=bucket_count,
            table_rows=len(row_list),
        )
    if non_null_rows == 0 or bucket_count > non_null_rows:
        fail(
            "E_HIST",
            "分桶数超过非空联接键行数",
            "params.bucket_count",
            bucket_count=bucket_count,
            non_null_rows=non_null_rows,
        )

    boundaries_raw: list[int | float] = []
    if bucket_count > 1:
        for bucket_index in range(1, bucket_count):
            position = (bucket_index * non_null_rows) // bucket_count
            if quantile_policy == "nearest_rank" and key_type == "integer":
                index = min(non_null_rows - 1, position)
            else:
                index = max(0, min(non_null_rows - 1, position - 1))
            value = values[index]
            boundaries_raw.append(normalize_float(value) if key_type == "float" else int(value))

    if key_type == "integer":
        assignment = lambda value: bisect.bisect_right(boundaries_raw, value)
        lower_bounds = [values[0]] + boundaries_raw
        upper_bounds = boundaries_raw + [values[-1] + 1]
        bucket_lower_inclusive = [True] * bucket_count
        bucket_upper_inclusive = [False] * bucket_count
    else:
        assignment = lambda value: bisect.bisect_left(boundaries_raw, value)
        lower_bounds = [normalize_float(values[0])] + boundaries_raw
        upper_bounds = boundaries_raw + [normalize_float(values[-1])]
        bucket_lower_inclusive = [True] + [False] * (bucket_count - 1)
        bucket_upper_inclusive = [True] * bucket_count

    row_counts = [0] * bucket_count
    key_values: list[list[int | float]] = [[] for _ in range(bucket_count)]
    for value in values:
        index = min(bucket_count - 1, assignment(value))
        row_counts[index] += 1
        key_values[index].append(value)

    buckets: list[Bucket] = []
    for index in range(bucket_count):
        counter = Counter(key_values[index])
        if counter:
            top_key, top_count = min(counter.items(), key=lambda item: (-item[1], item[0]))
        else:
            top_key, top_count = None, 0
        buckets.append(
            Bucket(
                index=index,
                lower_bound=lower_bounds[index],
                upper_bound=upper_bounds[index],
                lower_inclusive=bucket_lower_inclusive[index],
                upper_inclusive=bucket_upper_inclusive[index],
                row_count=row_counts[index],
                null_count=0,
                unique_count=len(counter),
                top_key=top_key,
                top_count=top_count,
            )
        )

    digest_payload = {
        "table": table,
        "key": key,
        "key_type": key_type,
        "bucket_count": bucket_count,
        "quantile_policy": quantile_policy,
        "boundary_rule": _boundary_rule(key_type),
        "total_rows": len(row_list),
        "non_null_rows": non_null_rows,
        "null_count": null_count,
        "buckets": [bucket.as_dict() for bucket in buckets],
    }
    return Histogram(
        table=table,
        key=key,
        key_type=key_type,
        bucket_count=bucket_count,
        quantile_policy=quantile_policy,
        boundary_rule=_boundary_rule(key_type),
        total_rows=len(row_list),
        non_null_rows=non_null_rows,
        null_count=null_count,
        buckets=tuple(buckets),
        internal_boundaries=tuple(boundaries_raw),
        digest=digest_json(digest_payload),
    )


def histogram_bytes(histogram: Histogram) -> bytes:
    return canonical_json(histogram.as_dict()).encode("utf-8")

