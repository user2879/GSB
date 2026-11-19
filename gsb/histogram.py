from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .errors import GSBError
from .model import BOUNDARY_RULES
from .util import FIXED_SCALE, json_number, sha256_json


@dataclass(frozen=True)
class Bucket:
    index: int
    lower: int | float
    upper: int | float
    lower_inclusive: bool
    upper_inclusive: bool
    row_count: int
    null_count: int
    digest: str

    def interval_label(self) -> str:
        left = "[" if self.lower_inclusive else "("
        right = "]" if self.upper_inclusive else ")"
        return f"{left}{self.lower}, {self.upper}{right}"

    def range_for_mass(self, kind: str) -> tuple[float, float, bool, bool] | None:
        lower = float(self.lower)
        upper = float(self.upper)
        low = lower + (0 if self.lower_inclusive else 1)
        high = upper - (0 if self.upper_inclusive else 1)
        if kind == "int":
            if high < low:
                if self.row_count > 0 and upper == lower:
                    return (lower, upper, True, True)
                return None
            return (low, high, True, True)
        if upper < lower:
            return None
        return (lower, upper, self.lower_inclusive, self.upper_inclusive)

    def width_for_mass(self, kind: str) -> float:
        mass_range = self.range_for_mass(kind)
        if mass_range is None:
            return 1.0 if self.row_count > 0 else 0.0
        low, high, _, _ = mass_range
        if kind == "int":
            width = int(round(high - low + 1))
            return float(max(width, 1))
        width = high - low
        return float(width if width > 0 else 1.0)

    def contains(self, value: int | float, kind: str) -> bool:
        number = float(value)
        lower = float(self.lower)
        upper = float(self.upper)
        if number < lower or number > upper:
            return False
        if self.lower_inclusive and self.upper_inclusive:
            return True
        if self.lower_inclusive and number == lower:
            return True
        if self.upper_inclusive and number == upper:
            return True
        return lower < number < upper

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "lower": json_number(self.lower),
            "upper": json_number(self.upper),
            "lower_inclusive": self.lower_inclusive,
            "upper_inclusive": self.upper_inclusive,
            "interval": self.interval_label(),
            "row_count": self.row_count,
            "null_count": self.null_count,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class Histogram:
    table: str
    column: str
    kind: str
    rule: str
    bucket_count: int
    row_count: int
    null_count: int
    edges: tuple[int | float, ...]
    buckets: tuple[Bucket, ...]
    source_digest: str
    quantile_scope: str
    quantile_sample_size: int
    quantile_basis: str
    null_bucket_digest: str
    digest: str

    def to_dict(self, *, include_buckets: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "table": self.table,
            "column": self.column,
            "key_type": self.kind,
            "boundary_rule": self.rule,
            "bucket_count": self.bucket_count,
            "row_count": self.row_count,
            "null_count": self.null_count,
            "edges": [json_number(edge) for edge in self.edges],
            "quantile_scope": self.quantile_scope,
            "quantile_sample_size": self.quantile_sample_size,
            "quantile_basis": self.quantile_basis,
            "source_digest": self.source_digest,
            "null_bucket": {
                "row_count": self.null_count,
                "digest": self.null_bucket_digest,
                "semantics": "空值不参与等值联接与分位边界，单列为空值桶。",
            },
            "digest": self.digest,
        }
        if include_buckets:
            payload["buckets"] = [bucket.to_dict() for bucket in self.buckets]
        return payload


def _clean_values(values: np.ndarray) -> np.ndarray:
    if np.issubdtype(values.dtype, np.floating):
        return values[~np.isnan(values)]
    from .util import INT64_NULL

    return values[values != INT64_NULL]


def _bucket_digest(values: np.ndarray) -> str:
    if len(values) == 0:
        return hashlib.sha256(b"GSB-BUCKET-v1\0empty").hexdigest()
    ordered = np.sort(values, kind="mergesort")
    digest = hashlib.sha256()
    digest.update(b"GSB-BUCKET-v1\0")
    digest.update(ordered.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(len(ordered)).encode("ascii"))
    digest.update(b"\0")
    digest.update(ordered.tobytes(order="C"))
    return digest.hexdigest()


def _edge_values(sorted_values: np.ndarray, bucket_count: int) -> list[int | float]:
    n = len(sorted_values)
    if n == 0:
        return []
    edges = []
    for index in range(bucket_count + 1):
        position = min(n - 1, (index * n) // bucket_count)
        value = sorted_values[position]
        edges.append(int(value) if np.issubdtype(sorted_values.dtype, np.integer) else float(value))
    return edges


def assign_bucket_indices(values: np.ndarray, edges: Iterable[int | float], rule: str) -> np.ndarray:
    edge_array = np.asarray(list(edges))
    if len(edge_array) == 0:
        return np.full(len(values), -1, dtype=np.int64)
    side = "right" if rule == "left_closed_right_open" else "left"
    indices = np.searchsorted(edge_array, values, side=side) - 1
    return np.clip(indices, 0, len(edge_array) - 2).astype(np.int64, copy=False)


def _inclusive_flags(rule: str, index: int, bucket_count: int) -> tuple[bool, bool]:
    if rule == "left_closed_right_open":
        return True, index == bucket_count - 1
    return index == 0, True


def build_histogram(
    values: np.ndarray,
    *,
    table: str,
    column: str,
    kind: str,
    bucket_count: int,
    rule: str,
    source_digest: str,
) -> Histogram:
    if bucket_count <= 0:
        raise GSBError("E_BUCKET", "分桶数必须为正整数", {"buckets": bucket_count})
    if rule not in set(BOUNDARY_RULES.values()):
        raise GSBError("E_SPEC", "未知边界归属规则", {"rule": rule})
    if values.ndim != 1:
        raise GSBError("E_SPEC", "联接键必须为一维列式数组", {"table": table, "column": column})
    clean = _clean_values(values)
    null_count = int(len(values) - len(clean))
    sorted_values = np.sort(clean, kind="mergesort")
    edges = _edge_values(sorted_values, bucket_count)
    buckets: list[Bucket] = []
    if edges:
        indices = assign_bucket_indices(clean, edges, rule)
        for index in range(bucket_count):
            member_mask = indices == index
            member_values = clean[member_mask]
            lower = edges[index]
            upper = edges[index + 1]
            lower_inclusive, upper_inclusive = _inclusive_flags(rule, index, bucket_count)
            buckets.append(
                Bucket(
                    index=index,
                    lower=lower,
                    upper=upper,
                    lower_inclusive=lower_inclusive,
                    upper_inclusive=upper_inclusive,
                    row_count=int(len(member_values)),
                    null_count=0,
                    digest=_bucket_digest(member_values),
                )
            )
    else:
        for index in range(bucket_count):
            lower_inclusive, upper_inclusive = _inclusive_flags(rule, index, bucket_count)
            buckets.append(
                Bucket(
                    index=index,
                    lower=0,
                    upper=0,
                    lower_inclusive=lower_inclusive,
                    upper_inclusive=upper_inclusive,
                    row_count=0,
                    null_count=0,
                    digest=_bucket_digest(np.empty(0, dtype=values.dtype)),
                )
            )
    null_bucket_digest = hashlib.sha256(
        b"GSB-NULL-BUCKET-v1\0" + str(null_count).encode("ascii")
    ).hexdigest()
    context = {
        "table": table,
        "column": column,
        "kind": kind,
        "rule": rule,
        "bucket_count": bucket_count,
        "row_count": int(len(clean)),
        "null_count": null_count,
        "edges": [json_number(edge) for edge in edges],
        "source_digest": source_digest,
        "null_bucket": {"row_count": null_count, "digest": null_bucket_digest},
        "buckets": [bucket.to_dict() for bucket in buckets],
        "quantile_scope": "full_filtered",
    }
    digest = sha256_json(context)
    return Histogram(
        table=table,
        column=column,
        kind=kind,
        rule=rule,
        bucket_count=bucket_count,
        row_count=int(len(clean)),
        null_count=null_count,
        edges=tuple(edges),
        buckets=tuple(buckets),
        source_digest=source_digest,
        quantile_scope="full_filtered",
        quantile_sample_size=int(len(clean)),
        quantile_basis=(
            "分位边界由过滤后全量非空联接键排序统计确定；"
            "不使用随机分位抽样；等深位置为 floor(i*n/buckets)。"
        ),
        null_bucket_digest=null_bucket_digest,
        digest=digest,
    )


def bucket_for_values(histogram: Histogram, values: np.ndarray) -> np.ndarray:
    if not histogram.edges:
        return np.full(len(values), -1, dtype=np.int64)
    return assign_bucket_indices(values, histogram.edges, histogram.rule)


def _intersection_mass(
    left_bucket: Bucket,
    right_bucket: Bucket,
    *,
    left_kind: str,
    right_kind: str,
) -> float | None:
    left_range = left_bucket.range_for_mass(left_kind)
    right_range = right_bucket.range_for_mass(right_kind)
    if left_range is None or right_range is None:
        if left_bucket.contains(right_bucket.lower, left_kind) and right_bucket.contains(
            right_bucket.lower, right_kind
        ):
            return 1.0
        return None
    low = max(left_range[0], right_range[0])
    high = min(left_range[1], right_range[1])
    if low > high:
        return None
    if left_kind == "int" and right_kind == "int":
        width = int(round(high - low + 1))
        return float(max(width, 0))
    width = high - low
    if width > 0:
        return float(width)
    if left_bucket.contains(high, left_kind) and right_bucket.contains(high, right_kind):
        return 1.0
    return None


def overlap_pairs(left: Histogram, right: Histogram) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_bucket in left.buckets:
        for right_bucket in right.buckets:
            overlap_width = _intersection_mass(
                left_bucket,
                right_bucket,
                left_kind=left.kind,
                right_kind=right.kind,
            )
            if overlap_width is None:
                continue
            left_width = left_bucket.width_for_mass(left.kind)
            right_width = right_bucket.width_for_mass(right.kind)
            if left_width <= 0 or right_width <= 0:
                estimate = 0.0
            else:
                estimate = (
                    (left_bucket.row_count / left_width)
                    * (right_bucket.row_count / right_width)
                    * overlap_width
                )
            pair_id = f"L{left_bucket.index}:R{right_bucket.index}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "left_bucket": left_bucket.index,
                    "right_bucket": right_bucket.index,
                    "left_lower": left_bucket.lower,
                    "left_upper": left_bucket.upper,
                    "right_lower": right_bucket.lower,
                    "right_upper": right_bucket.upper,
                    "left_interval": left_bucket.interval_label(),
                    "right_interval": right_bucket.interval_label(),
                    "left_rows": left_bucket.row_count,
                    "right_rows": right_bucket.row_count,
                    "left_width": left_width,
                    "right_width": right_width,
                    "overlap_width": float(overlap_width),
                    "histogram_estimate": float(estimate),
                    "histogram_fixed": int(round(estimate * FIXED_SCALE)),
                }
            )
    pairs.sort(key=lambda item: (item["left_lower"], item["left_bucket"], item["right_lower"], item["right_bucket"]))
    return pairs


def overlap_row_ratios(left: Histogram, right: Histogram, pairs: list[dict[str, Any]]) -> dict[str, float]:
    left_indices = {int(pair["left_bucket"]) for pair in pairs}
    right_indices = {int(pair["right_bucket"]) for pair in pairs}
    left_rows = sum(left.buckets[index].row_count for index in left_indices)
    right_rows = sum(right.buckets[index].row_count for index in right_indices)
    return {
        "left": (left_rows / left.row_count) if left.row_count else 0.0,
        "right": (right_rows / right.row_count) if right.row_count else 0.0,
    }
