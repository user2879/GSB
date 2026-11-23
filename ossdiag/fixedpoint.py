from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SATURATED_FIXED = (1 << 63) - 1


@dataclass(frozen=True)
class FixedPoint:
    """Non-negative integer fixed point with explicit flooring and saturation."""

    exponent: int

    @property
    def scale(self) -> int:
        return 10**self.exponent

    @property
    def error_bound_units(self) -> int:
        return 2

    def saturate(self, value: int) -> int:
        if value <= 0:
            return 0
        return min(value, SATURATED_FIXED)

    def ratio(self, numerator: int, denominator: int) -> int:
        if denominator <= 0:
            return 0
        return self.saturate((max(0, numerator) * self.scale) // denominator)

    def duration_ns_as_ms(self, nanoseconds: int) -> int:
        if nanoseconds <= 0:
            return 0
        return self.saturate((nanoseconds * self.scale) // 1_000_000)

    def decimal(self, value: int, unit: str = "") -> str:
        whole, fraction = divmod(value, self.scale)
        if self.exponent == 0:
            rendered = str(whole)
        else:
            rendered = f"{whole}.{fraction:0{self.exponent}d}"
        return f"{rendered}{unit}"

    def error_decimal(self, units: int, unit: str = "") -> str:
        return self.decimal(units, unit)

    def policy(self) -> dict[str, object]:
        return {
            "scale_exponent": self.exponent,
            "scale": self.scale,
            "rounding": "floor toward zero for non-negative values",
            "saturation": f"clamp to [0, {SATURATED_FIXED}]",
            "single_value_quantization_error": self.error_decimal(1),
            "proportion_comparison_error_bound": self.error_decimal(2),
            "object_aggregate_error_upper": self.error_decimal(2),
            "units": {
                "ratio": "dimensionless fixed-point fraction",
                "duration": "milliseconds in fixed-point units",
            },
        }


@dataclass(frozen=True)
class ObjectNumericStats:
    object_key: str
    total_bytes: int
    completed_bytes: int
    completion_fixed: int
    completed_shards: int
    shard_count: int
    unknown_shards: int
    average_digest_ms_fixed: int
    digest_duration_ns_total: int
    lease_grants: int
    renew_count: int
    renewal_rate_fixed: int
    ratio_quantization_error: int
    duration_quantization_error: int
    comparison_error_bound: int
    object_aggregate_error_upper: int

    def as_dict(self, fixed: FixedPoint) -> dict[str, object]:
        return {
            "object_key": self.object_key,
            "total_bytes": self.total_bytes,
            "completed_bytes": self.completed_bytes,
            "completion_fixed": self.completion_fixed,
            "completion_display": fixed.decimal(self.completion_fixed),
            "completed_shards": self.completed_shards,
            "shard_count": self.shard_count,
            "unknown_shards": self.unknown_shards,
            "average_digest_ms_fixed": self.average_digest_ms_fixed,
            "average_digest_ms_display": fixed.decimal(self.average_digest_ms_fixed, " ms"),
            "digest_duration_ns_total": self.digest_duration_ns_total,
            "lease_grants": self.lease_grants,
            "renew_count": self.renew_count,
            "renewal_rate_fixed": self.renewal_rate_fixed,
            "renewal_rate_display": fixed.decimal(self.renewal_rate_fixed),
            "ratio_quantization_error": self.ratio_quantization_error,
            "duration_quantization_error": self.duration_quantization_error,
            "comparison_error_bound": self.comparison_error_bound,
            "object_aggregate_error_upper": self.object_aggregate_error_upper,
            "error_bounds_display": {
                "single_ratio": fixed.error_decimal(self.ratio_quantization_error),
                "single_duration": fixed.error_decimal(self.duration_quantization_error, " ms"),
                "comparison": fixed.error_decimal(self.comparison_error_bound),
                "object_aggregate": fixed.error_decimal(self.object_aggregate_error_upper),
            },
        }


def compute_object_stats(
    *,
    object_key: str,
    total_bytes: int,
    shard_rows: Iterable[dict],
    fixed: FixedPoint,
    lease_grants: int = 0,
    renew_count: int = 0,
) -> ObjectNumericStats:
    rows = list(shard_rows)
    completed_bytes = 0
    completed_shards = 0
    unknown_shards = 0
    duration_total = 0
    duration_count = 0
    for row in rows:
        start = int(row["byte_start"])
        end = int(row["byte_end"])
        cursor = int(row.get("progress_cursor") or start)
        completed_bytes += max(0, min(end, cursor) - start)
        if row.get("status") == "complete":
            completed_shards += 1
        if row.get("status") == "unknown" or int(row.get("unknown") or 0):
            unknown_shards += 1
        if int(row.get("digest_duration_ns") or 0) > 0:
            duration_total += int(row["digest_duration_ns"])
            duration_count += 1
    average_ns = duration_total // duration_count if duration_count else 0
    return ObjectNumericStats(
        object_key=object_key,
        total_bytes=total_bytes,
        completed_bytes=completed_bytes,
        completion_fixed=fixed.ratio(completed_bytes, total_bytes),
        completed_shards=completed_shards,
        shard_count=len(rows),
        unknown_shards=unknown_shards,
        average_digest_ms_fixed=fixed.duration_ns_as_ms(average_ns),
        digest_duration_ns_total=duration_total,
        lease_grants=lease_grants,
        renew_count=renew_count,
        renewal_rate_fixed=fixed.ratio(renew_count, lease_grants),
        ratio_quantization_error=1,
        duration_quantization_error=1,
        comparison_error_bound=2,
        object_aggregate_error_upper=2,
    )


def rank_objects_by_completion(
    stats: Iterable[ObjectNumericStats], fixed: FixedPoint
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Stable evidence-aware ordering.

    Values sorted high-to-low are clustered while adjacent gaps are below the
    two-value comparison bound (2 fixed units).  A cluster is then ordered by
    object key, and every multi-object cluster is explicitly marked as affected
    by quantization error.
    """

    values = sorted(stats, key=lambda item: (-item.completion_fixed, item.object_key))
    clusters: list[list[ObjectNumericStats]] = []
    for item in values:
        if not clusters:
            clusters.append([item])
            continue
        previous = clusters[-1][-1]
        if abs(previous.completion_fixed - item.completion_fixed) < fixed.error_bound_units:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    ordered: list[dict[str, object]] = []
    affected_pairs: list[dict[str, object]] = []
    rank = 1
    for cluster in clusters:
        cluster.sort(key=lambda item: item.object_key)
        cluster_affected = len(cluster) > 1
        for left_index, left in enumerate(cluster):
            for right in cluster[left_index + 1 :]:
                delta = abs(left.completion_fixed - right.completion_fixed)
                if delta < fixed.error_bound_units:
                    affected_pairs.append(
                        {
                            "left": left.object_key,
                            "right": right.object_key,
                            "completion_delta_fixed": delta,
                            "bound_fixed": fixed.error_bound_units,
                        }
                    )
        for item in cluster:
            ordered.append(
                {
                    "rank": rank,
                    "object_key": item.object_key,
                    "completion_fixed": item.completion_fixed,
                    "completion_display": fixed.decimal(item.completion_fixed),
                    "ordering_affected_by_error_bound": cluster_affected,
                }
            )
            rank += 1
    return ordered, affected_pairs
