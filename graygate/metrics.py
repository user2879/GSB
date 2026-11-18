from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Iterable, Sequence

from .spec import GatewaySpec
from .util import canonical_json, first_difference


STAT_FIELDS = (
    "stage_weight_bp",
    "stable_hits",
    "canary_hits",
    "canary_ratio_bp",
    "ratio_error_upper_bound_bp",
    "stable_total_us",
    "canary_total_us",
    "stable_max_us",
    "canary_max_us",
    "stable_outputs",
    "canary_outputs",
)


def _empty_row(rule_id: str, stage_id: str, stage_weight_bp: int) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "stage_id": stage_id,
        "stage_weight_bp": stage_weight_bp,
        "stable_hits": 0,
        "canary_hits": 0,
        "canary_ratio_bp": 0,
        "ratio_error_upper_bound_bp": 0,
        "stable_total_us": 0,
        "canary_total_us": 0,
        "stable_max_us": 0,
        "canary_max_us": 0,
        "stable_outputs": [],
        "canary_outputs": [],
    }


def _add_hit(row: dict[str, Any], hit: Any) -> None:
    side = hit.side
    if side == "stable":
        row["stable_hits"] += 1
        row["stable_total_us"] += int(hit.elapsed_us)
        row["stable_max_us"] = max(row["stable_max_us"], int(hit.elapsed_us))
        row["stable_outputs"].append(hit.output_digest)
    elif side == "canary":
        row["canary_hits"] += 1
        row["canary_total_us"] += int(hit.elapsed_us)
        row["canary_max_us"] = max(row["canary_max_us"], int(hit.elapsed_us))
        row["canary_outputs"].append(hit.output_digest)
    else:
        raise ValueError(f"unknown side {side!r}")


def _finalize_row(row: dict[str, Any]) -> dict[str, Any]:
    total = row["stable_hits"] + row["canary_hits"]
    if total:
        row["canary_ratio_bp"] = (row["canary_hits"] * 10_000) // total
        row["ratio_error_upper_bound_bp"] = 1
    else:
        row["canary_ratio_bp"] = 0
        row["ratio_error_upper_bound_bp"] = 0
    row["stable_outputs"] = sorted(set(row["stable_outputs"]))
    row["canary_outputs"] = sorted(set(row["canary_outputs"]))
    return row


def _ordered_stage_pairs(spec: GatewaySpec) -> list[tuple[str, str, int]]:
    result: list[tuple[str, str, int]] = []
    for route in spec.routes:
        for stage in route.stages:
            result.append((route.rule_id, stage.stage_id, stage.gray_weight_bp))
    return result


def aggregate_full(hits: Sequence[Any], spec: GatewaySpec) -> list[dict[str, Any]]:
    """Full grouping path: sort, group, then aggregate each group."""
    ordered = sorted(hits, key=lambda hit: (hit.rule_id, hit.stage_id, hit.ledger_version))
    grouped: dict[tuple[str, str], list[Any]] = {}
    for key, values in groupby(ordered, key=lambda hit: (hit.rule_id, hit.stage_id)):
        grouped[key] = list(values)
    rows: list[dict[str, Any]] = []
    for rule_id, stage_id, weight in _ordered_stage_pairs(spec):
        row = _empty_row(rule_id, stage_id, weight)
        for hit in grouped.get((rule_id, stage_id), []):
            _add_hit(row, hit)
        rows.append(_finalize_row(row))
    return rows


def aggregate_streaming(hits: Iterable[Any], spec: GatewaySpec) -> list[dict[str, Any]]:
    """Single-pass path: maintain one accumulator per rule/stage."""
    rows = {
        (rule_id, stage_id): _empty_row(rule_id, stage_id, weight)
        for rule_id, stage_id, weight in _ordered_stage_pairs(spec)
    }
    for hit in hits:
        key = (hit.rule_id, hit.stage_id)
        row = rows.get(key)
        if row is None:
            raise ValueError(
                f"hit references unknown rule/stage pair {hit.rule_id!r}/{hit.stage_id!r}"
            )
        _add_hit(row, hit)
    result: list[dict[str, Any]] = []
    for rule_id, stage_id, _weight in _ordered_stage_pairs(spec):
        result.append(_finalize_row(rows[(rule_id, stage_id)]))
    return result


def choose_aggregation_path(
    spec: GatewaySpec, hit_count: int, threshold: int | None = None
) -> str:
    limit = spec.stream_threshold if threshold is None else threshold
    if spec.stage_count >= limit or hit_count >= limit:
        return "streaming"
    return "full"


def aggregate_hits(
    hits: Sequence[Any], spec: GatewaySpec, *, path: str | None = None
) -> tuple[list[dict[str, Any]], str]:
    selected = path or choose_aggregation_path(spec, len(hits))
    if selected == "full":
        return aggregate_full(hits, spec), selected
    if selected == "streaming":
        return aggregate_streaming(hits, spec), selected
    raise ValueError(f"unknown aggregation path {selected!r}")


def stats_map(stats: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["rule_id"]), str(row["stage_id"])): row for row in stats}


def compare_stats(
    expected: Sequence[dict[str, Any]],
    actual: Sequence[dict[str, Any]],
) -> tuple[bool, str, str, str, Any, Any]:
    """Compare statistics; return first rule, stage and field mismatch."""
    expected_map = stats_map(expected)
    actual_map = stats_map(actual)
    ordered_keys = list(expected_map.keys())
    for key in actual_map:
        if key not in expected_map:
            ordered_keys.append(key)
    for key in ordered_keys:
        rule_id, stage_id = key
        left = expected_map.get(key)
        right = actual_map.get(key)
        if left is None:
            return False, rule_id, stage_id, "stage_row", None, right
        if right is None:
            return False, rule_id, stage_id, "stage_row", left, None
        for field in STAT_FIELDS:
            if left.get(field) != right.get(field):
                return False, rule_id, stage_id, field, left.get(field), right.get(field)
        if canonical_json(left) != canonical_json(right):
            return False, rule_id, stage_id, "row_bytes", left, right
    if canonical_json(expected) != canonical_json(actual):
        first = ordered_keys[0] if ordered_keys else ("-", "-")
        return False, first[0], first[1], "stats_bytes", expected, actual
    return True, "-", "-", "-", None, None


@dataclass(frozen=True)
class LatencySummary:
    count: int
    scale: int
    mean_scaled_ms: int
    p95_ms: int
    max_ms: int
    quantization_error_upper_bound_scaled_ms: int

    def canonical(self) -> dict[str, int]:
        return {
            "count": self.count,
            "scale": self.scale,
            "mean_scaled_ms": self.mean_scaled_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
            "quantization_error_upper_bound_scaled_ms": (
                self.quantization_error_upper_bound_scaled_ms
            ),
        }


def _latency_summary(
    count: int, total: int, p95: int, maximum: int, *, scale: int
) -> LatencySummary:
    if count == 0:
        return LatencySummary(0, scale, 0, 0, 0, 0)
    return LatencySummary(
        count=count,
        scale=scale,
        mean_scaled_ms=(total * scale) // count,
        p95_ms=p95,
        max_ms=maximum,
        quantization_error_upper_bound_scaled_ms=1,
    )


def summarize_latency_full(values: Sequence[int], *, scale: int = 1000) -> LatencySummary:
    if not values:
        return _latency_summary(0, 0, 0, 0, scale=scale)
    ordered = sorted(values)
    p95_index = ((95 * len(ordered) + 99) // 100) - 1
    return _latency_summary(
        len(ordered), sum(ordered), ordered[p95_index], ordered[-1], scale=scale
    )


def summarize_latency_streaming(
    values: Iterable[int], *, scale: int = 1000
) -> LatencySummary:
    counts: Counter[int] = Counter()
    count = 0
    total = 0
    maximum = 0
    for raw in values:
        value = int(raw)
        counts[value] += 1
        count += 1
        total += value
        maximum = max(maximum, value)
    if count == 0:
        return _latency_summary(0, 0, 0, 0, scale=scale)
    rank = (95 * count + 99) // 100
    seen = 0
    p95 = 0
    for value in sorted(counts):
        seen += counts[value]
        if seen >= rank:
            p95 = value
            break
    return _latency_summary(count, total, p95, maximum, scale=scale)
