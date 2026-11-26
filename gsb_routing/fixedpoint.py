"""Integer-only fixed-point waterline accounting.

Rules are deliberately explicit:

* every multiplication/division is performed with Python integers;
* fractional truncation is toward zero (not floor);
* results saturate to the signed 64-bit range;
* lease counts use ``value * scale_factor`` and are therefore exact until the
  saturation boundary is reached.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .spec import RoutingSpec


FIXED_MIN = -(1 << 63)
FIXED_MAX = (1 << 63) - 1


def trunc_div(num: int, den: int) -> int:
    if den <= 0:
        raise ValueError("denominator must be positive")
    negative = num < 0
    quotient = abs(num) // den
    return -quotient if negative else quotient


def ceil_div(num: int, den: int) -> int:
    if den <= 0:
        raise ValueError("denominator must be positive")
    return -((-num) // den)


def saturate(value: int) -> int:
    if value < FIXED_MIN:
        return FIXED_MIN
    if value > FIXED_MAX:
        return FIXED_MAX
    return value


def quantize_count(value: int, scale_factor: int) -> int:
    return saturate(value * scale_factor)


def quantize_ratio(numerator: int, denominator: int, scale_factor: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return saturate(trunc_div(numerator * scale_factor, denominator))


def quantization_header(scale_factor: int) -> dict[str, Any]:
    return {
        "scale_factor": scale_factor,
        "rounding": "toward_zero",
        "saturation": {"min": FIXED_MIN, "max": FIXED_MAX},
        "arithmetic": "integer_only",
    }


def event_waterline(scale_factor: int, relative_numerator: int, relative_denominator: int, interval_width: int) -> dict[str, Any]:
    """Quantize one event's relative drift position and its error budget.

    Boundary ownership uncertainty is one raw shard-key unit mapped into the
    fixed-point scale (``ceil(S / interval_width)``).  Range scaling and drift
    offsets each lose less than one fixed-point unit under toward-zero
    truncation.
    """

    drift_fp = quantize_ratio(relative_numerator, relative_denominator, scale_factor)
    if interval_width <= 0:
        raise ValueError("interval width must be positive")
    boundary_fp = saturate(ceil_div(scale_factor, interval_width))
    range_fp = 1
    drift_error_fp = 1
    total = saturate(boundary_fp + range_fp + drift_error_fp)
    return {
        **quantization_header(scale_factor),
        "drift_offset_fp": drift_fp,
        "boundary_ownership_error_fp": boundary_fp,
        "range_scaling_error_fp": range_fp,
        "drift_offset_error_fp": drift_error_fp,
        "single_value_error_fp": total,
    }


def _dominant(contributions: Mapping[str, int]) -> str:
    ordered = ("boundary_ownership", "range_scaling", "drift_offset")
    maximum = max(contributions[name] for name in ordered)
    if maximum == 0:
        return "none"
    for name in ordered:
        if contributions[name] == maximum:
            return name
    return "none"


def build_waterline(
    spec: RoutingSpec,
    event_results: Iterable[Mapping[str, Any]],
    *,
    peak_in_flight: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build deterministic per-database and global waterline summaries.

    ``event_results`` are the canonical event conclusions produced by routing;
    their already-quantized per-event waterline is aggregated per database.
    Peak actual in-flight probe counts may be supplied by the scheduler.
    """

    peak_in_flight = peak_in_flight or {}
    ordered_databases = sorted(spec.databases, key=lambda item: item.id)
    per_database: dict[str, dict[str, Any]] = {
        database.id: {
            "database_id": database.id,
            "lease_limit_fp": quantize_count(database.lease_limit, spec.scale_factor),
            "in_flight_fp": quantize_count(peak_in_flight.get(database.id, 0), spec.scale_factor),
            "drift_offset_fp": 0,
            "boundary_ownership_error_fp": 0,
            "range_scaling_error_fp": 0,
            "drift_offset_error_fp": 0,
            "event_count": 0,
        }
        for database in ordered_databases
    }

    for event in event_results:
        route = event.get("route") or {}
        database_id = route.get("database_id")
        waterline = event.get("waterline") or {}
        if database_id not in per_database or not waterline:
            continue
        row = per_database[database_id]
        row["event_count"] += 1
        row["drift_offset_fp"] = saturate(row["drift_offset_fp"] + int(waterline["drift_offset_fp"]))
        row["boundary_ownership_error_fp"] = saturate(
            row["boundary_ownership_error_fp"] + int(waterline["boundary_ownership_error_fp"])
        )
        row["range_scaling_error_fp"] = saturate(
            row["range_scaling_error_fp"] + int(waterline["range_scaling_error_fp"])
        )
        row["drift_offset_error_fp"] = saturate(
            row["drift_offset_error_fp"] + int(waterline["drift_offset_error_fp"])
        )

    global_split = {
        "boundary_ownership": 0,
        "range_scaling": 0,
        "drift_offset": 0,
    }
    max_single_error = 0
    for row in per_database.values():
        boundary = row["boundary_ownership_error_fp"]
        scaling = row["range_scaling_error_fp"]
        drift = row["drift_offset_error_fp"]
        single = saturate(boundary + scaling + drift)
        row["single_value_error_fp"] = single
        row["waterline_comparison_error_bound_fp"] = saturate(single * 2)
        row["headroom_fp"] = saturate(
            row["lease_limit_fp"] - row["in_flight_fp"] - row["drift_offset_fp"]
        )
        row["waterline_compare"] = {
            "left_fp": row["lease_limit_fp"],
            "right_fp": saturate(row["in_flight_fp"] + row["drift_offset_fp"]),
            "relation": (
                "gt"
                if row["lease_limit_fp"] > row["in_flight_fp"] + row["drift_offset_fp"]
                else "eq"
                if row["lease_limit_fp"] == row["in_flight_fp"] + row["drift_offset_fp"]
                else "lt"
            ),
            "boundary_ownership": boundary,
            "range_scaling": scaling,
            "drift_offset": drift,
            "dominant": _dominant({"boundary_ownership": boundary, "range_scaling": scaling, "drift_offset": drift}),
        }
        global_split["boundary_ownership"] = saturate(global_split["boundary_ownership"] + boundary)
        global_split["range_scaling"] = saturate(global_split["range_scaling"] + scaling)
        global_split["drift_offset"] = saturate(global_split["drift_offset"] + drift)
        max_single_error = max(max_single_error, single)

    global_error = saturate(sum(global_split.values()))
    return {
        "format": quantization_header(spec.scale_factor),
        "per_database": [per_database[database.id] for database in ordered_databases],
        "decision_error_bound_fp": saturate(max_single_error * 2),
        "global_occupancy_error_bound_fp": global_error,
        "error_contributions": {
            **global_split,
            "dominant_segment": _dominant(global_split),
        },
        "aggregation_rule": "sum_of_saturated_per_database_bounds",
    }


__all__ = [
    "FIXED_MAX",
    "FIXED_MIN",
    "build_waterline",
    "ceil_div",
    "event_waterline",
    "quantization_header",
    "quantize_count",
    "quantize_ratio",
    "saturate",
    "trunc_div",
]
