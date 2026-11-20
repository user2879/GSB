"""Integer fixed-point pool metrics and order-independent error propagation.

Quantization model
------------------
Inputs are integer basis points (1 bp = 1/10000).  We treat a declared rate as
the rounding of an underlying exact rate to the nearest basis point, so its
rounding half-width is 0.5 bp.  Separately, the deterministic hash maps into
10001 buckets (0..10000).  A threshold ``w`` selects exactly ``w`` buckets out
of 10001, while the declared probability is ``w/10000``; that discretization
is also propagated.  Both are conservative bounds, not claims about random
sampling noise.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable, Mapping

from .hashing import BUCKET_SPACE


BASIS_POINTS = 10000
FIXED_SCALE = 1000
RATE_QUANTUM_BP = Fraction(1, 2)


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _scaled_bound(value: Fraction) -> int:
    return _ceil_fraction(value * FIXED_SCALE)


def _share_bp(part: Fraction, total: Fraction) -> int:
    if total <= 0:
        return 0
    return int((part / total) * BASIS_POINTS)


def _as_outcomes(outcomes: Iterable[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for outcome in outcomes:
        result.append(outcome if isinstance(outcome, dict) else outcome.to_dict())
    return result


def build_error_report(
    config: Any,
    outcomes: Iterable[Any],
    pool_stats: Iterable[Any],
) -> dict[str, Any]:
    """Build a deterministic error report from aggregate counts only.

    No component iterates over the *ordered* request list.  Contributions are
    accumulated in maps and serialized in sorted key order, so reversing the
    request traversal cannot change the attribution.
    """

    outcome_dicts = _as_outcomes(outcomes)
    stats_by_pool: dict[str, dict[str, Any]] = {}
    for item in pool_stats:
        value = item if isinstance(item, dict) else item.to_dict()
        stats_by_pool[value["pool_id"]] = value

    request_counts: dict[str, int] = {}
    actual_gray: dict[str, int] = {}
    actual_mirror: dict[str, int] = {}
    for outcome in outcome_dicts:
        rule_id = outcome.get("rule_id")
        if not rule_id:
            continue
        request_counts[rule_id] = request_counts.get(rule_id, 0) + 1
        if outcome.get("main_destination") == "gray":
            actual_gray[rule_id] = actual_gray.get(rule_id, 0) + 1
        if outcome.get("mirrored"):
            actual_mirror[rule_id] = actual_mirror.get(rule_id, 0) + 1

    # Exact contribution bounds are accumulated as Fractions keyed by sorted
    # dimensions.  Every later loop sorts the keys.
    pool_contrib: dict[str, list[dict[str, Any]]] = {}
    mirror_contrib: dict[str, list[dict[str, Any]]] = {}

    def add(
        target: dict[str, list[dict[str, Any]]],
        pool_id: str,
        *,
        rule_id: str,
        component: str,
        level: str,
        exact_count_bound: Fraction,
    ) -> None:
        target.setdefault(pool_id, []).append(
            {
                "rule_id": rule_id,
                "component": component,
                "level": level,
                "exact_count_bound": exact_count_bound,
            }
        )

    for rule_id in sorted(config.rules):
        rule = config.rules[rule_id]
        count = request_counts.get(rule_id, 0)
        if count == 0:
            continue
        weight = rule.gray_weight_bp
        sample = rule.mirror_sample_bp

        rate_bound = Fraction(count, 1) * RATE_QUANTUM_BP / BASIS_POINTS
        gray_bucket_bound = Fraction(count * weight, BASIS_POINTS * BUCKET_SPACE)
        stable_bucket_bound = Fraction(count * (BASIS_POINTS - weight), BASIS_POINTS * BUCKET_SPACE)
        add(
            pool_contrib,
            rule.gray_pool,
            rule_id=rule_id,
            component="rate_quantization_gray",
            level="rate_to_basis_points",
            exact_count_bound=rate_bound,
        )
        add(
            pool_contrib,
            rule.gray_pool,
            rule_id=rule_id,
            component="hash_bucket_discretization_gray",
            level="hash_threshold_discretization",
            exact_count_bound=gray_bucket_bound,
        )
        add(
            pool_contrib,
            rule.stable_pool,
            rule_id=rule_id,
            component="rate_quantization_stable",
            level="rate_to_basis_points",
            exact_count_bound=rate_bound,
        )
        add(
            pool_contrib,
            rule.stable_pool,
            rule_id=rule_id,
            component="hash_bucket_discretization_stable",
            level="hash_threshold_discretization",
            exact_count_bound=stable_bucket_bound,
        )
        mirror_rate = Fraction(count, 1) * RATE_QUANTUM_BP / BASIS_POINTS
        mirror_bucket = Fraction(count * sample, BASIS_POINTS * BUCKET_SPACE)
        add(
            mirror_contrib,
            rule.mirror_pool,
            rule_id=rule_id,
            component="rate_quantization_mirror_sample",
            level="rate_to_basis_points",
            exact_count_bound=mirror_rate,
        )
        add(
            mirror_contrib,
            rule.mirror_pool,
            rule_id=rule_id,
            component="hash_bucket_discretization_mirror_sample",
            level="hash_threshold_discretization",
            exact_count_bound=mirror_bucket,
        )

    def render_contributions(source: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
        rendered: dict[str, Any] = {}
        fixed_point_by_pool: dict[str, Fraction] = {}
        for pool_id in sorted(source):
            entries = sorted(
                source[pool_id],
                key=lambda item: (item["rule_id"], item["level"], item["component"]),
            )
            exact_total = sum((item["exact_count_bound"] for item in entries), Fraction(0, 1))
            # One fixed-point truncation unit is added for the averaged pool
            # watermark.  Counts themselves are exact integers and need none.
            fixed_point_by_pool[pool_id] = Fraction(1, FIXED_SCALE)
            serial_entries = []
            for item in entries:
                exact = item["exact_count_bound"]
                serial_entries.append(
                    {
                        "rule_id": item["rule_id"],
                        "component": item["component"],
                        "level": item["level"],
                        "count_bound_scaled": _scaled_bound(exact),
                        "share_bp": _share_bp(exact, exact_total),
                    }
                )
            bound_scaled = _scaled_bound(exact_total + fixed_point_by_pool[pool_id])
            observed = stats_by_pool.get(pool_id, {})
            observed_average = int(observed.get("average_watermark_scaled", 0))
            observed_borrows = int(observed.get("borrow_count_scaled", 0))
            borrow_bound = _scaled_bound(exact_total)
            rendered[pool_id] = {
                "aggregate_error_upper_bound_scaled": bound_scaled,
                "borrow_count_error_upper_bound_scaled": borrow_bound,
                "fixed_point_truncation_scaled": 1,
                "observed_average_watermark_scaled": observed_average,
                "watermark_interval_scaled": {
                    "lower": max(0, observed_average - bound_scaled),
                    "upper": observed_average + bound_scaled,
                },
                "observed_borrow_count_scaled": observed_borrows,
                "borrow_count_interval_scaled": {
                    "lower": max(0, observed_borrows - borrow_bound),
                    "upper": observed_borrows + borrow_bound,
                },
                "propagation_chain": [
                    {
                        "stage": "quantized_request_or_sample_count",
                        "bound_scaled": _scaled_bound(exact_total),
                    },
                    {
                        "stage": "pool_watermark",
                        "bound_scaled": bound_scaled,
                    },
                    {
                        "stage": "borrow_count",
                        "bound_scaled": borrow_bound,
                    },
                    {
                        "stage": "fixed_point_truncation",
                        "bound_scaled": 1,
                    },
                ],
                "contributions": serial_entries,
            }
        return rendered

    pools_report = render_contributions(pool_contrib)
    mirrors_report = render_contributions(mirror_contrib)

    # Stable, deterministic comparison of the stable and gray watermarks for
    # every rule.  When within the combined error band, the stable destination
    # wins the tie regardless of request arrival order.
    comparisons: list[dict[str, Any]] = []
    for rule_id in sorted(config.rules):
        rule = config.rules[rule_id]
        stable_stats = stats_by_pool.get(rule.stable_pool, {})
        gray_stats = stats_by_pool.get(rule.gray_pool, {})
        stable_scaled = int(stable_stats.get("average_watermark_scaled", 0))
        gray_scaled = int(gray_stats.get("average_watermark_scaled", 0))
        stable_error = int(
            pools_report.get(rule.stable_pool, {}).get(
                "aggregate_error_upper_bound_scaled", 1
            )
        )
        gray_error = int(
            pools_report.get(rule.gray_pool, {}).get("aggregate_error_upper_bound_scaled", 1)
        )
        combined = stable_error + gray_error
        delta = abs(gray_scaled - stable_scaled)
        within = delta <= combined
        if within:
            ordered = [rule.stable_pool, rule.gray_pool]
            winner = "stable"
        else:
            ordered = sorted(
                [rule.stable_pool, rule.gray_pool],
                key=lambda pool_id: (
                    -int(
                        (
                            gray_stats
                            if pool_id == rule.gray_pool
                            else stable_stats
                        ).get("average_watermark_scaled", 0)
                    ),
                    pool_id,
                ),
            )
            winner = "gray" if gray_scaled > stable_scaled else "stable"
        comparisons.append(
            {
                "rule_id": rule_id,
                "stable_pool": rule.stable_pool,
                "gray_pool": rule.gray_pool,
                "stable_watermark_scaled": stable_scaled,
                "gray_watermark_scaled": gray_scaled,
                "delta_scaled": delta,
                "combined_error_upper_bound_scaled": combined,
                "within_error_bound": within,
                "winner_by_stable_tiebreak": winner,
                "stable_order": ordered,
            }
        )

    source_levels: dict[str, Fraction] = {}
    for mapping in (pool_contrib, mirror_contrib):
        for entries in mapping.values():
            for item in entries:
                source_levels[item["level"]] = source_levels.get(
                    item["level"], Fraction(0, 1)
                ) + item["exact_count_bound"]
    total_source = sum(source_levels.values(), Fraction(0, 1))
    level_report = [
        {
            "level": level,
            "bound_scaled": _scaled_bound(value),
            "share_bp": _share_bp(value, total_source),
        }
        for level, value in sorted(source_levels.items())
    ]

    actual_rates = []
    for rule_id in sorted(config.rules):
        rule = config.rules[rule_id]
        count = request_counts.get(rule_id, 0)
        if not count:
            continue
        actual_rates.append(
            {
                "rule_id": rule_id,
                "request_count": count,
                "declared_gray_weight_bp": rule.gray_weight_bp,
                "observed_gray_bp": (actual_gray.get(rule_id, 0) * BASIS_POINTS) // count,
                "declared_mirror_sample_bp": rule.mirror_sample_bp,
                "observed_mirror_sample_bp": (
                    actual_mirror.get(rule_id, 0) * BASIS_POINTS
                )
                // count,
            }
        )

    return {
        "model": {
            "ordinal_rate_scale": BASIS_POINTS,
            "rate_quantum_bp_scaled": RATE_QUANTUM_BP.numerator * 1000 // RATE_QUANTUM_BP.denominator,
            "hash_bucket_space": BUCKET_SPACE,
            "fixed_point_scale": FIXED_SCALE,
            "truncation": "floor_toward_negative_infinity",
            "order_independent": True,
        },
        "levels": level_report,
        "pool_errors": {key: pools_report[key] for key in sorted(pools_report)},
        "mirror_pool_errors": {key: mirrors_report[key] for key in sorted(mirrors_report)},
        "watermark_comparisons": comparisons,
        "observed_rates": actual_rates,
    }


def numpy_contribution_matrix(report: Mapping[str, Any]) -> Any:
    """Optional NumPy view used by tests to validate share aggregation.

    NumPy is deliberately not required on the runtime path.  If installed, it
    provides an independent vectorized check of the integer shares.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("NumPy is required for numpy_contribution_matrix") from exc
    rows = []
    for pool_id in sorted(report.get("pool_errors", {})):
        values = [item["share_bp"] for item in report["pool_errors"][pool_id]["contributions"]]
        rows.append((pool_id, values, sum(values)))
    width = max((len(values) for _, values, _ in rows), default=0)
    matrix = np.full((len(rows), width), np.nan, dtype=np.float64)
    for index, (_, values, _) in enumerate(rows):
        matrix[index, : len(values)] = values
    return {
        "pool_ids": [row[0] for row in rows],
        "shares": matrix,
        "row_sums": np.array([row[2] for row in rows], dtype=np.int64),
    }
