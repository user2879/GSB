from __future__ import annotations

from decimal import Decimal
from typing import Any

from .config import LoadedConfig
from .fixedpoint import half_step_q, round_div
from .hashing import fingerprint


WINDOW_NAMES = ("reference", "observation")


def select_rows(
    product: dict[str, Any], start_seq: int, end_seq: int
) -> list[dict[str, Any]]:
    return [
        row
        for row in product["rows"]
        if start_seq <= int(row["seq"]) <= end_seq
    ]


def compute_feature_stats(
    product: dict[str, Any],
    start_seq: int,
    end_seq: int,
    window_name: str,
) -> list[dict[str, Any]]:
    rows = select_rows(product, start_seq, end_seq)
    result: list[dict[str, Any]] = []
    feature_count = len(product["features"])
    for feature_index, feature in enumerate(product["features"]):
        values = [int(row["values"][feature_index]) for row in rows]
        flags = [bool(row.get("imputed", [False] * feature_count)[feature_index]) for row in rows]
        count = len(values)
        sample_set = sorted(int(row["seq"]) for row in rows)
        if count:
            sum_q = sum(values)
            mean_q = round_div(sum_q, count)
            sum_sq_q2 = sum(v * v for v in values)
            variance_q2 = max(0, round_div(sum((v - mean_q) ** 2 for v in values), count))
            min_q = min(values)
            max_q = max(values)
        else:
            sum_q = mean_q = sum_sq_q2 = variance_q2 = 0
            min_q = max_q = 0
        result.append(
            {
                "stage": product["stage"],
                "feature_index": feature_index,
                "feature_id": feature["id"],
                "feature_name": feature["name"],
                "window": window_name,
                "scale": product["scale"],
                "gear": product.get("gear"),
                "count": count,
                "missing_or_imputed_count": sum(flags),
                "sample_set": sample_set,
                "sample_set_fingerprint": fingerprint(sample_set),
                "sum_q": sum_q,
                "mean_q": mean_q,
                "min_q": min_q,
                "max_q": max_q,
                "sum_sq_q2": sum_sq_q2,
                "variance_q2": variance_q2,
            }
        )
    return result


def compute_stage_window_stats(
    product: dict[str, Any], config: LoadedConfig
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for window_name in WINDOW_NAMES:
        window = config.raw["windows"][window_name]
        output[window_name] = compute_feature_stats(
            product,
            int(window["start_seq"]),
            int(window["end_seq"]),
            window_name,
        )
    return output


def drift_decision(
    reference: dict[str, Any],
    observation: dict[str, Any],
    threshold_bp: int,
    gear: str,
    denominator_floor_q: int = 1,
) -> dict[str, Any]:
    scale = int(reference["scale"])
    ref_mean = int(reference["mean_q"])
    obs_mean = int(observation["mean_q"])
    delta_q = abs(obs_mean - ref_mean)
    denominator_q = max(abs(ref_mean), max(1, int(denominator_floor_q)))

    left = delta_q * 10_000
    threshold_right = int(threshold_bp) * denominator_q
    is_drift = left >= threshold_right
    drift_bp_rounded = round_div(delta_q * 10_000, denominator_q)

    required_delta = Decimal(threshold_bp) * Decimal(denominator_q) / Decimal(10_000)
    distance_to_threshold_q = abs(Decimal(delta_q) - required_delta)
    half_step = half_step_q(scale, gear)
    exact_equality = left == threshold_right
    boundary = distance_to_threshold_q <= half_step
    if exact_equality:
        boundary_reason = "exact_threshold_equality"
    elif boundary:
        boundary_reason = "within_half_gear_step"
    else:
        boundary_reason = None

    return {
        "feature_id": observation["feature_id"],
        "feature_name": observation["feature_name"],
        "threshold_bp": int(threshold_bp),
        "drift": bool(is_drift),
        "boundary": bool(boundary),
        "boundary_reason": boundary_reason,
        "decision_rule": "delta_bp >= threshold_bp => drift; exact equality is drift",
        "reference_mean_q": ref_mean,
        "observation_mean_q": obs_mean,
        "delta_mean_q": delta_q,
        "denominator_q": denominator_q,
        "denominator_floor_q": max(1, int(denominator_floor_q)),
        "drift_bp_rounded": drift_bp_rounded,
        "exact_compare": {
            "delta_times_10000": left,
            "threshold_times_denominator": threshold_right,
            "relation": "ge" if is_drift else "lt",
        },
        "required_delta_q": format(required_delta, "f"),
        "distance_to_threshold_q": format(distance_to_threshold_q, "f"),
        "half_gear_step_q": format(half_step, "f"),
        "reference_sample_set": reference["sample_set"],
        "observation_sample_set": observation["sample_set"],
        "reference_count": reference["count"],
        "observation_count": observation["count"],
        "reference_imputed_count": reference["missing_or_imputed_count"],
        "observation_imputed_count": observation["missing_or_imputed_count"],
    }


def decide_all_drift(
    final_product: dict[str, Any],
    stats_by_window: dict[str, list[dict[str, Any]]],
    config: LoadedConfig,
) -> list[dict[str, Any]]:
    thresholds = config.raw["thresholds_bp"]
    floor_q = int(config.raw.get("drift_denominator_floor_q", 1))
    decisions: list[dict[str, Any]] = []
    for index, feature in enumerate(final_product["features"]):
        feature_id = feature["id"]
        threshold = thresholds.get(feature_id)
        if threshold is None:
            threshold = thresholds.get(feature["name"])
        if threshold is None:
            raise KeyError(f"missing threshold for feature {feature_id}")
        decisions.append(
            drift_decision(
                stats_by_window["reference"][index],
                stats_by_window["observation"][index],
                int(threshold),
                config.gear,
                floor_q,
            )
        )
    return decisions
