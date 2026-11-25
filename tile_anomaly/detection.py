"""Fixed-point robust statistics, Mahalanobis distance, KDE, and diagnostics."""

from __future__ import annotations

import math
from typing import Any

from .features import DIMENSION_NAMES, compute_window_feature
from .tiles import Mosaic, TileData
from .util import decimal_string, median_doubled, sha256_json
from .windows import WindowSpec


ROBUST_STAT_VERSION = "robust-median-mad-v1"
SCORE_ALGORITHM_VERSION = "mahalanobis-kde-fixed-v1"


def _window_order(record: dict[str, Any]) -> tuple[int, int, int, int]:
    core = record["core_rect"]
    return (int(core[0]), int(core[1]), int(core[2]), int(core[3]))


def _statistics_payload(features: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "robust_stat_version": ROBUST_STAT_VERSION,
        "feature_keys": [record["feature_key"] for record in features],
        "minimum_robust_scale_micro": config["detector"]["minimum_robust_scale_micro"],
    }


def statistics_key_for_features(
    features: list[dict[str, Any]], config: dict[str, Any]
) -> str:
    ordered = sorted(features, key=_window_order)
    return sha256_json(_statistics_payload(ordered, config))


def compute_robust_statistics(
    features: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    ordered = sorted(features, key=_window_order)
    dimension_stats: list[dict[str, Any]] = []
    minimum_scale = int(config["detector"]["minimum_robust_scale_micro"])
    for dimension_index, dimension_name in enumerate(DIMENSION_NAMES):
        values = [int(record["observation"][dimension_index]) for record in ordered]
        center_doubled = median_doubled(values)
        deviations_doubled = [abs(2 * value - center_doubled) for value in values]
        mad_deviation_doubled = median_doubled(deviations_doubled)
        scale_micro = (mad_deviation_doubled * 14826 * 1_000_000) // 40000
        if scale_micro < minimum_scale:
            scale_micro = minimum_scale
        dimension_stats.append(
            {
                "name": dimension_name,
                "center_doubled": center_doubled,
                "center": center_doubled / 2.0,
                "center_decimal": decimal_string(center_doubled / 2.0),
                "mad_deviation_doubled": mad_deviation_doubled,
                "scale_micro": scale_micro,
                "scale": scale_micro / 1_000_000.0,
                "scale_decimal": decimal_string(scale_micro / 1_000_000.0),
                "rounding": "integer_median_doubled_then_14826_over_10000",
            }
        )
    key_payload = _statistics_payload(ordered, config)
    return {
        "statistics_key": sha256_json(key_payload),
        "dependency": {
            "feature_keys": [record["feature_key"] for record in ordered],
            "minimum_robust_scale_micro": minimum_scale,
        },
        "window_order": [record["window_id"] for record in ordered],
        "dimensions": dimension_stats,
    }


def _fixed_exp_negative(x_micro: int) -> int:
    if x_micro <= 0:
        return 1_000_000
    if x_micro >= 20_000_000:
        return 0
    shift_bits = 0
    reduced = x_micro
    while reduced > 1000:
        reduced >>= 1
        shift_bits += 1
    unit = 1_000_000_000_000
    term = unit
    result = unit
    for order in range(1, 11):
        term = -(term * reduced) // (1_000_000 * order)
        result += term
    for _ in range(shift_bits):
        result = (result * result) // unit
    return (result * 1_000_000) // unit


def _score_config_key(config: dict[str, Any]) -> str:
    detector = config["detector"]
    return sha256_json(
        {
            "score_algorithm": SCORE_ALGORITHM_VERSION,
            "kde_bandwidth_milli": detector["kde_bandwidth_milli"],
            "distance_weight_milli": detector["distance_weight_milli"],
            "density_weight_milli": detector["density_weight_milli"],
            "minimum_robust_scale_micro": detector["minimum_robust_scale_micro"],
        }
    )


def score_key_for_feature(
    feature: dict[str, Any],
    statistics_key: str,
    feature_set_key: str,
    config: dict[str, Any],
) -> str:
    score_config_key = _score_config_key(config)
    return sha256_json(
        {
            "score_algorithm": SCORE_ALGORITHM_VERSION,
            "score_config_key": score_config_key,
            "statistics_key": statistics_key,
            "feature_set_key": feature_set_key,
            "feature_key": feature["feature_key"],
            "window_id": feature["window_id"],
        }
    )


def _standardized_vectors(
    features: list[dict[str, Any]], statistics: dict[str, Any]
) -> list[dict[str, Any]]:
    standardized = []
    for record in features:
        vector_micro = []
        squared_sum = 0
        for dimension_index, dimension in enumerate(statistics["dimensions"]):
            value = int(record["observation"][dimension_index])
            center_doubled = int(dimension["center_doubled"])
            scale_micro = int(dimension["scale_micro"])
            delta_doubled = 2 * value - center_doubled
            standardized_micro = (
                delta_doubled * 1_000_000_000_000 // (2 * scale_micro)
            )
            vector_micro.append(standardized_micro)
            squared_sum += standardized_micro * standardized_micro
        standardized.append(
            {
                "window_id": record["window_id"],
                "vector_micro": vector_micro,
                "squared_sum": squared_sum,
                "mahalanobis_micro": math.isqrt(squared_sum),
            }
        )
    return standardized


def prepare_score_context(
    features: list[dict[str, Any]],
    statistics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(features, key=_window_order)
    standardized = _standardized_vectors(ordered, statistics)
    distances = [entry["mahalanobis_micro"] for entry in standardized]
    bandwidth_micro = int(config["detector"]["kde_bandwidth_milli"]) * 1000
    bandwidth_squared = bandwidth_micro * bandwidth_micro
    densities = []
    for row_index, distance in enumerate(distances):
        density_sum = 0
        for column_index, reference_distance in enumerate(distances):
            delta = distance - reference_distance
            exponent_micro = (delta * delta * 1_000_000) // (2 * bandwidth_squared)
            density_sum += _fixed_exp_negative(exponent_micro)
        densities.append(density_sum // max(1, len(distances)))
    maximum_distance = max(distances) if distances else 0
    maximum_density = max(densities) if densities else 0
    return {
        "ordered": ordered,
        "standardized": standardized,
        "distances": distances,
        "densities": densities,
        "maximum_distance": maximum_distance,
        "maximum_density": maximum_density,
        "score_config_key": _score_config_key(config),
        "feature_set_key": sha256_json(
            [record["feature_key"] for record in ordered]
        ),
        "bandwidth_micro": bandwidth_micro,
    }


def score_from_context(
    index: int,
    record: dict[str, Any],
    context: dict[str, Any],
    statistics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    distances = context["distances"]
    densities = context["densities"]
    maximum_distance = int(context["maximum_distance"])
    maximum_density = int(context["maximum_density"])
    distance = int(distances[index])
    density = int(densities[index])
    distance_weight = int(config["detector"]["distance_weight_milli"])
    density_weight = int(config["detector"]["density_weight_milli"])
    if maximum_distance > 0:
        distance_normalized = min(1_000_000, distance * 1_000_000 // maximum_distance)
    else:
        distance_normalized = 0
    if maximum_density > 0:
        density_normalized = min(1_000_000, density * 1_000_000 // maximum_density)
    else:
        density_normalized = 0
    novelty = 1_000_000 - density_normalized
    score_micro = (
        distance_weight * distance_normalized + density_weight * novelty
    ) // 1000
    score_micro = max(0, min(1_000_000, score_micro))
    score_key = sha256_json(
        {
            "score_algorithm": SCORE_ALGORITHM_VERSION,
            "score_config_key": context["score_config_key"],
            "statistics_key": statistics["statistics_key"],
            "feature_set_key": context["feature_set_key"],
            "feature_key": record["feature_key"],
            "window_id": record["window_id"],
        }
    )
    return {
        "window_id": record["window_id"],
        "owner_tile_id": record["owner_tile_id"],
        "feature_key": record["feature_key"],
        "statistics_key": statistics["statistics_key"],
        "feature_set_key": context["feature_set_key"],
        "score_key": score_key,
        "standardized_vector_micro": context["standardized"][index]["vector_micro"],
        "mahalanobis_distance_micro": distance,
        "mahalanobis_distance": distance / 1_000_000.0,
        "mahalanobis_distance_decimal": decimal_string(distance / 1_000_000.0),
        "kde_density_fixed": density,
        "kde_density_normalized_micro": density_normalized,
        "kde_novelty_micro": novelty,
        "distance_normalized_micro": distance_normalized,
        "distance_weight_milli": distance_weight,
        "density_weight_milli": density_weight,
        "kde_bandwidth_milli": config["detector"]["kde_bandwidth_milli"],
        "score_micro": score_micro,
        "score_permille_decimal": decimal_string(score_micro / 1000.0),
        "accumulation_order": "window_row_col_ascending_dimension_ascending_kde_reference_ascending",
    }


def compute_window_scores(
    features: list[dict[str, Any]],
    statistics: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    context = prepare_score_context(features, statistics, config)
    ordered = context["ordered"]
    scores = []
    for index, record in enumerate(ordered):
        scores.append(score_from_context(index, record, context, statistics, config))
    return scores


def threshold_decisions(scores: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    threshold_micro = int(config["detector"]["threshold_permille"]) * 1000
    final_step = int(config["detector"]["final_score_step_micro"])
    half_step = final_step // 2
    decisions = []
    for score in scores:
        score_micro = int(score["score_micro"])
        difference = score_micro - threshold_micro
        boundary = abs(difference) <= half_step
        decisions.append(
            {
                "window_id": score["window_id"],
                "owner_tile_id": score["owner_tile_id"],
                "score_key": score["score_key"],
                "score_micro": score_micro,
                "threshold_permille": config["detector"]["threshold_permille"],
                "threshold_micro": threshold_micro,
                "difference_micro": difference,
                "boundary": boundary,
                "half_step_micro": half_step,
                "rule": "anomalous_when_score_ge_threshold_equality_anomalous",
                "is_anomaly": score_micro >= threshold_micro,
            }
        )
    return decisions


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_boundary_diagnostics(
    windows: list[WindowSpec],
    features: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    statistics: dict[str, Any],
    mosaic: Mosaic,
    tiles: dict[str, TileData],
    config: dict[str, Any],
) -> dict[str, Any]:
    boundary_decisions = [decision for decision in decisions if decision["boundary"]]
    if not boundary_decisions:
        return {
            "boundary_count": 0,
            "flip_count": 0,
            "records": [],
            "higher_width_fraction_bits": config["quantization"]["fraction_bits"] + 1,
        }
    high_bits = int(config["quantization"]["fraction_bits"]) + 1
    high_features = [
        compute_window_feature(window, mosaic, tiles, config, fraction_bits=high_bits)
        for window in windows
    ]
    high_statistics = compute_robust_statistics(high_features, config)
    high_scores = compute_window_scores(high_features, high_statistics, config)
    feature_by_window = {record["window_id"]: record for record in features}
    high_feature_by_window = {record["window_id"]: record for record in high_features}
    score_by_window = {record["window_id"]: record for record in scores}
    high_score_by_window = {record["window_id"]: record for record in high_scores}
    threshold_micro = int(config["detector"]["threshold_permille"]) * 1000
    final_step = int(config["detector"]["final_score_step_micro"])
    records = []
    flip_count = 0
    for decision in boundary_decisions:
        window_id = decision["window_id"]
        current_feature = feature_by_window[window_id]
        reference_feature = high_feature_by_window[window_id]
        current_score = score_by_window[window_id]
        reference_score = high_score_by_window[window_id]
        current_anomaly = bool(decision["is_anomaly"])
        reference_anomaly = int(reference_score["score_micro"]) >= threshold_micro
        current_index_values = [
            current_feature["dimensions"][0]["dequantized"],
            current_feature["dimensions"][1]["dequantized"],
        ]
        reference_index_values = [
            reference_feature["dimensions"][0]["dequantized"],
            reference_feature["dimensions"][1]["dequantized"],
        ]
        ratio_deviation = _mean(
            [
                abs(reference_value - current_value)
                / max(1.0e-12, abs(current_value))
                for current_value, reference_value in zip(
                    current_index_values, reference_index_values
                )
            ]
        )
        scale_deviation = _mean(
            [
                abs(
                    int(reference_dimension["scale_micro"])
                    - int(current_dimension["scale_micro"])
                )
                / max(1, int(current_dimension["scale_micro"]))
                for current_dimension, reference_dimension in zip(
                    statistics["dimensions"], high_statistics["dimensions"]
                )
            ]
        )
        threshold_deviation = abs(current_score["score_micro"] - threshold_micro) / 1_000_000.0
        score_delta = int(reference_score["score_micro"]) - int(current_score["score_micro"])
        if reference_anomaly == current_anomaly:
            primary_cause = None
        else:
            flip_count += 1
            if abs(score_delta) <= final_step:
                primary_cause = "threshold_comparison"
            elif ratio_deviation >= scale_deviation:
                primary_cause = "index_ratio_quantization"
            else:
                primary_cause = "robust_scale_rounding"
        records.append(
            {
                "window_id": window_id,
                "owner_tile_id": current_feature["owner_tile_id"],
                "current_fraction_bits": config["quantization"]["fraction_bits"],
                "reference_fraction_bits": high_bits,
                "current_score_micro": current_score["score_micro"],
                "reference_score_micro": reference_score["score_micro"],
                "score_delta_micro": score_delta,
                "current_decision": current_anomaly,
                "reference_decision": reference_anomaly,
                "decision_flipped": reference_anomaly != current_anomaly,
                "primary_flip_cause": primary_cause,
                "deviations": {
                    "index_ratio_quantization_relative": decimal_string(ratio_deviation),
                    "robust_scale_estimation_rounding_relative": decimal_string(scale_deviation),
                    "threshold_comparison_micro": int(
                        abs(current_score["score_micro"] - threshold_micro)
                    ),
                    "threshold_comparison_permille": decimal_string(threshold_deviation),
                    "score_delta_micro": score_delta,
                },
                "current_index_dequantized": current_index_values,
                "reference_index_dequantized": reference_index_values,
                "current_scales": [
                    dimension["scale_micro"] for dimension in statistics["dimensions"]
                ],
                "reference_scales": [
                    dimension["scale_micro"] for dimension in high_statistics["dimensions"]
                ],
            }
        )
    return {
        "boundary_count": len(records),
        "flip_count": flip_count,
        "higher_width_fraction_bits": high_bits,
        "records": records,
    }


def summarize_tiles(
    tiles: dict[str, TileData],
    features: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    score_by_window = {record["window_id"]: record for record in scores}
    decision_by_window = {record["window_id"]: record for record in decisions}
    by_tile: dict[str, list[dict[str, Any]]] = {}
    for feature in features:
        by_tile.setdefault(feature["owner_tile_id"], []).append(feature)
    summaries: dict[str, dict[str, Any]] = {}
    for tile_id in sorted(tiles, key=lambda item: (tiles[item].spec.row, tiles[item].spec.col)):
        tile_features = sorted(by_tile.get(tile_id, []), key=_window_order)
        window_scores = [score_by_window[record["window_id"]] for record in tile_features]
        window_decisions = [decision_by_window[record["window_id"]] for record in tile_features]
        score_values = [int(record["score_micro"]) for record in window_scores]
        anomaly_count = sum(1 for record in window_decisions if record["is_anomaly"])
        summary_key = sha256_json(
            {
                "summary_version": "tile-summary-v1",
                "tile_id": tile_id,
                "score_keys": [record["score_key"] for record in window_scores],
                "threshold_permille": config["detector"]["threshold_permille"],
                "tile_content_key": tiles[tile_id].content_key,
            }
        )
        summaries[tile_id] = {
            "tile_id": tile_id,
            "row": tiles[tile_id].spec.row,
            "col": tiles[tile_id].spec.col,
            "tile_content_key": tiles[tile_id].content_key,
            "summary_key": summary_key,
            "window_count": len(tile_features),
            "window_ids": [record["window_id"] for record in tile_features],
            "window_score_keys": [record["score_key"] for record in window_scores],
            "anomaly_window_count": anomaly_count,
            "boundary_window_count": sum(
                1 for record in window_decisions if record["boundary"]
            ),
            "max_score_micro": max(score_values) if score_values else 0,
            "min_score_micro": min(score_values) if score_values else 0,
            "mean_score_micro": (
                sum(score_values) // len(score_values) if score_values else 0
            ),
            "threshold_permille": config["detector"]["threshold_permille"],
        }
    return summaries
