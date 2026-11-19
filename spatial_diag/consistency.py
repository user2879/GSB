"""Cross-tier direction-by-direction consistency diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .canonical import canonical_hash, json_safe
from .config import Config
from .sh import component_descriptor, real_sh


def _dominant_component(direction: np.ndarray, coefficients: np.ndarray) -> int:
    max_order = int(np.sqrt(len(coefficients))) - 1
    basis = real_sh(np.asarray(direction, dtype=np.float64)[None, :], max_order)[0]
    contribution = np.abs(coefficients * basis)
    return int(np.argmax(contribution))


def _coefficient_divergence(
    left: dict[str, Any], right: dict[str, Any], tolerance: float
) -> dict[str, Any] | None:
    left_coefficients = np.asarray(left["coefficients"], dtype=np.float64)
    right_coefficients = np.asarray(right["coefficients"], dtype=np.float64)
    shared = min(len(left_coefficients), len(right_coefficients))
    left_norm = max(float(np.linalg.norm(left_coefficients[:shared])), 1.0e-15)
    right_norm = max(float(np.linalg.norm(right_coefficients[:shared])), 1.0e-15)
    left_normalized = left_coefficients[:shared] / left_norm
    right_normalized = right_coefficients[:shared] / right_norm
    relative = np.abs(left_normalized - right_normalized) / np.maximum(
        np.maximum(np.abs(left_normalized), np.abs(right_normalized)), 1.0e-15
    )
    candidates = np.flatnonzero(relative > float(tolerance))
    if len(candidates) == 0:
        return None
    index = int(candidates[0])
    descriptor = component_descriptor(index)
    return {
        "coefficient_index": index,
        "component": descriptor,
        "left_normalized_value": float(left_normalized[index]),
        "right_normalized_value": float(right_normalized[index]),
        "relative_difference": float(relative[index]),
        "shared_component_count": shared,
        "tolerance": float(tolerance),
    }


def _reason(
    low_confidence_at_direction: bool,
    left: dict[str, Any],
    right: dict[str, Any],
    coefficient_divergence: dict[str, Any] | None,
) -> str:
    if low_confidence_at_direction:
        return "sampling_density_insufficient"
    if left["decision"]["selected_order"] != right["decision"]["selected_order"]:
        if coefficient_divergence is not None:
            return "order_mismatch"
        return "sampling_density_insufficient"
    if left["decision"]["selected_strategy"] != right["decision"]["selected_strategy"]:
        return "strategy_choice_difference"
    return "sampling_density_insufficient"


def assess_consistency(config: Config, tiers: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare every pair direction by direction, never by aggregate error alone."""

    tolerance = float(config.consistency()["relative_tolerance"])
    coefficient_tolerance = float(config.consistency()["coefficient_relative_tolerance"])
    pairs: list[dict[str, Any]] = []
    for left_index in range(len(tiers)):
        for right_index in range(left_index + 1, len(tiers)):
            left = tiers[left_index]
            right = tiers[right_index]
            left_field = np.asarray(left["fields"]["reconstructed_normalized"], dtype=np.float64)
            right_field = np.asarray(right["fields"]["reconstructed_normalized"], dtype=np.float64)
            if len(left_field) != len(right_field):
                raise ValueError("cross-tier fields use different evaluation grids")
            denominator = np.maximum(
                (np.abs(left_field) + np.abs(right_field)) / 2.0, 1.0e-15
            )
            relative = np.abs(left_field - right_field) / denominator
            reference_left = np.asarray(
                left["fields"]["reference_normalized"], dtype=np.float64
            )
            reference_right = np.asarray(
                right["fields"]["reference_normalized"], dtype=np.float64
            )
            reference_envelope = np.maximum(reference_left, reference_right)
            significant = reference_envelope >= (
                float(np.max(reference_envelope)) * 0.05 if len(reference_envelope) else 0.0
            )
            diverging = significant & (relative > tolerance)
            low_left = np.asarray(left["fields"]["low_confidence_grid"], dtype=np.int8).astype(bool)
            low_right = np.asarray(right["fields"]["low_confidence_grid"], dtype=np.int8).astype(bool)
            low_any = low_left | low_right
            azimuth = np.asarray(left["fields"]["azimuth_deg"], dtype=np.float64)
            elevation = np.asarray(left["fields"]["elevation_deg"], dtype=np.float64)
            directions = np.asarray(
                [
                    [
                        np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                        np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                        np.sin(np.radians(lat)),
                    ]
                    for lon, lat in zip(azimuth, elevation)
                ],
                dtype=np.float64,
            )
            coefficient_divergence = _coefficient_divergence(
                left, right, coefficient_tolerance
            )

            candidate_indices = np.flatnonzero(diverging)
            first_direction = None
            if len(candidate_indices):
                left_coefficients = np.asarray(left["coefficients"], dtype=np.float64)
                right_coefficients = np.asarray(right["coefficients"], dtype=np.float64)
                keys: list[tuple[int, float, float, int]] = []
                for index in candidate_indices:
                    component_key = min(
                        _dominant_component(directions[index], left_coefficients),
                        _dominant_component(directions[index], right_coefficients),
                    )
                    keys.append(
                        (
                            component_key,
                            float(azimuth[index]),
                            float(elevation[index]),
                            int(index),
                        )
                    )
                component_key, _, _, index = sorted(keys)[0]
                component = component_descriptor(component_key)
                reason = _reason(bool(low_any[index]), left, right, coefficient_divergence)
                first_direction = {
                    "index": int(index),
                    "azimuth_deg": float(azimuth[index]),
                    "elevation_deg": float(elevation[index]),
                    "unit_vector": directions[index].tolist(),
                    "sh_component_index": int(component_key),
                    "sh_component": component,
                    "left_energy": float(left_field[index]),
                    "right_energy": float(right_field[index]),
                    "relative_difference": float(relative[index]),
                    "low_confidence_left": bool(low_left[index]),
                    "low_confidence_right": bool(low_right[index]),
                    "reason": reason,
                }

            values = relative[significant]
            pair = {
                "left_tier": left["decision"]["tier"],
                "right_tier": right["decision"]["tier"],
                "left_sources": left["decision"]["inputs"]["source_positions"],
                "right_sources": right["decision"]["inputs"]["source_positions"],
                "left_density": left["decision"]["inputs"]["sphere_points"],
                "right_density": right["decision"]["inputs"]["sphere_points"],
                "left_order": left["decision"]["selected_order"],
                "right_order": right["decision"]["selected_order"],
                "left_strategy": left["decision"]["selected_strategy"],
                "right_strategy": right["decision"]["selected_strategy"],
                "relative_tolerance": tolerance,
                "direction_count": int(len(relative)),
                "significant_direction_count": int(np.sum(significant)),
                "divergence_count": int(np.sum(diverging)),
                "divergence_ratio": float(np.sum(diverging) / np.sum(significant))
                if np.any(significant)
                else 0.0,
                "relative_difference_interval": {
                    "p50": float(np.quantile(values, 0.50)) if len(values) else 0.0,
                    "p90": float(np.quantile(values, 0.90)) if len(values) else 0.0,
                    "p95": float(np.quantile(values, 0.95)) if len(values) else 0.0,
                    "max": float(np.max(values)) if len(values) else 0.0,
                },
                "first_coefficient_divergence": coefficient_divergence,
                "first_direction_divergence": first_direction,
                "consistent": bool(first_direction is None),
                "ordering": (
                    "direction divergences sorted by (dominant ACN component index, "
                    "azimuth deg, elevation deg); coefficient divergences sorted by ACN index"
                ),
            }
            pairs.append(pair)

    report = {
        "schema": "spatial-diagnostic-consistency/1",
        "tolerance": tolerance,
        "coefficient_tolerance": coefficient_tolerance,
        "listening_point_m": config.room()["listener_m"],
        "pairs": pairs,
        "consistent": all(pair["consistent"] for pair in pairs),
        "policy": (
            "agreement is decided direction by direction; aggregate error similarity is "
            "never used as a substitute for the first-divergence test"
        ),
    }
    report["content_hash"] = canonical_hash(report)
    return json_safe(report)
