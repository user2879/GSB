"""Directional reference construction and error diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .canonical import canonical_hash, json_safe
from .config import Config
from .sh import fit_weighted_sh, sh_component_count, synth_energy_field
from .sphere import (
    EquirectangularGrid,
    angular_distance_deg,
    area_weighted_normalize,
    energy_centroid,
    equirectangular_grid,
    nearest_indices,
)


def build_truth_reference(
    manifest: dict[str, Any],
    grid: EquirectangularGrid,
    kernel_width_deg: float,
    support_deg: float,
) -> np.ndarray:
    """Build the reference field from generator-emitted paths only.

    The interface is intentionally a manifest object, never reconstructed
    coefficients or energy.  Each path contributes a Gaussian-in-angle von
    Mises kernel centred on its generator-provided direction.
    """

    directions: list[list[float]] = []
    energies: list[float] = []
    for source in manifest["sources"]:
        for path in source["paths"]:
            directions.append(path["direction"])
            energies.append(float(path["rendered_energy"]))
    if not directions:
        return np.zeros(len(grid.directions), dtype=np.float64)
    path_directions = np.asarray(directions, dtype=np.float64)
    path_energies = np.asarray(energies, dtype=np.float64)
    width = float(kernel_width_deg)
    if not (0.0 < width < 90.0):
        raise ValueError("reference_kernel_deg must be in (0, 90)")
    support = min(float(support_deg), 89.0)
    kappa = math.log(2.0) / max(1.0 - math.cos(math.radians(width)), 1.0e-15)
    chord = 2.0 * math.sin(math.radians(support) / 2.0)
    tree = cKDTree(grid.directions)
    neighbours = tree.query_ball_point(path_directions, r=chord)
    reference = np.zeros(len(grid.directions), dtype=np.float64)
    for path_index, indices in enumerate(neighbours):
        if not indices:
            continue
        cosine = np.clip(grid.directions[indices] @ path_directions[path_index], -1.0, 1.0)
        kernel = np.exp(kappa * (cosine - 1.0))
        reference[indices] += path_energies[path_index] * kernel
    reference /= float(len(manifest["sources"]))
    return area_weighted_normalize(reference, grid.area_weights)


def _relative_quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def field_metrics(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    grid: EquirectangularGrid,
    low_confidence_grid: np.ndarray,
    tolerance: float,
    significant_floor: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reconstructed_raw = np.asarray(reconstructed, dtype=np.float64)
    reference_raw = np.asarray(reference, dtype=np.float64)
    reconstructed_norm = area_weighted_normalize(reconstructed_raw, grid.area_weights)
    reference_norm = area_weighted_normalize(reference_raw, grid.area_weights)
    maximum_reference = float(np.max(reference_norm)) if reference_norm.size else 0.0
    significant = reference_norm >= maximum_reference * float(significant_floor)
    if not np.any(significant):
        significant = reference_norm > 0.0
    denominator = np.maximum(reference_norm, 1.0e-15)
    relative = np.abs(reconstructed_norm - reference_norm) / denominator
    relative = np.where(significant, relative, 0.0)
    weighted_relative = np.where(significant, relative, np.nan)
    over = significant & (relative > float(tolerance))
    significant_count = int(np.sum(significant))
    over_count = int(np.sum(over))

    reference_centroid = energy_centroid(grid.directions, reference_norm, grid.area_weights)
    reconstructed_centroid = energy_centroid(grid.directions, reconstructed_norm, grid.area_weights)
    centroid_deviation = angular_distance_deg(reference_centroid, reconstructed_centroid)
    angular_weight = reference_norm * grid.area_weights
    if float(np.sum(angular_weight)) > 0.0:
        angular_weight = angular_weight / float(np.sum(angular_weight))
    angular_weighted_deviation = float(
        np.sum(
            angular_weight
            * np.degrees(
                np.arccos(
                    np.clip(grid.directions @ reconstructed_centroid, -1.0, 1.0)
                )
            )
        )
    )
    l1 = float(np.sum(np.abs(reconstructed_norm - reference_norm) * grid.area_weights) / 2.0)

    if over_count:
        candidate_indices = np.flatnonzero(over)
        worst_index = int(candidate_indices[np.argmax(relative[candidate_indices])])
    elif significant_count:
        candidate_indices = np.flatnonzero(significant)
        worst_index = int(candidate_indices[np.argmax(relative[candidate_indices])])
    else:
        worst_index = 0
    worst_coordinate = grid.directions[worst_index].tolist()
    summary = {
        "relative_tolerance": float(tolerance),
        "significant_direction_count": significant_count,
        "over_tolerance_direction_count": over_count,
        "over_tolerance_direction_ratio": float(over_count / significant_count)
        if significant_count
        else 0.0,
        "error_interval": _relative_quantiles(weighted_relative[np.isfinite(weighted_relative)]),
        "angular_weighted_deviation_deg": angular_weighted_deviation,
        "centroid_angular_deviation_deg": centroid_deviation,
        "normalized_l1_energy_error": l1,
        "worst_direction": {
            "index": worst_index,
            "azimuth_deg": float(grid.azimuth_deg[worst_index]),
            "elevation_deg": float(grid.elevation_deg[worst_index]),
            "unit_vector": worst_coordinate,
            "reference_energy": float(reference_norm[worst_index]),
            "reconstructed_energy": float(reconstructed_norm[worst_index]),
            "relative_error": float(relative[worst_index]),
            "low_confidence": bool(low_confidence_grid[worst_index]),
            "over_tolerance": bool(over[worst_index]),
        },
        "low_confidence_share": float(np.mean(low_confidence_grid)),
        "tolerance_met": bool(over_count == 0),
    }
    arrays = {
        "reconstructed_normalized": reconstructed_norm,
        "reference_normalized": reference_norm,
        "relative_error": relative,
        "over_tolerance": over.astype(np.int8),
        "significant": significant.astype(np.int8),
    }
    return summary, arrays


def _low_confidence_on_grid(projection: dict[str, Any], grid: EquirectangularGrid) -> np.ndarray:
    sample_directions = np.asarray(projection["sample_directions"], dtype=np.float64)
    low = np.asarray(projection["low_confidence"], dtype=np.int8).astype(bool)
    nearest = nearest_indices(grid.directions, sample_directions)
    return low[nearest]


def evaluate_tier(
    config: Config,
    tier: dict[str, Any],
    manifest: dict[str, Any],
    projection: dict[str, Any],
    coefficients: np.ndarray,
    decision: dict[str, Any],
) -> dict[str, Any]:
    validation_config = config.validation()
    grid = equirectangular_grid(float(validation_config["longitude_step_deg"]))
    reference = build_truth_reference(
        manifest,
        grid,
        float(validation_config["reference_kernel_deg"]),
        float(validation_config["reference_kernel_support_deg"]),
    )
    low_confidence_grid = _low_confidence_on_grid(projection, grid)
    reconstructed = synth_energy_field(coefficients, grid.directions)
    summary, arrays = field_metrics(
        reconstructed,
        reference,
        grid,
        low_confidence_grid,
        float(validation_config["relative_tolerance"]),
        float(validation_config["significant_relative_floor"]),
    )

    directions = np.asarray(projection["sample_directions"], dtype=np.float64)
    counts = np.asarray(projection["effective_sample_count"], dtype=np.int64)
    energy = np.asarray(projection["observed_energy"], dtype=np.float64)
    high_mask = counts >= int(config.projection()["min_effective_samples"])
    rows = np.flatnonzero(high_mask)
    row_values = energy[rows]
    row_weights = np.ones(len(rows), dtype=np.float64)
    supported_order = int(decision["inputs"]["supported_order"])
    order_sweep: list[dict[str, Any]] = []
    for order in range(0, supported_order + 1):
        components = sh_component_count(order)
        if len(rows) < components:
            order_sweep.append(
                {
                    "order": order,
                    "components": components,
                    "density": int(tier["sphere_points"]),
                    "status": "unsupported",
                    "reason": "fewer high-confidence directions than SH components",
                }
            )
            continue
        fit = fit_weighted_sh(
            directions[rows],
            row_values,
            row_weights,
            order,
            ridge=float(config.projection()["ridge"]),
        )
        field = synth_energy_field(fit.coefficients, grid.directions)
        order_summary, _ = field_metrics(
            field,
            reference,
            grid,
            low_confidence_grid,
            float(validation_config["relative_tolerance"]),
            float(validation_config["significant_relative_floor"]),
        )
        order_sweep.append(
            {
                "order": order,
                "components": components,
                "density": int(tier["sphere_points"]),
                "status": "ok",
                "rank": int(fit.rank),
                "residual_rms": float(fit.residual_rms),
                "error_interval": order_summary["error_interval"],
                "over_tolerance_direction_ratio": order_summary[
                    "over_tolerance_direction_ratio"
                ],
                "angular_weighted_deviation_deg": order_summary[
                    "angular_weighted_deviation_deg"
                ],
            }
        )

    fields = {
        "shape": list(grid.shape),
        "azimuth_deg": grid.azimuth_deg.tolist(),
        "elevation_deg": grid.elevation_deg.tolist(),
        "area_weights": grid.area_weights.tolist(),
        "reconstructed_normalized": arrays["reconstructed_normalized"].tolist(),
        "reference_normalized": arrays["reference_normalized"].tolist(),
        "relative_error": arrays["relative_error"].tolist(),
        "low_confidence_grid": low_confidence_grid.astype(np.int8).tolist(),
        "over_tolerance": arrays["over_tolerance"].tolist(),
        "significant": arrays["significant"].tolist(),
    }
    validation = {
        "schema": "spatial-diagnostic-validation/1",
        "tier": tier["name"],
        "density": int(tier["sphere_points"]),
        "selected_order": int(decision["selected_order"]),
        "selected_strategy": decision["selected_strategy"],
        "truth_source": "RIR generator manifest paths[].direction and paths[].rendered_energy",
        "reference_construction": (
            "per-source path-energy average convolved with a von Mises angular kernel; "
            "never derived from reconstructed coefficients"
        ),
        "selected": summary,
        "order_sweep": order_sweep,
        "error_distribution_policy": (
            "relative errors are reported as p50/p90/p95/max over significant reference "
            "directions; the complete per-direction list is persisted in fields.relative_error"
        ),
    }
    validation["content_hash"] = canonical_hash(validation)
    return json_safe({"validation": validation, "fields": fields})
