"""Scale-adaptive SH fitting strategies and the decision ledger."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .canonical import canonical_hash, json_safe
from .config import Config
from .sh import fit_weighted_sh, sh_component_count, synth_energy_field
from .sphere import cluster_directions, farthest_point_representatives


STRATEGY_FULL = "full_least_squares"
STRATEGY_CLUSTERED = "clustered_representatives"
STRATEGY_COARSE = "coarse_to_fine"


def resolution_bound_deg(order: int) -> float:
    """Declared SH angular resolution envelope."""

    return 90.0 / float(int(order) + 1)


def sampling_spacing_deg(sphere_points: int) -> float:
    return float(np.degrees(math.sqrt(4.0 * math.pi / float(sphere_points))))


def max_supported_order(n_high: int, sphere_points: int, requested: int, oversampling: float) -> int:
    best = 0
    density_cap = max(0, math.isqrt(int(sphere_points)) - 1)
    for order in range(0, min(int(requested), density_cap) + 1):
        components = sh_component_count(order)
        if components > sphere_points:
            break
        if float(components) * float(oversampling) <= float(n_high):
            best = order
        else:
            break
    return best


def min_order_for_resolution_budget(budget_deg: float, requested: int) -> int:
    for order in range(0, int(requested) + 1):
        if resolution_bound_deg(order) <= float(budget_deg) + 1.0e-12:
            return order
    return int(requested)


def estimate_memory_bytes(sphere_points: int, rows: int, order: int) -> int:
    components = sh_component_count(order)
    sample_map = int(sphere_points) * (3 + 2 + 2 + 2 + 1) * 8
    design = int(rows) * components * 8
    design_work = int(rows) * components * 8
    normal = components * components * 8
    vectors = components * 8 * 2
    return int(sample_map + design + design_work + normal + vectors)


def max_rows_for_memory(sphere_points: int, order: int, memory_budget: int) -> int:
    components = sh_component_count(order)
    fixed = sphere_points * (3 + 2 + 2 + 2 + 1) * 8 + components * components * 8 + components * 16
    per_row = components * 8 * 2 + 8
    available = int(memory_budget) - int(fixed)
    if available <= 0:
        return 0
    return int(available // per_row)


def _projection_arrays(projection: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions = np.asarray(projection["sample_directions"], dtype=np.float64)
    weights = np.asarray(projection["sample_weights"], dtype=np.float64)
    counts = np.asarray(projection["effective_sample_count"], dtype=np.int64)
    energy = np.asarray(projection["observed_energy"], dtype=np.float64)
    return directions, weights, counts, energy


def _fit_rows_full(
    high_mask: np.ndarray,
    directions: np.ndarray,
    counts: np.ndarray,
    energy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.flatnonzero(high_mask)
    row_counts = counts[rows].astype(np.float64)
    # Each equal-area direction is one observation of the binned directional
    # energy.  Counts establish confidence, not the target magnitude.
    values = energy[rows]
    weights = np.ones_like(row_counts)
    return rows, values, weights


def _coarse_to_fine_fit(
    directions: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    target_order: int,
    ridge: float,
) -> tuple[Any, list[dict[str, Any]]]:
    ladder: list[dict[str, Any]] = []
    final_fit = None
    for order in range(0, int(target_order) + 1):
        fit = fit_weighted_sh(directions, values, weights, order, ridge=ridge)
        ladder.append(
            {
                "order": order,
                "components": sh_component_count(order),
                "rank": fit.rank,
                "residual_rms": fit.residual_rms,
            }
        )
        final_fit = fit
    if final_fit is None:
        raise ValueError("coarse-to-fine requires at least one order")
    return final_fit, ladder


def _clustered_fit(
    directions: np.ndarray,
    counts: np.ndarray,
    energy: np.ndarray,
    high_mask: np.ndarray,
    target_rows: int,
    target_order: int,
    ridge: float,
) -> tuple[Any, dict[str, Any]]:
    high_rows = np.flatnonzero(high_mask)
    if len(high_rows) == 0:
        raise ValueError("cannot cluster an empty high-confidence fit set")
    target_rows = max(sh_component_count(target_order), min(int(target_rows), len(high_rows)))
    representatives = farthest_point_representatives(
        directions[high_rows], counts[high_rows].astype(np.float64), target_rows
    )
    representative_indices = high_rows[representatives]
    representative_directions = directions[representative_indices]
    assignment = np.argmax(directions @ representative_directions.T, axis=1).astype(np.int64)
    cluster_of = assignment.tolist()
    cluster_sizes = np.bincount(assignment, minlength=len(representative_indices)).astype(np.int64)
    cluster_counts = np.bincount(
        assignment, weights=counts.astype(np.float64), minlength=len(representative_indices)
    )
    cluster_energy = np.bincount(
        assignment, weights=energy.astype(np.float64), minlength=len(representative_indices)
    )
    cluster_high = np.bincount(
        assignment, weights=high_mask.astype(np.float64), minlength=len(representative_indices)
    )
    fit_mask = cluster_high > 0
    fit_counts = cluster_counts[fit_mask]
    fit_sizes = cluster_sizes[fit_mask].astype(np.float64)
    fit_values = cluster_energy[fit_mask] / np.maximum(fit_sizes, 1.0)
    fit_weights = fit_sizes
    fit_directions = representative_directions[fit_mask]
    fit = fit_weighted_sh(fit_directions, fit_values, fit_weights, target_order, ridge=ridge)
    details = {
        "cluster_count": int(len(representative_indices)),
        "fit_cluster_count": int(np.sum(fit_mask)),
        "representative_direction_indices": representative_indices.tolist(),
        "cluster_of_direction": cluster_of,
        "cluster_sizes": cluster_sizes.tolist(),
        "cluster_event_counts": cluster_counts.astype(np.int64).tolist(),
        "cluster_event_count_total": int(np.sum(cluster_counts)),
        "original_event_count_total": int(np.sum(counts)),
        "all_directions_assigned_exactly_once": bool(len(cluster_of) == len(directions)),
        "cluster_conservation": bool(int(np.sum(cluster_counts)) == int(np.sum(counts))),
        "low_confidence_clusters_excluded": bool(np.any(~fit_mask)),
    }
    return fit, details


def select_and_fit(
    config: Config,
    tier: dict[str, Any],
    projection: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    directions, sample_weights, counts, energy = _projection_arrays(projection)
    high_mask = counts >= int(config.projection()["min_effective_samples"])
    n_high = int(np.sum(high_mask))
    if n_high == 0:
        raise ValueError(f"tier {tier['name']} has no high-confidence directions")
    sphere_points = int(tier["sphere_points"])
    requested_order = int(tier["requested_order"])
    density_cap = max(0, math.isqrt(sphere_points) - 1)
    requested_order = min(requested_order, density_cap)
    oversampling = float(config.projection()["oversampling_factor"])
    ridge = float(config.projection()["ridge"])
    supported_order = max_supported_order(n_high, sphere_points, requested_order, oversampling)
    resolution_budget = float(tier["direction_resolution_budget_deg"])
    minimum_order = min_order_for_resolution_budget(resolution_budget, requested_order)
    memory_budget = int(tier["memory_budget_bytes"])
    spacing = sampling_spacing_deg(sphere_points)

    candidates: list[dict[str, Any]] = []
    for strategy in (STRATEGY_FULL, STRATEGY_CLUSTERED, STRATEGY_COARSE):
        candidates.append(
            {
                "strategy": strategy,
                "description": {
                    STRATEGY_FULL: "weighted least squares on all high-confidence sampling directions",
                    STRATEGY_CLUSTERED: "one fitted representative per deterministic direction cluster",
                    STRATEGY_COARSE: "nested order ladder from 0 through the supported order",
                }[strategy],
            }
        )

    degradations: list[dict[str, Any]] = []
    strategy: str
    chosen_order: int
    target_rows: int = n_high
    cluster_details: dict[str, Any] | None = None
    ladder: list[dict[str, Any]] | None = None
    reason: str

    if supported_order < requested_order:
        strategy = STRATEGY_COARSE
        chosen_order = max(supported_order, 0)
        reason = (
            "requested order is unsupported at this density/confidence level; "
            "using a nested coarse-to-fine ladder through the supported order"
        )
        degradations.append(
            {
                "kind": "order_reduction",
                "priority": 1,
                "from_order": requested_order,
                "to_order": chosen_order,
                "reason": f"{n_high} eligible directions cannot support {sh_component_count(requested_order)} components at oversampling {oversampling:.2f}",
            }
        )
    else:
        full_bytes = estimate_memory_bytes(sphere_points, n_high, requested_order)
        fitting_order = None
        if full_bytes <= memory_budget:
            fitting_order = requested_order
        else:
            for candidate_order in range(requested_order - 1, minimum_order - 1, -1):
                if estimate_memory_bytes(sphere_points, n_high, candidate_order) <= memory_budget:
                    fitting_order = candidate_order
                    break
        if fitting_order is not None:
            strategy = STRATEGY_FULL
            chosen_order = fitting_order
            target_rows = n_high
            if fitting_order < requested_order:
                reason = "full least squares after memory-driven order reduction"
                degradations.append(
                    {
                        "kind": "order_reduction",
                        "priority": 1,
                        "from_order": requested_order,
                        "to_order": fitting_order,
                        "reason": "full design matrix exceeded the tier memory budget",
                    }
                )
            else:
                reason = "all high-confidence directions fit within resolution and memory budgets"
        else:
            strategy = STRATEGY_CLUSTERED
            chosen_order = requested_order
            max_rows = max_rows_for_memory(sphere_points, chosen_order, memory_budget)
            if max_rows < sh_component_count(chosen_order):
                raise ValueError(
                    "memory budget cannot hold even an SH-sized clustered design matrix"
                )
            target_rows = min(n_high, max_rows)
            reason = (
                "no full design matrix at or above the required resolution order fits memory; "
                "using deterministic direction-cluster representatives"
            )
            degradations.append(
                {
                    "kind": "cluster_representatives",
                    "priority": 2,
                    "from_rows": n_high,
                    "to_rows": target_rows,
                    "reason": "full design matrix exceeded the tier memory budget",
                }
            )
            if target_rows < n_high:
                degradations.append(
                    {
                        "kind": "direction_reduction",
                        "priority": 3,
                        "from_rows": n_high,
                        "to_rows": target_rows,
                        "reason": "fit rows reduced by merging, never by dropping an original direction",
                    }
                )

    if strategy == STRATEGY_CLUSTERED:
        fit, cluster_details = _clustered_fit(
            directions,
            counts,
            energy,
            high_mask,
            target_rows,
            chosen_order,
            ridge,
        )
        fit_rows = int(cluster_details["fit_cluster_count"])
    else:
        rows, values, weights = _fit_rows_full(high_mask, directions, counts, energy)
        fit_rows = len(rows)
        if strategy == STRATEGY_COARSE:
            fit, ladder = _coarse_to_fine_fit(
                directions[rows], values, weights, chosen_order, ridge
            )
        else:
            fit = fit_weighted_sh(
                directions[rows], values, weights, chosen_order, ridge=ridge
            )

    peak_bytes = estimate_memory_bytes(sphere_points, fit_rows, chosen_order)
    sampling_bound = spacing / math.sqrt(float(max(fit_rows, 1)))
    error_bound = max(resolution_bound_deg(chosen_order), sampling_bound)
    before_error_bound = max(resolution_bound_deg(requested_order), sampling_bound)
    coefficients = fit.coefficients
    energy_probe = synth_energy_field(coefficients, directions)
    decision: dict[str, Any] = {
        "schema": "spatial-diagnostic-strategy-decision/1",
        "tier": tier["name"],
        "inputs": {
            "source_positions": int(tier["source_positions"]),
            "sphere_points": sphere_points,
            "requested_order": requested_order,
            "resolution_budget_deg": resolution_budget,
            "memory_budget_bytes": memory_budget,
            "minimum_resolution_order": minimum_order,
            "supported_order": supported_order,
            "high_confidence_directions": n_high,
            "oversampling_factor": oversampling,
        },
        "candidates": candidates,
        "selected_strategy": strategy,
        "selected_order": chosen_order,
        "selected_components": sh_component_count(chosen_order),
        "selection_reason": reason,
        "fit_rows": fit_rows,
        "fit_rank": int(fit.rank),
        "fit_residual_rms": float(fit.residual_rms),
        "peak_memory_bytes": int(peak_bytes),
        "memory_budget_met": bool(peak_bytes <= memory_budget),
        "resolution_error_bound_deg": float(error_bound),
        "resolution_budget_met": bool(error_bound <= resolution_budget + 1.0e-9),
        "error_bound_before_degradation_deg": float(before_error_bound),
        "error_bound_after_degradation_deg": float(error_bound),
        "sampling_spacing_deg": float(spacing),
        "degradations": degradations,
        "ladder": ladder,
        "cluster": cluster_details,
        "conservation": {
            "all_sampling_directions_accounted_for": bool(
                cluster_details is None
                or len(cluster_details["cluster_of_direction"]) == sphere_points
            ),
            "energy_not_duplicated": bool(
                cluster_details is None or cluster_details["cluster_conservation"]
            ),
            "finite_coefficients": bool(np.all(np.isfinite(coefficients))),
            "finite_energy_probe": bool(np.all(np.isfinite(energy_probe))),
        },
        "data_signature": {
            "n_high_confidence": n_high,
            "event_total": int(projection["event_total"]),
            "projection_hash": projection["content_hash"],
        },
    }
    decision["decision_hash"] = canonical_hash(decision)
    return coefficients, json_safe(decision)
