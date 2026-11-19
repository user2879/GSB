"""Residual attribution with exact additive reconciliation.

Three independent perturbations are evaluated: XYZ quantisation, sub-sampling
of the colour distribution, and a coarser integration convention.  Their
residual effects interact, so ordinary one-at-a-time deltas do not add to the
combined residual.  The ledger therefore reports both standalone deltas and
three-player Shapley contributions.  The Shapley values sum exactly to
``combined - baseline`` and the interaction share is shown explicitly.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from .canonical import sha256_obj, stable_float
from .errors import DiagnosticError, ErrorCode
from .matrix import (
    MatrixSolution,
    condition_diagnostics,
    matrix_offset_ledger,
    quantize_xyz,
    solve_adaptation,
)
from .scene import GeneratedScene
from .spectra import IntegrationConvention, integrate_reflectance

FACTORS = ("quantization", "distribution", "integration")
SETTINGS = tuple(
    dict(zip(FACTORS, bits))
    for bits in itertools.product((False, True), repeat=len(FACTORS))
)


def _fit_and_residual(
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    evaluation_source: np.ndarray,
    evaluation_target: np.ndarray,
    *,
    cond_threshold: float,
) -> tuple[float, MatrixSolution]:
    solution = solve_adaptation(fit_source, fit_target, cond_threshold=cond_threshold, allow_prune=False)
    if not solution.accepted or solution.matrix is None:
        raise DiagnosticError(
            code=ErrorCode.MATRIX_SINGULAR,
            message="an attribution perturbation produced a singular matrix",
            details=solution.to_dict(),
        )
    predictions = evaluation_source @ solution.matrix
    residual = predictions - evaluation_target
    rmse = float(np.sqrt(np.mean(residual**2)))
    return rmse, solution


def _distribution_drop_indices(source_xyz: np.ndarray, count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    diagnostics = condition_diagnostics(source_xyz)
    # Highest leverage first, then ascending index.  This is deterministic and
    # deliberately perturbs the most influential part of the sample cloud.
    order = sorted(range(source_xyz.shape[0]), key=lambda index: (-float(diagnostics.row_leverage[index]), index))
    return tuple(sorted(order[:count]))


def _fast_scene_xyz(
    scene: GeneratedScene,
    *,
    observer: str,
    normalization: str,
    density_nm: float,
    quadrature: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    convention = IntegrationConvention(
        observer=observer,
        density_nm=density_nm,
        quadrature=quadrature,
        normalization=normalization,
    )
    source_spd = np.asarray(scene.payload["source"]["spd_5nm"], dtype=float)
    camera_spd = np.asarray(scene.payload["camera"]["spd_5nm"], dtype=float)
    sample_reflectance = scene.source_reflectances
    material_reflectance = scene.material_reflectances
    sample_source = integrate_reflectance(sample_reflectance, source_spd, convention).xyz
    sample_target = integrate_reflectance(sample_reflectance, camera_spd, convention).xyz
    material_source = integrate_reflectance(material_reflectance, source_spd, convention).xyz
    material_target = integrate_reflectance(material_reflectance, camera_spd, convention).xyz
    return sample_source, sample_target, material_source, material_target


def _subset(source: np.ndarray, target: np.ndarray, subset: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    indices = list(subset)
    return source[indices], target[indices]


def compute_attribution(
    scene: GeneratedScene,
    baseline_solution: MatrixSolution,
    *,
    observer: str,
    normalization: str,
    quadrature: str,
    cond_threshold: float,
    quant_bits: int,
    quant_range: tuple[float, float],
    integration_density_nm: float,
    distribution_drop_count: int,
    base_indices: tuple[int, ...] | None = None,
    quantized_source_xyz: np.ndarray | None = None,
    quantized_target_xyz: np.ndarray | None = None,
    reference_source_xyz: np.ndarray | None = None,
    reference_target_xyz: np.ndarray | None = None,
    material_source_xyz: np.ndarray | None = None,
    material_target_xyz: np.ndarray | None = None,
    reference_density_nm: float | None = None,
) -> dict[str, Any]:
    reference_source = scene.source_xyz if reference_source_xyz is None else np.asarray(
        reference_source_xyz, dtype=float
    )
    reference_target = scene.target_xyz if reference_target_xyz is None else np.asarray(
        reference_target_xyz, dtype=float
    )
    material_source = scene.material_source_xyz if material_source_xyz is None else np.asarray(
        material_source_xyz, dtype=float
    )
    material_target = scene.material_target_xyz if material_target_xyz is None else np.asarray(
        material_target_xyz, dtype=float
    )
    if quantized_source_xyz is None or quantized_target_xyz is None:
        quantized_source, quant_meta_source = quantize_xyz(
            reference_source, quant_range=quant_range, quant_bits=quant_bits
        )
        quantized_target, quant_meta_target = quantize_xyz(
            reference_target, quant_range=quant_range, quant_bits=quant_bits
        )
        quant_meta = {
            "source": quant_meta_source,
            "target": quant_meta_target,
        }
    else:
        quantized_source = np.asarray(quantized_source_xyz, dtype=float)
        quantized_target = np.asarray(quantized_target_xyz, dtype=float)
        quant_meta = {
            "source": {"quant_step": (quant_range[1] - quant_range[0]) / ((1 << quant_bits) - 1)},
            "target": {"quant_step": (quant_range[1] - quant_range[0]) / ((1 << quant_bits) - 1)},
            "provided_by_caller": True,
        }

    fast_source, fast_target, _, _ = _fast_scene_xyz(
        scene,
        observer=observer,
        normalization=normalization,
        density_nm=integration_density_nm,
        quadrature=quadrature,
    )
    full_indices = tuple(range(reference_source.shape[0]))
    active_indices = full_indices if base_indices is None else tuple(sorted(base_indices))
    if len(active_indices) < 3:
        raise DiagnosticError(
            code=ErrorCode.MATRIX_SINGULAR,
            message="the accepted sample subset is too small for attribution",
            details={"base_indices": list(active_indices)},
        )
    drop_indices = _distribution_drop_indices(reference_source[list(active_indices)], distribution_drop_count)
    dropped_absolute = tuple(active_indices[position] for position in drop_indices)
    kept_indices = tuple(index for index in active_indices if index not in dropped_absolute)

    inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for setting in SETTINGS:
        source = quantized_source if setting["quantization"] else reference_source
        target = quantized_target if setting["quantization"] else reference_target
        if setting["integration"]:
            source = fast_source
            target = fast_target
        selected = kept_indices if setting["distribution"] else active_indices
        inputs[_key(setting)] = _subset(source, target, selected)

    residual_cache: dict[str, float] = {}
    solution_cache: dict[str, MatrixSolution] = {}
    for setting in SETTINGS:
        key = _key(setting)
        fit_source, fit_target = inputs[key]
        residual, solution = _fit_and_residual(
            fit_source,
            fit_target,
            reference_source,
            reference_target,
            cond_threshold=cond_threshold,
        )
        residual_cache[key] = residual
        solution_cache[key] = solution

    baseline_key = _key({factor: False for factor in FACTORS})
    combined_key = _key({factor: True for factor in FACTORS})
    baseline_residual = residual_cache[baseline_key]
    combined_residual = residual_cache[combined_key]
    baseline_solution = solution_cache[baseline_key]
    combined_solution = solution_cache[combined_key]

    shapley: dict[str, float] = {}
    standalone: dict[str, dict[str, Any]] = {}
    n = len(FACTORS)
    for factor in FACTORS:
        contribution = 0.0
        others = [value for value in FACTORS if value != factor]
        for size in range(n):
            for subset_tuple in itertools.combinations(others, size):
                subset = {value: False for value in FACTORS}
                for value in subset_tuple:
                    subset[value] = True
                without_key = _key(subset)
                with_subset = dict(subset)
                with_subset[factor] = True
                with_key = _key(with_subset)
                weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                contribution += weight * (residual_cache[with_key] - residual_cache[without_key])
        shapley[factor] = float(contribution)

        standalone_setting = {value: False for value in FACTORS}
        standalone_setting[factor] = True
        standalone_residual = residual_cache[_key(standalone_setting)]
        standalone[factor] = {
            "residual": stable_float(standalone_residual),
            "delta_from_baseline": stable_float(standalone_residual - baseline_residual),
            "shapley_delta": stable_float(contribution),
            "interaction_allocation": stable_float(contribution - (standalone_residual - baseline_residual)),
        }

    quant_key = _key({"quantization": True, "distribution": False, "integration": False})
    quant_solution = solution_cache[quant_key]
    offset_ledger = matrix_offset_ledger(baseline_solution, quant_solution)
    distribution_key = _key({"quantization": False, "distribution": True, "integration": False})
    integration_key = _key({"quantization": False, "distribution": False, "integration": True})

    total_delta = combined_residual - baseline_residual
    shapley_total = sum(shapley.values())
    tolerance = max(1e-10, abs(total_delta) * 1e-10 + 1e-12)
    additive = abs(shapley_total - total_delta) <= tolerance
    if not additive:
        raise DiagnosticError(
            code=ErrorCode.ATTRIB_MISMATCH,
            message="attribution contributions do not reconcile to the combined residual",
            details={
                "baseline_residual": baseline_residual,
                "combined_residual": combined_residual,
                "total_delta": total_delta,
                "shapley_total": shapley_total,
                "tolerance": tolerance,
            },
        )

    contributions = {
        factor: {
            "shapley_delta": stable_float(shapley[factor]),
            "standalone_residual": standalone[factor]["residual"],
            "standalone_delta": standalone[factor]["delta_from_baseline"],
            "interaction_allocation": standalone[factor]["interaction_allocation"],
        }
        for factor in FACTORS
    }

    material_baseline = material_source @ baseline_solution.matrix
    material_residual = material_baseline - material_target
    adaptation = {
        "baseline_matrix": baseline_solution.matrix.tolist(),
        "combined_matrix": combined_solution.matrix.tolist(),
        "material_adapted_xyz": material_baseline.tolist(),
        "material_truth_xyz": material_target.tolist(),
        "material_truth_residual": material_residual.tolist(),
        "material_truth_rmse": float(np.sqrt(np.mean(material_residual**2))),
        "matrix_offset_combined": matrix_offset_ledger(baseline_solution, combined_solution),
    }

    ledger: dict[str, Any] = {
        "schema_version": 1,
        "status": "reconciled" if additive else "mismatch",
        "reference": {
            "observer": observer,
            "normalization": normalization,
            "quadrature": quadrature,
            "reference_density_nm": scene.config.reference_density_nm
            if reference_density_nm is None
            else reference_density_nm,
        },
        "perturbations": {
            "quantization": {
                **quant_meta,
                "matrix_offset": offset_ledger,
                "reference_evaluation_residual": stable_float(residual_cache[quant_key]),
                "reference_evaluation_delta_from_baseline": stable_float(
                    residual_cache[quant_key] - baseline_residual
                ),
                "condition_number": None
                if quant_solution.condition_number is None
                else stable_float(quant_solution.condition_number),
                "condition_delta": None
                if quant_solution.condition_number is None or baseline_solution.condition_number is None
                else stable_float(quant_solution.condition_number - baseline_solution.condition_number),
            },
            "distribution": {
                "dropped_indices": list(dropped_absolute),
                "kept_indices": list(kept_indices),
                "residual": stable_float(residual_cache[distribution_key]),
                "condition_number": None
                if solution_cache[distribution_key].condition_number is None
                else stable_float(solution_cache[distribution_key].condition_number),
            },
            "integration": {
                "density_nm": integration_density_nm,
                "quadrature": quadrature,
                "residual": stable_float(residual_cache[integration_key]),
                "condition_number": None
                if solution_cache[integration_key].condition_number is None
                else stable_float(solution_cache[integration_key].condition_number),
            },
        },
        "residuals": {
            "baseline": stable_float(baseline_residual),
            "combined": stable_float(combined_residual),
            "total_delta": stable_float(total_delta),
            "standalone": standalone,
        },
        "contributions": contributions,
        "additivity": {
            "reconciled": additive,
            "sum_contributions": stable_float(shapley_total),
            "total_delta": stable_float(total_delta),
            "residual": stable_float(shapley_total - total_delta),
            "tolerance": stable_float(tolerance),
            "method": "three_factor_shapley",
        },
        "condition_numbers": {
            "baseline": None
            if baseline_solution.condition_number is None
            else stable_float(baseline_solution.condition_number),
            "quantization": None
            if quant_solution.condition_number is None
            else stable_float(quant_solution.condition_number),
            "combined": None
            if combined_solution.condition_number is None
            else stable_float(combined_solution.condition_number),
        },
        "adaptation": adaptation,
        "matrix_solution_hashes": {
            key: sha256_obj(
                {
                    "matrix": solution.matrix.tolist() if solution.matrix is not None else None,
                    "residual": residual_cache[key],
                    "used_indices": list(solution.used_indices),
                }
            )
            for key, solution in solution_cache.items()
        },
    }
    ledger["attribution_sha256"] = sha256_obj(ledger)
    return ledger


def _key(setting: dict[str, bool]) -> str:
    return "|".join(f"{factor}={int(setting[factor])}" for factor in FACTORS)


def verify_attribution(ledger: dict[str, Any]) -> None:
    additivity = ledger.get("additivity", {})
    contributions = ledger.get("contributions", {})
    if set(contributions) != set(FACTORS):
        raise DiagnosticError(
            code=ErrorCode.ATTRIB_MISMATCH,
            message="attribution ledger is missing a required factor",
            details={"expected": list(FACTORS), "actual": sorted(contributions)},
        )
    contribution_sum = sum(float(contributions[factor]["shapley_delta"]) for factor in FACTORS)
    total_delta = float(additivity.get("total_delta", 0.0))
    tolerance = max(1e-10, abs(total_delta) * 1e-10 + 1e-12)
    if not additivity.get("reconciled") or abs(contribution_sum - total_delta) > tolerance:
        raise DiagnosticError(
            code=ErrorCode.ATTRIB_MISMATCH,
            message="attribution ledger cannot be recomputed additively",
            details={
                "contribution_sum": contribution_sum,
                "total_delta": total_delta,
                "tolerance": tolerance,
            },
        )
    stored_hash = ledger.get("attribution_sha256")
    without_hash = {key: value for key, value in ledger.items() if key != "attribution_sha256"}
    if stored_hash != sha256_obj(without_hash):
        raise DiagnosticError(
            code=ErrorCode.ATTRIB_MISMATCH,
            message="attribution ledger hash does not match its content",
            details={"stored": stored_hash, "computed": sha256_obj(without_hash)},
        )
