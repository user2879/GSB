"""Per-series deterministic fit pipeline with stage caching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Any, Mapping

from .attribution import attribute_stage_errors
from .basis import BasisResult, build_basis, column_names
from .cache import CacheStore
from .errors import E_REFERENCE_INVALID, FitError
from .fingerprints import (
    CACHE_STAGES,
    all_stage_keys,
    config_computation_fingerprint,
    referenced_observations,
    series_content_fingerprint,
)
from .fixedpoint import accumulate_normal_equations, solve_fixed_point
from .interpolation import decimal_from_float, interpolate_series
from .jsonutil import fingerprint
from .model import CacheAction, FitConfig, SeriesRecord, assert_width
from .reference import (
    fit_unquantized,
    pad_vector,
    solve_integer_normal_in_float,
)


@dataclass(frozen=True)
class CoreSeriesResult:
    artifact: dict[str, Any]
    actions: tuple[CacheAction, ...]

    @property
    def series_id(self) -> str:
        return str(self.artifact["series_id"])

    @property
    def coefficients(self) -> tuple[int, ...]:
        return tuple(int(item["value"]) for item in self.artifact["coefficients"])

    @property
    def trend_slope(self) -> int:
        return int(self.artifact["trend_slope"])


def _quantize_values(values: tuple[float, ...], config: FitConfig) -> tuple[int, ...]:
    scale = Decimal(config.quantization_scale)
    quantized: list[int] = []
    with localcontext() as context:
        context.prec = 60
        for index, value in enumerate(values):
            raw = decimal_from_float(value) * scale
            fixed = int(raw.to_integral_value(rounding=ROUND_HALF_UP))
            assert_width(fixed, config.fixed_point_bits, f"quantized observed value[{index}]")
            quantized.append(fixed)
    return tuple(quantized)


def _coefficient_entries(
    config: FitConfig,
    basis: BasisResult,
    values: tuple[int, ...],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for column, value in zip(basis.columns, values):
        entries.append(
            {
                "index": column["index"],
                "name": column["name"],
                "kind": column["kind"],
                "harmonic_order": column["harmonic_order"],
                "phase": column["phase"],
                "phase_radians": column["phase_radians"],
                "value": int(value),
            }
        )
    return entries


def _validate_finite_reference(vector: tuple[float, ...], label: str) -> None:
    for index, value in enumerate(vector):
        if not math.isfinite(value):
            raise FitError(
                E_REFERENCE_INVALID,
                f"{label} reference coefficient {index} is non-finite",
            )


def _cache_actions(
    keys: Mapping[str, str], *, hit: bool, reason: str
) -> tuple[CacheAction, ...]:
    action = "hit" if hit else "miss"
    return tuple(
        CacheAction(stage=stage, key=keys[stage], action=action, reason=reason)
        for stage in CACHE_STAGES
    )


def load_cached_core(
    series: SeriesRecord,
    config: FitConfig,
    cache: CacheStore,
    *,
    config_fp: str | None = None,
    series_fp: str | None = None,
) -> CoreSeriesResult | None:
    content_fp = series_fp or series_content_fingerprint(series)
    compute_fp = config_fp or config_computation_fingerprint(config)
    keys = all_stage_keys(
        series_fingerprint=content_fp,
        config=config,
        config_fingerprint=compute_fp,
    )
    lookups = {stage: cache.get(stage, keys[stage]) for stage in CACHE_STAGES}
    if not all(lookup.hit for lookup in lookups.values()):
        return None
    coefficient_payload = lookups["coefficients"].value
    if not isinstance(coefficient_payload, dict):
        return None
    artifact = coefficient_payload.get("artifact")
    if not isinstance(artifact, dict):
        return None
    if artifact.get("series_id") != series.series_id:
        return None
    if artifact.get("content_fingerprint") != content_fp:
        return None
    if artifact.get("cache_keys") != keys:
        return None
    expected = coefficient_payload.get("core_fingerprint")
    if expected != fingerprint(artifact):
        return None
    return CoreSeriesResult(
        artifact=artifact,
        actions=tuple(
            CacheAction(
                stage=stage,
                key=keys[stage],
                action="hit",
                reason="all_stage_keys_valid",
            )
            for stage in CACHE_STAGES
        ),
    )


def fit_series(
    series: SeriesRecord,
    config: FitConfig,
    cache: CacheStore,
    *,
    planned_action: str | None = None,
) -> CoreSeriesResult:
    """Fit one series.  Partial cache hits trigger a deterministic full rebuild."""

    content_fp = series_content_fingerprint(series)
    compute_fp = config_computation_fingerprint(config)
    keys = all_stage_keys(
        series_fingerprint=content_fp,
        config=config,
        config_fingerprint=compute_fp,
    )

    cached = load_cached_core(
        series,
        config,
        cache,
        config_fp=compute_fp,
        series_fp=content_fp,
    )
    if cached is not None:
        if planned_action == "miss":
            # Resume of an uncommitted planned fit must look exactly like the
            # uninterrupted miss ledger while still allowing cache recovery.
            actions = tuple(
                CacheAction(
                    stage=stage,
                    key=keys[stage],
                    action="miss",
                    reason="series_pipeline_recomputed",
                )
                for stage in CACHE_STAGES
            )
            return CoreSeriesResult(cached.artifact, actions)
        return cached

    interpolation = interpolate_series(series)
    basis = build_basis(series.length, config)
    quantized_values = _quantize_values(interpolation.values, config)
    normal = accumulate_normal_equations(
        basis.integer_matrix,
        quantized_values,
        config,
    )
    elimination = solve_fixed_point(normal, config)
    # Keep the normal-equation reference at full configured dimension so a
    # pivot-driven order reduction is attributed to integer_elimination, not
    # to accumulation.
    normal_float, normal_method = solve_integer_normal_in_float(normal)
    names = column_names(config)
    dimension = len(names)

    truth_values = tuple(
        value if value is not None else 0.0 for value in series.values
    )
    truth_fit = fit_unquantized(
        basis.float_matrix,
        truth_values,
        regularization=config.regularization_strength,
        observed_indices=series.observed_indices,
    )
    interpolated_fit = fit_unquantized(
        basis.float_matrix,
        interpolation.values,
        regularization=config.regularization_strength,
    )
    _validate_finite_reference(truth_fit.coefficients, "truth")
    _validate_finite_reference(interpolated_fit.coefficients, "interpolated")
    _validate_finite_reference(normal_float, "normal-equation")

    eliminated_float = pad_vector(elimination.float_back_substitution, dimension)
    final_float = pad_vector(elimination.coefficients, dimension)
    normal_padded = pad_vector(normal_float, dimension)
    truth_padded = pad_vector(truth_fit.coefficients, dimension)
    interpolated_padded = pad_vector(interpolated_fit.coefficients, dimension)
    error_attribution = attribute_stage_errors(
        names=names,
        truth=truth_padded,
        interpolated=interpolated_padded,
        normal_accumulated=normal_padded,
        eliminated=eliminated_float,
        final=final_float,
        relative_tolerance=config.tolerance_relative,
        absolute_tolerance=config.tolerance_absolute,
    )

    full_coefficients = [0] * dimension
    full_coefficients[: elimination.effective_dimension] = elimination.coefficients
    coefficient_entries = _coefficient_entries(
        config,
        basis,
        tuple(full_coefficients),
    )
    trend_slope = full_coefficients[1] if dimension > 1 else 0
    effective_columns = [
        item
        for item in coefficient_entries
        if item["index"] < elimination.effective_dimension
    ]

    core_artifact = {
        "series_id": series.series_id,
        "content_fingerprint": content_fp,
        "config_computation_fingerprint": compute_fp,
        "bit_width_tier": config.bit_tier,
        "observations": series.length,
        "observed_count": len(series.observed_indices),
        "missing_indices": list(series.missing_indices),
        "referenced_observations": referenced_observations(series),
        "interpolation": {
            "artifact_fingerprint": interpolation.artifact_fingerprint,
            "metadata": interpolation.metadata,
            "values": list(interpolation.values),
        },
        "basis": {
            "artifact_fingerprint": basis.artifact_fingerprint,
            "rows": series.length,
            "column_count": len(basis.columns),
            "degenerate_pure_trend": basis.degenerate_pure_trend,
            "configured_harmonic_order": config.harmonic_order,
            "columns": list(basis.columns),
            "integer_matrix_fingerprint": fingerprint([list(row) for row in basis.integer_matrix]),
            "float_matrix_fingerprint": fingerprint([list(row) for row in basis.float_matrix]),
        },
        "quantized_observations": list(quantized_values),
        "normal_equations": {
            "artifact_fingerprint": normal.artifact_fingerprint,
            "dimension": len(normal.rhs),
            "accumulation_order": list(normal.accumulation_order),
            "regularization_integer": normal.regularization_integer,
            "accumulator_max_bits": normal.accumulator_max_bits,
            "common_downshift": normal.downshift,
            "normalized_matrix": [list(row) for row in normal.matrix],
            "normalized_rhs": list(normal.rhs),
            "raw_matrix_fingerprint": fingerprint([list(row) for row in normal.raw_matrix]),
            "raw_rhs_fingerprint": fingerprint(list(normal.raw_rhs)),
            "float_solution_method": normal_method,
        },
        "elimination": {
            "artifact_fingerprint": elimination.artifact_fingerprint,
            "effective_dimension": elimination.effective_dimension,
            "effective_harmonic_order": elimination.effective_harmonic_order,
            "configured_harmonic_order": config.harmonic_order,
            "trend_disabled_by_fallback": elimination.trend_disabled,
            "reduced_to_pure_trend": (
                elimination.effective_harmonic_order == 0
                and not elimination.trend_disabled
            ),
            "reductions": list(elimination.reductions),
            "pivots": list(elimination.pivots),
            "row_operations": list(elimination.row_operations),
            "back_substitution": list(elimination.back_substitution),
            "upper_matrix_fingerprint": fingerprint([list(row) for row in elimination.upper_matrix]),
            "upper_rhs_fingerprint": fingerprint(list(elimination.upper_rhs)),
        },
        "fixed_point": {
            "bits": config.fixed_point_bits,
            "quantization_shift": config.quantization_shift,
            "quantization_scale": config.quantization_scale,
            "rounding": "nearest_half_away_from_zero",
            "coefficient_domain": "integer",
        },
        "coefficient_columns": list(basis.columns),
        "coefficients": coefficient_entries,
        "coefficient_values": [int(item["value"]) for item in coefficient_entries],
        "effective_coefficients": effective_columns,
        "trend_slope": int(trend_slope),
        "trend_slope_sign": _sign(int(trend_slope)),
        "model_degenerate_flags": {
            "configured_harmonic_zero_pure_trend": config.harmonic_order == 0,
            "effective_harmonic_zero_pure_trend": (
                elimination.effective_harmonic_order == 0
                and not elimination.trend_disabled
            ),
            "trend_disabled_intercept_only": elimination.trend_disabled,
        },
        "error_attribution": {
            **error_attribution,
            "truth_fit_method": truth_fit.method,
            "truth_observation_rows": truth_fit.observation_rows,
            "truth_normal_rank": truth_fit.rank,
            "interpolated_fit_method": interpolated_fit.method,
            "normal_float_method": normal_method,
        },
        "cache_keys": keys,
        "cache_stage_order": list(CACHE_STAGES),
    }
    core_fingerprint = fingerprint(core_artifact)

    interpolation_payload = interpolation.as_payload()
    basis_payload = basis.as_payload()
    normal_payload = normal.as_payload()
    coefficient_payload = {
        "artifact": core_artifact,
        "core_fingerprint": core_fingerprint,
    }
    cache.put("interpolation", keys["interpolation"], interpolation_payload)
    cache.put("basis", keys["basis"], basis_payload)
    cache.put("normal_equations", keys["normal_equations"], normal_payload)
    cache.put("coefficients", keys["coefficients"], coefficient_payload)

    return CoreSeriesResult(
        artifact=core_artifact,
        actions=_cache_actions(
            keys,
            hit=False,
            reason="series_pipeline_recomputed",
        ),
    )


def _sign(value: int) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"
