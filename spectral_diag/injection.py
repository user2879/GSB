"""Reproducible parameter-level anomaly injection and first-impact location."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np

from .canonical import sha256_obj, stable_float
from .cmf import get_observer
from .config import RunConfig
from .errors import DiagnosticError, ErrorCode
from .matrix import COMPONENT_NAMES
from .scene import GeneratedScene, generate_scene

ALLOWED_KINDS = (
    "collinear_sample",
    "zero_white_point",
    "quant_out_of_range",
    "observer_invalid",
    "low_integration_density",
)
MAX_REFERENCE_DENSITY_NM = 1.0


def normalize_injections(injections: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for position, injection in enumerate(injections):
        if not isinstance(injection, dict):
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="each injection must be a JSON object",
                details={"position": position},
            )
        kind = injection.get("kind")
        if kind not in ALLOWED_KINDS:
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="unknown injection kind",
                details={"kind": kind, "allowed": list(ALLOWED_KINDS), "position": position},
            )
        params = injection.get("params", {})
        if not isinstance(params, dict):
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="injection params must be a JSON object",
                details={"position": position, "kind": kind},
            )
        normalized.append({"kind": kind, "params": deepcopy(params)})
    return tuple(normalized)


def _check_config_level(config: RunConfig) -> None:
    get_observer(config.observer)
    if config.reference_density_nm > MAX_REFERENCE_DENSITY_NM:
        raise DiagnosticError(
            code=ErrorCode.INTEGRATION_DENSITY,
            message="reference integration sampling density is too low (wavelength spacing too coarse)",
            details={
                "density_nm": config.reference_density_nm,
                "maximum_spacing_nm": MAX_REFERENCE_DENSITY_NM,
                "sample_index": 0,
                "component": 0,
                "component_name": "X",
                "observation_kind": "integration_density",
            },
        )


def _first_out_of_range(
    source: np.ndarray,
    target: np.ndarray,
    quant_range: tuple[float, float],
) -> tuple[int, int, str, float] | None:
    low, high = quant_range
    for index in range(source.shape[0]):
        for component in range(3):
            for group, values in (("source", source), ("target", target)):
                value = float(values[index, component])
                if value < low or value > high:
                    return index, component, group, value
    return None


def _set_sample_xyz(scene: GeneratedScene, index: int, group: str, values: list[float]) -> None:
    scene.payload["sample_pairs"][index][group]["xyz"] = values


def _get_sample_xyz(scene: GeneratedScene, index: int, group: str) -> np.ndarray:
    return np.asarray(scene.payload["sample_pairs"][index][group]["xyz"], dtype=float)


def _raise_with_scene(error: DiagnosticError, scene: GeneratedScene) -> None:
    error.details["_scene_source_xyz"] = scene.source_xyz.tolist()
    error.details["_scene_target_xyz"] = scene.target_xyz.tolist()
    raise error


def _apply_scene_injection(
    scene: GeneratedScene,
    config: RunConfig,
    injection: dict[str, Any],
) -> RunConfig:
    kind = injection["kind"]
    params = injection["params"]

    if kind == "zero_white_point":
        group = str(params.get("group", "source"))
        if group not in {"source", "camera"}:
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="zero_white_point group must be source or camera",
                details={"group": group},
            )
        component = int(params.get("component", 1))
        if component not in (0, 1, 2):
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="zero_white_point component must be 0, 1, or 2",
                details={"component": component},
            )
        white = list(scene.payload[group]["white_xyz"])
        white[component] = 0.0
        scene.payload[group]["white_xyz"] = white
        scene.payload[group]["integration"]["white_xyz"] = white
        error = DiagnosticError(
            code=ErrorCode.WHITE_POINT,
            message="injected zero component in illuminant white point",
            details={
                "group": group,
                "white_xyz": white,
                "component": component,
                "component_name": COMPONENT_NAMES[component],
                "sample_index": 0,
                "observation_kind": "white_point",
                "injection": injection,
            },
        )
        _raise_with_scene(error, scene)

    if kind == "quant_out_of_range":
        index = int(params.get("sample_index", 0))
        component = int(params.get("component", 0))
        group = str(params.get("group", "target"))
        if not 0 <= index < config.sample_count or component not in (0, 1, 2):
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="quant_out_of_range location is invalid",
                details={"sample_index": index, "component": component},
            )
        low, high = config.quant_range
        direction = params.get("direction", "high")
        value = float(high + max(1.0, (high - low) * 0.05)) if direction != "low" else float(
            low - max(1.0, (high - low) * 0.05)
        )
        values = _get_sample_xyz(scene, index, group).tolist()
        values[component] = value
        _set_sample_xyz(scene, index, group, values)
        error = DiagnosticError(
            code=ErrorCode.QUANT_RANGE,
            message="injected XYZ observation lies outside the declared quantisation range",
            details={
                "sample_index": index,
                "component": component,
                "component_name": COMPONENT_NAMES[component],
                "group": group,
                "value": value,
                "quant_range": [low, high],
                "observation_kind": "xyz",
                "injection": injection,
            },
        )
        _raise_with_scene(error, scene)

    if kind == "collinear_sample":
        base_source = _get_sample_xyz(scene, 0, "source")
        base_target = _get_sample_xyz(scene, 0, "target")
        indices = params.get("indices")
        if indices is None:
            indices = list(range(config.sample_count))
        indices = [int(index) for index in indices]
        if not indices or min(indices) < 0 or max(indices) >= config.sample_count:
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="collinear_sample indices are invalid",
                details={"indices": indices, "sample_count": config.sample_count},
            )
        for position, index in enumerate(sorted(set(indices))):
            scale = 0.45 + 0.07 * position
            _set_sample_xyz(scene, index, "source", (base_source * scale).tolist())
            _set_sample_xyz(scene, index, "target", (base_target * scale).tolist())
        return config

    return config


def prepare_scene(config: RunConfig) -> GeneratedScene:
    """Validate config-level injections, generate, then apply scene injections."""

    injections = normalize_injections(config.injections)
    effective = config
    scene_injections: list[dict[str, Any]] = []
    for injection in injections:
        kind = injection["kind"]
        params = injection["params"]
        if kind == "observer_invalid":
            marker = str(params.get("observer", "cie1955_2"))
            effective = replace(effective, observer=marker)
            _check_config_level(effective)
        elif kind == "low_integration_density":
            density = float(params.get("density_nm", 20.0))
            effective = replace(effective, reference_density_nm=density)
            _check_config_level(effective)
        else:
            scene_injections.append(injection)

    _check_config_level(effective)
    scene = generate_scene(effective)
    scene.payload["injections"] = list(injections)
    scene.payload["injection_sha256"] = sha256_obj(list(injections))

    source = scene.source_xyz.copy()
    target = scene.target_xyz.copy()
    out_of_range = _first_out_of_range(source, target, effective.quant_range)
    if out_of_range is not None:
        index, component, group, value = out_of_range
        error = DiagnosticError(
            code=ErrorCode.QUANT_RANGE,
            message="generated XYZ lies outside the declared quantisation range",
            details={
                "sample_index": index,
                "component": component,
                "component_name": COMPONENT_NAMES[component],
                "group": group,
                "value": value,
                "quant_range": list(effective.quant_range),
                "observation_kind": "xyz",
            },
        )
        _raise_with_scene(error, scene)

    for injection in scene_injections:
        effective = _apply_scene_injection(scene, effective, injection)
    return scene


def _diagnostic_matrix(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    source = np.asarray(source_xyz, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    try:
        matrix, _, rank, _ = np.linalg.lstsq(source, target, rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    if matrix.shape != (3, 3) or rank < 1:
        return None, None
    residual = source @ matrix - target
    return np.asarray(matrix, dtype=float), np.asarray(residual, dtype=float)


def locate_first_affected(
    error: DiagnosticError,
    *,
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    matrix: np.ndarray | None = None,
    quant_step: float | None = None,
) -> dict[str, Any]:
    """Locate the first affected observation by sample index, then XYZ order."""

    details = dict(error.details)
    source = np.asarray(source_xyz, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    index = int(details.get("sample_index", 0))
    component = int(details.get("component", 0))
    index = max(0, min(index, source.shape[0] - 1))
    component = max(0, min(component, 2))
    group = str(details.get("group", "target"))
    if group not in {"source", "target"}:
        group = "target"

    _, residual = _diagnostic_matrix(source, target)
    if residual is None:
        residual = np.zeros_like(target)
    if matrix is not None:
        residual = source @ np.asarray(matrix, dtype=float) - target

    observed_vector = source[index] if group == "source" else target[index]
    observed_value = float(observed_vector[component])
    predicted_vector = observed_vector - residual[index]
    predicted_value = float(predicted_vector[component])
    residual_component = float(residual[index, component])
    row_norm = float(np.linalg.norm(residual[index]))
    margin = max(row_norm, 0.0 if quant_step is None else quant_step / 2.0, 1e-12)
    unbounded = error.code == ErrorCode.MATRIX_SINGULAR
    interval = (
        [None, None]
        if unbounded
        else [stable_float(predicted_value - margin), stable_float(predicted_value + margin)]
    )

    return {
        "error_code": str(error.code),
        "message": error.message,
        "first_affected": {
            "sample_index": index,
            "component": component,
            "component_name": COMPONENT_NAMES[component],
            "group": group,
            "observation_kind": details.get("observation_kind", "xyz"),
            "observed_value": stable_float(observed_value),
            "predicted_value": stable_float(predicted_value),
            "residual": stable_float(residual_component),
            "residual_row_l2": stable_float(row_norm),
            "error_interval": interval,
            "error_interval_unbounded": unbounded,
            "error_interval_definition": (
                "unbounded: no admissible matrix was committed"
                if unbounded
                else "predicted +/- max(row_L2_residual, quant_step/2)"
            ),
        },
        "scan_order": "sample_index_ascending_then_component_X_Y_Z",
        "detail_location": {
            key: details[key]
            for key in ("sample_index", "component", "component_name", "group", "value")
            if key in details
        },
        "diagnostic_matrix_committed": False,
    }


def matrix_singular_error(
    solution: Any,
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> DiagnosticError:
    diagnostics = solution.diagnostics.to_dict()
    offending = diagnostics.get("offending_indices") or [0]
    first = int(offending[0])
    return DiagnosticError(
        code=ErrorCode.MATRIX_SINGULAR,
        message=solution.message or "colour sample design is singular or ill-conditioned",
        details={
            "sample_index": first,
            "component": 0,
            "component_name": "X",
            "group": "source",
            "observation_kind": "sample_set",
            "offending_indices": offending,
            "condition_number": diagnostics.get("condition_number"),
            "condition_number_text": diagnostics.get("condition_number_text"),
            "prune_trace": [step.to_dict() for step in solution.prune_trace],
            "solution": solution.to_dict(),
        },
    )
