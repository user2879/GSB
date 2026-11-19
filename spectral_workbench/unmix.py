"""Non-negative least squares unmixing for a small known endmember set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import UnmixConfig
from .errors import ErrorCode, WorkbenchError
from .io_utils import canonical_json, key_for, sha256_array, sha256_text


@dataclass(frozen=True)
class EndmemberModel:
    endmembers: np.ndarray
    names: Tuple[str, ...]
    order: int
    condition_number: float
    warn_threshold: float
    fatal_threshold: float
    pinv_by_subset: Dict[int, np.ndarray]
    subset_columns: Dict[int, Tuple[int, ...]]
    subset_condition: Dict[int, float]
    row_norms: np.ndarray
    endmember_sha256: str
    config_sha256: str

    @property
    def ill_conditioned(self) -> bool:
        return self.condition_number > self.warn_threshold


@dataclass(frozen=True)
class UnmixResult:
    concentrations: np.ndarray
    residual: np.ndarray
    active_subset: np.ndarray
    ill_conditioned: np.ndarray
    invalid_mask: np.ndarray
    signal_norm: np.ndarray
    condition_number: float
    ill_conditioned_pixels: int
    concentrations_sha256: str
    residual_sha256: str
    active_sha256: str
    ill_conditioned_sha256: str
    record: Dict[str, Any]

    def record_bytes(self) -> bytes:
        return (canonical_json(self.record) + "\n").encode("utf-8")


def build_endmember_model(
    endmembers: np.ndarray,
    names: Sequence[str],
    config: UnmixConfig,
) -> EndmemberModel:
    matrix = np.asarray(endmembers, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "endmember matrix must be non-empty and 2-D")
    if not np.all(np.isfinite(matrix)):
        raise WorkbenchError(ErrorCode.E_UNMIX_COND, "endmember matrix contains non-finite values")
    order = int(config.order) if config.order is not None else int(matrix.shape[1])
    if order < 1 or order > matrix.shape[1]:
        raise WorkbenchError(
            ErrorCode.E_INJECT_CONFIG,
            "unmix order must be within endmember column count",
            {"order": order, "available": int(matrix.shape[1])},
        )
    if order > 8:
        raise WorkbenchError(ErrorCode.E_UNMIX_COND, "enumeration NNLS supports at most 8 fluorophores", {"order": order})
    selected = np.ascontiguousarray(matrix[:, :order], dtype=np.float64)
    norms = np.linalg.norm(selected, axis=0)
    if np.any(norms <= np.finfo(np.float64).tiny):
        raise WorkbenchError(ErrorCode.E_UNMIX_COND, "endmember column has zero norm")
    condition = float(np.linalg.cond(selected, p=2))
    if not np.isfinite(condition) or condition > config.cond_fatal:
        raise WorkbenchError(
            ErrorCode.E_UNMIX_COND,
            "endmember matrix is numerically rank deficient",
            {"condition_number": condition, "fatal_threshold": config.cond_fatal},
        )

    pinv_by_subset: Dict[int, np.ndarray] = {}
    subset_columns: Dict[int, Tuple[int, ...]] = {}
    subset_condition: Dict[int, float] = {}
    for subset_id in range(1, 1 << order):
        columns = tuple(index for index in range(order) if subset_id & (1 << index))
        submatrix = selected[:, columns]
        subset_condition[subset_id] = float(np.linalg.cond(submatrix, p=2))
        pinv_by_subset[subset_id] = np.linalg.pinv(submatrix)
        subset_columns[subset_id] = columns

    pinv_full = np.linalg.pinv(selected)
    row_norms = np.linalg.norm(pinv_full, axis=1).astype(np.float64)
    resolved_names = tuple(str(name) for name in names[:order])
    if len(resolved_names) != order:
        resolved_names = tuple(f"fluorophore_{index}" for index in range(order))
    return EndmemberModel(
        endmembers=selected,
        names=resolved_names,
        order=order,
        condition_number=condition,
        warn_threshold=float(config.cond_warn),
        fatal_threshold=float(config.cond_fatal),
        pinv_by_subset=pinv_by_subset,
        subset_columns=subset_columns,
        subset_condition=subset_condition,
        row_norms=row_norms,
        endmember_sha256=sha256_array(np.ascontiguousarray(selected, dtype="<f8")),
        config_sha256=sha256_text(canonical_json(config.hash_payload())),
    )


def unmix_key(
    model: EndmemberModel,
    corrected_sha256: str,
    quantization_upper_bound: float,
    config: UnmixConfig,
) -> str:
    return key_for(
        {
            "stage": "unmix-nnls-v1",
            "endmembers": model.endmember_sha256,
            "corrected": corrected_sha256,
            "quantization_upper_bound": float(quantization_upper_bound),
            "config": config.hash_payload(),
        }
    )


def unmix_frame(
    corrected: np.ndarray,
    invalid_mask: np.ndarray,
    model: EndmemberModel,
    config: UnmixConfig,
    *,
    quantization_upper_bound: float,
    frame_index: int = 0,
) -> UnmixResult:
    cube = np.asarray(corrected, dtype=np.float64)
    invalid = np.asarray(invalid_mask, dtype=np.uint8)
    if cube.ndim != 3 or cube.shape[0] != model.endmembers.shape[0]:
        raise WorkbenchError(
            ErrorCode.E_CUBE_FORMAT,
            "calibrated frame band count does not match endmember matrix",
            {"frame": frame_index, "cube_shape": list(cube.shape), "endmembers": list(model.endmembers.shape)},
        )
    if invalid.shape != cube.shape[1:]:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "invalid mask shape mismatch", {"frame": frame_index})

    bands, height, width = cube.shape
    pixels = height * width
    y_all = cube.reshape(bands, pixels)
    valid_flat = invalid.reshape(pixels) == 0
    y = np.where(valid_flat[None, :], y_all, 0.0)
    signal_norm = np.linalg.norm(y, axis=0)
    order = model.order
    subset_count = 1 << order

    best_x = np.zeros((order, pixels), dtype=np.float64)
    best_res2 = np.sum(y * y, axis=0)
    best_subset = np.zeros(pixels, dtype=np.uint8)
    best_cond = np.ones(pixels, dtype=np.float64)
    for subset_id in range(1, subset_count):
        columns = model.subset_columns[subset_id]
        submatrix = model.endmembers[:, columns]
        pinv = model.pinv_by_subset[subset_id]
        candidate = pinv @ y
        nonnegative = np.all(candidate >= 0.0, axis=0)
        residual = submatrix @ candidate - y
        residual2 = np.sum(residual * residual, axis=0)
        better = nonnegative & (residual2 < best_res2)
        if not np.any(better):
            continue
        best_res2[better] = residual2[better]
        best_x[:, better] = 0.0
        for local_index, column in enumerate(columns):
            best_x[column, better] = candidate[local_index, better]
        best_subset[better] = subset_id
        best_cond[better] = model.subset_condition[subset_id]

    residual_map = np.sqrt(np.maximum(best_res2, 0.0)).reshape(height, width)
    concentrations = best_x.reshape(order, height, width).astype(np.float32)
    active = best_subset.reshape(height, width)
    ill = ((best_cond > model.warn_threshold) & valid_flat & (best_subset > 0)).reshape(height, width).astype(np.uint8)
    concentrations[:, invalid.astype(bool)] = 0.0
    residual_map[invalid.astype(bool)] = 0.0
    active[invalid.astype(bool)] = 0
    ill[invalid.astype(bool)] = 0

    concentration_bytes = np.ascontiguousarray(concentrations, dtype="<f4")
    residual_bytes = np.ascontiguousarray(residual_map, dtype="<f4")
    active_bytes = np.ascontiguousarray(active, dtype="u1")
    ill_bytes = np.ascontiguousarray(ill, dtype="u1")
    record: Dict[str, Any] = {
        "formula_version": "nnls-enumeration-v1",
        "frame": frame_index,
        "shape": [order, height, width],
        "order": order,
        "condition_number": model.condition_number,
        "condition_warn_threshold": model.warn_threshold,
        "condition_fatal_threshold": model.fatal_threshold,
        "matrix_ill_conditioned": bool(model.ill_conditioned),
        "ill_conditioned_pixels": int(np.count_nonzero(ill)),
        "invalid_pixels": int(np.count_nonzero(invalid)),
        "quantization_upper_bound_dn": float(quantization_upper_bound),
        "residual_mean_dn": float(np.mean(residual_map[valid_flat.reshape(height, width)])),
        "residual_p95_dn": float(np.percentile(residual_map[valid_flat.reshape(height, width)], 95)),
        "concentrations_sha256": sha256_array(concentration_bytes),
        "residual_sha256": sha256_array(residual_bytes),
        "active_subset_sha256": sha256_array(active_bytes),
        "ill_conditioned_sha256": sha256_array(ill_bytes),
        "signal_norm_sha256": sha256_array(np.ascontiguousarray(signal_norm.reshape(height, width), dtype="<f4")),
        "endmember_sha256": model.endmember_sha256,
        "config_sha256": model.config_sha256,
    }
    return UnmixResult(
        concentrations=concentrations,
        residual=residual_map.astype(np.float32),
        active_subset=active,
        ill_conditioned=ill,
        invalid_mask=invalid.copy(),
        signal_norm=signal_norm.astype(np.float64).reshape(height, width),
        condition_number=model.condition_number,
        ill_conditioned_pixels=int(np.count_nonzero(ill)),
        concentrations_sha256=sha256_array(concentration_bytes),
        residual_sha256=sha256_array(residual_bytes),
        active_sha256=sha256_array(active_bytes),
        ill_conditioned_sha256=sha256_array(ill_bytes),
        record=record,
    )


def concentration_error_components(
    model: EndmemberModel,
    result: UnmixResult,
    *,
    frame_index: int,
    y: int,
    x: int,
) -> Dict[str, Any]:
    """Recompute the per-fluorophore interval and its four components."""
    if not (0 <= y < result.active_subset.shape[0] and 0 <= x < result.active_subset.shape[1]):
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "pixel coordinate outside frame", {"frame": frame_index, "y": y, "x": x})
    subset_id = int(result.active_subset[y, x])
    q = float(result.record["quantization_upper_bound_dn"])
    residual = float(result.residual[y, x])
    y_norm = max(float(result.signal_norm[y, x]), 1.0e-12)
    active_condition = model.subset_condition.get(subset_id, 1.0)
    eps = float(np.finfo(np.float32).eps)
    band_count = model.endmembers.shape[0]
    items: List[Dict[str, Any]] = []
    for fluorophore in range(model.order):
        concentration = float(result.concentrations[fluorophore, y, x])
        row_norm = float(model.row_norms[fluorophore])
        calibration = row_norm * q
        residual_component = row_norm * residual * (float(np.sqrt(band_count)) * 2.0)
        conditioning = active_condition * (q / y_norm) * abs(concentration)
        roundoff = 4.0 * eps * (abs(concentration) + 1.0)
        total = calibration + residual_component + conditioning + roundoff
        items.append(
            {
                "fluorophore": fluorophore,
                "name": model.names[fluorophore],
                "estimate": concentration,
                "lower": concentration - total,
                "upper": concentration + total,
                "error_bound": total,
                "components": {
                    "calibration_quantization": calibration,
                    "unmix_residual": residual_component,
                    "condition_amplification": conditioning,
                    "float_roundoff": roundoff,
                },
            }
        )
    return {
        "frame": frame_index,
        "pixel": [y, x],
        "active_subset": list(model.subset_columns.get(subset_id, ())),
        "active_condition_number": active_condition,
        "global_condition_number": model.condition_number,
        "ill_conditioned": bool(result.ill_conditioned[y, x]),
        "quantization_upper_bound_dn": q,
        "residual_dn": residual,
        "signal_norm_dn": y_norm,
        "fluorophores": items,
    }
