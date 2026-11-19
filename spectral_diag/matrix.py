"""White-point adaptation matrix inversion and conditioning diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .canonical import sha256_obj, stable_float
from .errors import DiagnosticError, ErrorCode

COMPONENT_NAMES = ("X", "Y", "Z")


@dataclass(frozen=True)
class ConditionDiagnostics:
    rank: int
    singular_values: np.ndarray
    condition_number: float | None
    row_leverage: np.ndarray
    collinear_pairs: tuple[tuple[int, int], ...]
    leave_one_out_condition: tuple[float | None, ...]
    offending_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "singular_values": self.singular_values.tolist(),
            "condition_number": None if self.condition_number is None else stable_float(self.condition_number),
            "condition_number_text": "inf" if self.condition_number is None else None,
            "row_leverage": self.row_leverage.tolist(),
            "collinear_pairs": [list(pair) for pair in self.collinear_pairs],
            "leave_one_out_condition": [
                None if value is None else stable_float(value) for value in self.leave_one_out_condition
            ],
            "offending_indices": list(self.offending_indices),
        }


@dataclass
class PruneStep:
    removed_index: int
    condition_before: float | None
    condition_after: float | None
    residual_rmse_before: float
    residual_rmse_after: float
    remaining_indices: tuple[int, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "removed_index": self.removed_index,
            "condition_before": None if self.condition_before is None else stable_float(self.condition_before),
            "condition_after": None if self.condition_after is None else stable_float(self.condition_after),
            "residual_rmse_before": stable_float(self.residual_rmse_before),
            "residual_rmse_after": stable_float(self.residual_rmse_after),
            "remaining_indices": list(self.remaining_indices),
            "reason": self.reason,
        }


@dataclass
class MatrixSolution:
    status: str
    matrix: np.ndarray | None
    used_indices: tuple[int, ...]
    condition_number: float | None
    matrix_condition_number: float | None
    residuals: np.ndarray
    residual_componentwise: np.ndarray
    residual_norm: np.ndarray
    residual_rmse: float
    residual_max_abs: float
    diagnostics: ConditionDiagnostics
    prune_trace: tuple[PruneStep, ...] = ()
    error_code: str | None = None
    message: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status in {"accepted", "accepted_after_pruning"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "accepted": self.accepted,
            "matrix": None if self.matrix is None else self.matrix.tolist(),
            "used_indices": list(self.used_indices),
            "condition_number": None if self.condition_number is None else stable_float(self.condition_number),
            "condition_number_text": "inf" if self.condition_number is None else None,
            "matrix_condition_number": None
            if self.matrix_condition_number is None
            else stable_float(self.matrix_condition_number),
            "residuals": self.residuals.tolist(),
            "residual_componentwise": self.residual_componentwise.tolist(),
            "residual_norm": self.residual_norm.tolist(),
            "residual_rmse": stable_float(self.residual_rmse),
            "residual_max_abs": stable_float(self.residual_max_abs),
            "diagnostics": self.diagnostics.to_dict(),
            "prune_trace": [step.to_dict() for step in self.prune_trace],
            "error_code": self.error_code,
            "message": self.message,
            "extra": self.extra,
            "solution_sha256": sha256_obj(
                {
                    "matrix": None if self.matrix is None else self.matrix.tolist(),
                    "residuals": self.residuals.tolist(),
                    "used_indices": list(self.used_indices),
                    "status": self.status,
                }
            ),
        }


def _validate_pairs(source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_xyz, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    if source.ndim != 2 or source.shape[1] != 3 or target.ndim != 2 or target.shape[1] != 3:
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="source and target XYZ must be N x 3 arrays",
            details={"source_shape": list(source.shape), "target_shape": list(target.shape)},
        )
    if source.shape != target.shape:
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="source and target XYZ arrays must have the same shape",
            details={"source_shape": list(source.shape), "target_shape": list(target.shape)},
        )
    if source.shape[0] < 3:
        raise DiagnosticError(
            code=ErrorCode.MATRIX_SINGULAR,
            message="fewer than three colour samples cannot determine a 3x3 matrix",
            details={"sample_count": int(source.shape[0]), "sample_indices": list(range(source.shape[0]))},
        )
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="sample XYZ contains non-finite values",
            details={"source_finite": bool(np.all(np.isfinite(source))), "target_finite": bool(np.all(np.isfinite(target)))},
        )
    return source, target


def _condition_number(singular_values: np.ndarray, tolerance: float) -> float | None:
    if singular_values.size < 3 or singular_values[-1] <= tolerance:
        return None
    return float(singular_values[0] / singular_values[-1])


def _rank_and_singular_values(values: np.ndarray) -> tuple[int, np.ndarray, float]:
    singular_values = np.linalg.svd(values, compute_uv=False)
    tolerance = np.finfo(float).eps * max(values.shape) * (singular_values[0] if singular_values.size else 0.0)
    rank = int(np.sum(singular_values > tolerance))
    return rank, singular_values, float(tolerance)


def _pairwise_collinearity(values: np.ndarray) -> tuple[tuple[int, int], ...]:
    norms = np.linalg.norm(values, axis=1)
    pairs: list[tuple[int, int]] = []
    for left in range(values.shape[0]):
        if norms[left] == 0:
            continue
        for right in range(left + 1, values.shape[0]):
            if norms[right] == 0:
                continue
            cosine = abs(float(np.dot(values[left], values[right]) / (norms[left] * norms[right])))
            if cosine >= 1.0 - 1e-10:
                pairs.append((left, right))
    return tuple(pairs)


def condition_diagnostics(source_xyz: np.ndarray) -> ConditionDiagnostics:
    source = np.asarray(source_xyz, dtype=float)
    rank, singular_values, tolerance = _rank_and_singular_values(source)
    condition = _condition_number(singular_values, tolerance)
    _, _, vt = np.linalg.svd(source, full_matrices=False)
    projected = source @ vt.T
    denominator = float(np.sum(projected**2))
    leverage = (
        np.sum(projected**2, axis=1) / denominator
        if denominator > 0
        else np.zeros(source.shape[0], dtype=float)
    )
    collinear = _pairwise_collinearity(source)

    leave_one_out: list[float | None] = []
    for index in range(source.shape[0]):
        if source.shape[0] - 1 < 3:
            leave_one_out.append(None)
            continue
        candidate = np.delete(source, index, axis=0)
        candidate_rank, candidate_singular, candidate_tolerance = _rank_and_singular_values(candidate)
        leave_one_out.append(
            _condition_number(candidate_singular, candidate_tolerance)
            if candidate_rank == 3
            else None
        )

    offending: set[int] = set()
    for left, right in collinear:
        offending.update((left, right))
    if not offending:
        finite_pairs = [
            (value, index) for index, value in enumerate(leave_one_out) if value is not None
        ]
        if finite_pairs:
            offending.add(min(finite_pairs, key=lambda pair: (pair[0], pair[1]))[1])
        else:
            offending.update(range(min(source.shape[0], 3)))
    return ConditionDiagnostics(
        rank=rank,
        singular_values=singular_values,
        condition_number=condition,
        row_leverage=leverage,
        collinear_pairs=collinear,
        leave_one_out_condition=tuple(leave_one_out),
        offending_indices=tuple(sorted(offending)),
    )


def _fit(
    source: np.ndarray,
    target: np.ndarray,
    used_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fit_source = source[list(used_indices)]
    fit_target = target[list(used_indices)]
    matrix, _, _, _ = np.linalg.lstsq(fit_source, fit_target, rcond=None)
    predictions = source @ matrix
    componentwise = predictions - target
    norms = np.linalg.norm(componentwise, axis=1)
    return np.asarray(matrix, dtype=float), componentwise, norms, predictions


def _residual_rmse(componentwise: np.ndarray) -> float:
    return float(np.sqrt(np.mean(componentwise**2)))


def _rejection(
    status: str,
    source: np.ndarray,
    target: np.ndarray,
    used_indices: tuple[int, ...],
    diagnostics: ConditionDiagnostics,
    trace: tuple[PruneStep, ...],
    message: str,
) -> MatrixSolution:
    matrix, componentwise, norms, _ = _fit(source, target, used_indices)
    matrix_condition = (
        float(np.linalg.cond(matrix)) if matrix is not None and np.all(np.isfinite(matrix)) else None
    )
    return MatrixSolution(
        status=status,
        matrix=None,
        used_indices=used_indices,
        condition_number=None,
        matrix_condition_number=matrix_condition,
        residuals=norms,
        residual_componentwise=componentwise,
        residual_norm=norms,
        residual_rmse=_residual_rmse(componentwise),
        residual_max_abs=float(np.max(np.abs(componentwise))) if componentwise.size else 0.0,
        diagnostics=diagnostics,
        prune_trace=trace,
        error_code=str(ErrorCode.MATRIX_SINGULAR),
        message=message,
        extra={"diagnostic_matrix_not_committed": matrix.tolist()},
    )


def solve_adaptation(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    cond_threshold: float = 10000.0,
    allow_prune: bool = True,
) -> MatrixSolution:
    """Solve ``source @ M = target`` with deterministic ill-conditioning repair.

    A candidate is acceptable only when the design matrix has rank three and
    its 2-norm condition number is at or below ``cond_threshold``.  When a
    sample subset is too ill-conditioned, the sample whose removal minimises
    the resulting condition number (lowest index on ties) is removed.  Every
    removal is retained in the trace.  A failure never substitutes identity.
    """

    source, target = _validate_pairs(source_xyz, target_xyz)
    used = tuple(range(source.shape[0]))
    trace: list[PruneStep] = []
    initial_diagnostics = condition_diagnostics(source)

    while True:
        current_source = source[list(used)]
        diagnostics = condition_diagnostics(current_source)
        current_condition = diagnostics.condition_number
        acceptable = diagnostics.rank == 3 and current_condition is not None and current_condition <= cond_threshold
        if acceptable:
            matrix, componentwise, norms, _ = _fit(source, target, used)
            matrix_condition = float(np.linalg.cond(matrix))
            status = "accepted_after_pruning" if trace else "accepted"
            return MatrixSolution(
                status=status,
                matrix=matrix,
                used_indices=used,
                condition_number=current_condition,
                matrix_condition_number=matrix_condition,
                residuals=norms,
                residual_componentwise=componentwise,
                residual_norm=norms,
                residual_rmse=_residual_rmse(componentwise),
                residual_max_abs=float(np.max(np.abs(componentwise))),
                diagnostics=initial_diagnostics if not trace else diagnostics,
                prune_trace=tuple(trace),
            )

        if not allow_prune or len(used) - 1 < 3:
            reason = "rank_deficient" if diagnostics.rank < 3 else "condition_threshold_exceeded"
            return _rejection(
                status="rejected",
                source=source,
                target=target,
                used_indices=used,
                diagnostics=diagnostics,
                trace=tuple(trace),
                message=(
                    "colour sample set is rank deficient or no admissible subset remains "
                    f"({reason})"
                ),
            )

        before_rmse = _residual_rmse(_fit(source, target, used)[1])
        candidates: list[tuple[int, int, float | None, float]] = []
        for index in used:
            candidate_used = tuple(value for value in used if value != index)
            candidate_source = source[list(candidate_used)]
            candidate_rank, candidate_singular, candidate_tolerance = _rank_and_singular_values(candidate_source)
            candidate_condition = (
                _condition_number(candidate_singular, candidate_tolerance)
                if candidate_rank == 3
                else None
            )
            candidate_rmse = _residual_rmse(_fit(source, target, candidate_used)[1])
            candidates.append((3 - candidate_rank, index, candidate_condition, candidate_rmse))

        # Highest rank first, then lowest condition (None ranks last), then the
        # lowest sample index.  The ordering is the reproducibility contract.
        _rank_penalty, removed_index, after_condition, after_rmse = min(
            candidates,
            key=lambda item: (
                item[0],
                float("inf") if item[2] is None else item[2],
                item[1],
            ),
        )
        trace.append(
            PruneStep(
                removed_index=removed_index,
                condition_before=current_condition,
                condition_after=after_condition,
                residual_rmse_before=before_rmse,
                residual_rmse_after=after_rmse,
                remaining_indices=tuple(value for value in used if value != removed_index),
                reason="max_condition_reduction",
            )
        )
        used = tuple(value for value in used if value != removed_index)


def quantize_xyz(
    xyz: np.ndarray,
    *,
    quant_range: tuple[float, float],
    quant_bits: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(xyz, dtype=float)
    low, high = quant_range
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise DiagnosticError(
            code=ErrorCode.QUANT_RANGE,
            message="quantisation range is invalid",
            details={"quant_range": [low, high], "sample_index": 0, "component": 0},
        )
    if not 1 <= quant_bits <= 24:
        raise DiagnosticError(
            code=ErrorCode.QUANT_RANGE,
            message="quantisation bit depth is invalid",
            details={"quant_bits": quant_bits, "sample_index": 0, "component": 0},
        )
    outside = np.argwhere((values < low) | (values > high))
    if outside.size:
        flat_index = int(np.argwhere((values < low) | (values > high))[0, 0])
        component = int(np.argwhere((values < low) | (values > high))[0, 1])
        raise DiagnosticError(
            code=ErrorCode.QUANT_RANGE,
            message="XYZ observation lies outside the declared quantisation range",
            details={
                "sample_index": flat_index,
                "component": component,
                "component_name": COMPONENT_NAMES[component],
                "value": float(values[flat_index, component]),
                "quant_range": [low, high],
            },
        )
    step = (high - low) / ((1 << quant_bits) - 1)
    quantized = np.round((values - low) / step) * step + low
    quantized = np.clip(quantized, low, high)
    error = quantized - values
    return quantized, {
        "quant_bits": quant_bits,
        "quant_range": [low, high],
        "quant_step": step,
        "max_abs_error": float(np.max(np.abs(error))) if error.size else 0.0,
        "rmse": float(np.sqrt(np.mean(error**2))) if error.size else 0.0,
    }


def matrix_offset_ledger(
    baseline: MatrixSolution,
    perturbed: MatrixSolution,
) -> dict[str, Any]:
    if baseline.matrix is None or perturbed.matrix is None:
        return {
            "available": False,
            "reason": "both solutions must be accepted to compare matrices",
        }
    delta = perturbed.matrix - baseline.matrix
    return {
        "available": True,
        "element_offsets": delta.tolist(),
        "max_abs_offset": float(np.max(np.abs(delta))),
        "frobenius_offset": float(np.linalg.norm(delta)),
        "condition_before": baseline.condition_number,
        "condition_after": perturbed.condition_number,
        "condition_delta": (
            None
            if baseline.condition_number is None or perturbed.condition_number is None
            else float(perturbed.condition_number - baseline.condition_number)
        ),
        "residual_rmse_before": baseline.residual_rmse,
        "residual_rmse_after": perturbed.residual_rmse,
        "residual_rmse_delta": float(perturbed.residual_rmse - baseline.residual_rmse),
    }
