"""Float64 reference fits used only for diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .errors import E_REFERENCE_INVALID, FitError
from .fixedpoint import NormalEquationResult


@dataclass(frozen=True)
class FloatFitResult:
    coefficients: tuple[float, ...]
    method: str
    observation_rows: int
    rank: int


def _regularized_normal_solve(
    matrix: Sequence[Sequence[float]],
    values: Sequence[float],
    regularization: float,
) -> tuple[tuple[float, ...], str, int]:
    design = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(values, dtype=np.float64)
    if design.ndim != 2 or target.ndim != 1 or design.shape[0] != target.shape[0]:
        raise FitError(E_REFERENCE_INVALID, "reference design/target shape mismatch")
    if design.shape[0] == 0 or design.shape[1] == 0:
        raise FitError(E_REFERENCE_INVALID, "reference design is empty")
    gram = design.T @ design
    rhs = design.T @ target
    penalty = np.zeros(design.shape[1], dtype=np.float64)
    if design.shape[1] > 1:
        penalty[1:] = float(regularization)
    gram = gram + np.diag(penalty)
    rank = int(np.linalg.matrix_rank(gram))
    try:
        coefficients = np.linalg.solve(gram, rhs)
        method = "float64_normal_solve"
    except np.linalg.LinAlgError:
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        method = "float64_lstsq_rank_fallback"
    return tuple(float(value) for value in coefficients), method, rank


def fit_unquantized(
    float_matrix: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    regularization: float,
    observed_indices: Sequence[int] | None = None,
) -> FloatFitResult:
    if observed_indices is None:
        selected_rows = list(range(len(values)))
    else:
        selected_rows = list(observed_indices)
    matrix = [float_matrix[index] for index in selected_rows]
    target = [values[index] for index in selected_rows]
    coefficients, method, rank = _regularized_normal_solve(
        matrix, target, regularization
    )
    return FloatFitResult(
        coefficients=coefficients,
        method=method,
        observation_rows=len(selected_rows),
        rank=rank,
    )


def solve_integer_normal_in_float(
    normal: NormalEquationResult,
) -> tuple[tuple[float, ...], str]:
    matrix = np.asarray(normal.matrix, dtype=np.float64)
    values = np.asarray(normal.rhs, dtype=np.float64)
    try:
        coefficients = np.linalg.solve(matrix, values)
        method = "float64_integer_normal_solve"
    except np.linalg.LinAlgError:
        coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=None)
        method = "float64_integer_normal_lstsq_fallback"
    return tuple(float(value) for value in coefficients), method


def float_back_substitute_upper(
    upper: Sequence[Sequence[int]], rhs: Sequence[int]
) -> tuple[float, ...]:
    dimension = len(upper)
    result = [0.0] * dimension
    for row in range(dimension - 1, -1, -1):
        pivot = float(upper[row][row])
        if pivot == 0.0:
            raise FitError(E_REFERENCE_INVALID, "zero upper-triangular pivot")
        value = float(rhs[row])
        for column in range(row + 1, dimension):
            value -= float(upper[row][column]) * result[column]
        result[row] = value / pivot
    return tuple(result)


def pad_vector(values: Sequence[float | int], dimension: int) -> tuple[float, ...]:
    padded = [float(value) for value in values[:dimension]]
    if len(padded) < dimension:
        padded.extend([0.0] * (dimension - len(padded)))
    return tuple(padded)
