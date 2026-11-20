from __future__ import annotations

from typing import Any

import numpy as np

from .errors import E_NONFINITE_INPUT, E_SHAPE_MISMATCH, BusinessError


def _shape_of(value: Any) -> list[int]:
    try:
        return [int(item) for item in np.shape(value)]
    except Exception:
        return []


def _as_float32(value: Any, name: str, stage: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float32, order="C")
    except (TypeError, ValueError) as exc:
        raise BusinessError(
            code=E_NONFINITE_INPUT,
            stage=stage,
            message=f"{name} cannot be represented as float32.",
            shape=_shape_of(value),
            details={"matrix": name, "error": str(exc)},
        ) from exc


def _validate_scalar(value: Any, name: str, stage: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise BusinessError(
            code=E_NONFINITE_INPUT,
            stage=stage,
            message=f"{name} is not a finite scalar.",
            details={"matrix": name},
        ) from exc
    if not np.isfinite(converted):
        raise BusinessError(
            code=E_NONFINITE_INPUT,
            stage=stage,
            message=f"{name} is not a finite scalar.",
            details={"matrix": name, "value": converted},
        )
    return converted


def _validate_finite(array: np.ndarray, name: str, stage: str) -> None:
    mask = ~np.isfinite(array)
    if not bool(np.any(mask)):
        return
    index = [int(item) for item in np.argwhere(mask)[0].tolist()]
    batch_index = index[0] if array.ndim == 3 and index else None
    raise BusinessError(
        code=E_NONFINITE_INPUT,
        stage=stage,
        message=f"{name} contains a non-finite element.",
        shape=[int(item) for item in array.shape],
        batch_index=batch_index,
        details={"matrix": name, "index": index, "value": float(array[tuple(index)])},
    )


def prepare_gemm_arrays(
    A: Any,
    B: Any,
    C: Any,
    alpha: Any = 1.0,
    beta: Any = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    a_shape = _shape_of(A)
    b_shape = _shape_of(B)
    c_shape = _shape_of(C)
    if len(a_shape) != 2 or len(b_shape) != 2 or len(c_shape) != 2:
        raise BusinessError(
            code=E_SHAPE_MISMATCH,
            stage="gemm_shape_validation",
            message="GEMM expects A, B, and C to be two-dimensional matrices.",
            shape={"A": a_shape, "B": b_shape, "C": c_shape},
        )
    if a_shape[1] != b_shape[0]:
        raise BusinessError(
            code=E_SHAPE_MISMATCH,
            stage="gemm_shape_validation",
            message="GEMM inner dimensions are incompatible.",
            shape={"A": a_shape, "B": b_shape, "C": c_shape},
            details={"required": "A.shape[1] == B.shape[0]"},
        )
    expected_c = [a_shape[0], b_shape[1]]
    if c_shape != expected_c:
        raise BusinessError(
            code=E_SHAPE_MISMATCH,
            stage="gemm_shape_validation",
            message="GEMM output shape is incompatible with A and B.",
            shape={"A": a_shape, "B": b_shape, "C": c_shape},
            details={"expected_C": expected_c},
        )
    A32 = _as_float32(A, "A", "input_validation")
    B32 = _as_float32(B, "B", "input_validation")
    C32 = _as_float32(C, "C", "input_validation")
    for name, array in (("A", A32), ("B", B32), ("C", C32)):
        _validate_finite(array, name, "input_validation")
    alpha_value = _validate_scalar(alpha, "alpha", "input_validation")
    beta_value = _validate_scalar(beta, "beta", "input_validation")
    return A32, B32, C32, alpha_value, beta_value


def prepare_bmm_arrays(
    A: Any,
    B: Any,
    C: Any,
    alpha: Any = 1.0,
    beta: Any = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    a_shape = _shape_of(A)
    b_shape = _shape_of(B)
    c_shape = _shape_of(C)
    if len(a_shape) != 3 or len(b_shape) != 3 or len(c_shape) != 3:
        raise BusinessError(
            code=E_SHAPE_MISMATCH,
            stage="bmm_rank_validation",
            message="BMM expects A, B, and C to be three-dimensional tensors.",
            shape={"A": a_shape, "B": b_shape, "C": c_shape},
        )
    batch_sizes = {"A": a_shape[0], "B": b_shape[0], "C": c_shape[0]}
    if len(set(batch_sizes.values())) != 1:
        batch_index = min(batch_sizes.values())
        raise BusinessError(
            code=E_SHAPE_MISMATCH,
            stage="bmm_batch_validation",
            message="BMM batch sizes are inconsistent.",
            shape={"A": a_shape, "B": b_shape, "C": c_shape},
            batch_index=int(batch_index),
            details={"batch_sizes": batch_sizes},
        )
    batch_size = a_shape[0]
    expected_b = [batch_size, a_shape[2], b_shape[2]]
    expected_c = [batch_size, a_shape[1], b_shape[2]]
    for batch_index in range(batch_size):
        if a_shape[2] != b_shape[1]:
            raise BusinessError(
                code=E_SHAPE_MISMATCH,
                stage="bmm_shape_validation",
                message="BMM inner dimensions are incompatible.",
                shape={"A": a_shape, "B": b_shape, "C": c_shape},
                batch_index=batch_index,
                details={"required": "A.shape[2] == B.shape[1]"},
            )
        if b_shape[:2] != expected_b[:2] or b_shape[2] != expected_b[2]:
            raise BusinessError(
                code=E_SHAPE_MISMATCH,
                stage="bmm_shape_validation",
                message="BMM B shape is incompatible with A.",
                shape={"A": a_shape, "B": b_shape, "C": c_shape},
                batch_index=batch_index,
                details={"expected_B": expected_b},
            )
        if c_shape != expected_c:
            raise BusinessError(
                code=E_SHAPE_MISMATCH,
                stage="bmm_shape_validation",
                message="BMM C shape is incompatible with the expected output.",
                shape={"A": a_shape, "B": b_shape, "C": c_shape},
                batch_index=batch_index,
                details={"expected_C": expected_c},
            )
    A32 = _as_float32(A, "A", "input_validation")
    B32 = _as_float32(B, "B", "input_validation")
    C32 = _as_float32(C, "C", "input_validation")
    for name, array in (("A", A32), ("B", B32), ("C", C32)):
        _validate_finite(array, name, "input_validation")
    alpha_value = _validate_scalar(alpha, "alpha", "input_validation")
    beta_value = _validate_scalar(beta, "beta", "input_validation")
    return A32, B32, C32, alpha_value, beta_value
