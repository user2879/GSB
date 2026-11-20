from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .util import array_sha256
from .validation import prepare_bmm_arrays, prepare_gemm_arrays


@dataclass(frozen=True)
class ReferenceResult:
    c: np.ndarray
    mode: str
    input_sha256: str

    @property
    def output_sha256(self) -> str:
        return array_sha256(self.c)

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "path": "float64_serial_k_ascending",
            "shape": [int(item) for item in self.c.shape],
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
        }


def _reference_one(A: np.ndarray, B: np.ndarray, C: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    M, K = A.shape
    N = B.shape[1]
    result = np.empty((M, N), dtype=np.float64)
    beta_value = float(beta)
    alpha_value = float(alpha)
    for i in range(M):
        for j in range(N):
            total = 0.0
            for k in range(K):
                total += float(A[i, k]) * float(B[k, j])
            scaled = alpha_value * total
            result[i, j] = scaled + beta_value * float(C[i, j]) if beta_value != 0.0 else scaled
    return result


def reference_gemm(
    A: Any,
    B: Any,
    C: Any,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> ReferenceResult:
    A32, B32, C32, alpha_value, beta_value = prepare_gemm_arrays(A, B, C, alpha, beta)
    output = _reference_one(A32, B32, C32, alpha_value, beta_value)
    return ReferenceResult(
        c=output,
        mode="gemm",
        input_sha256=array_sha256(A32, B32, C32),
    )


def reference_bmm(
    A: Any,
    B: Any,
    C: Any,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> ReferenceResult:
    A32, B32, C32, alpha_value, beta_value = prepare_bmm_arrays(A, B, C, alpha, beta)
    batch_size = A32.shape[0]
    outputs = [
        _reference_one(A32[batch], B32[batch], C32[batch], alpha_value, beta_value)
        for batch in range(batch_size)
    ]
    output = np.stack(outputs, axis=0)
    return ReferenceResult(
        c=output,
        mode="bmm",
        input_sha256=array_sha256(A32, B32, C32),
    )
