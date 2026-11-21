"""Canonical serialization, hashing and deterministic numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically and reject non-finite values."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(array: np.ndarray, *, dtype: str = "<f8") -> str:
    """Hash an array together with its shape and explicit little-endian dtype."""

    contiguous = np.ascontiguousarray(array, dtype=dtype)
    header = canonical_json(
        {"dtype": np.dtype(dtype).str, "shape": list(contiguous.shape)}
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_float(value: float) -> float:
    """Normalize negative zero so JSON round-trips compare exactly."""

    result = float(value)
    if result == 0.0:
        return 0.0
    return result


def truncate_significant(value: float, digits: int) -> float:
    """Truncate a positive scale toward zero at ``digits`` significant digits.

    The Decimal path is intentional: it removes platform-dependent shortest
    float formatting and gives the same scale across repeated runs.
    """

    number = abs(float(value))
    if not math.isfinite(number):
        raise ValueError("scale must be finite")
    if number == 0.0:
        return 0.0
    if digits <= 0:
        raise ValueError("significant digits must be positive")

    with localcontext() as context:
        context.prec = max(digits + 40, 80)
        decimal = Decimal.from_float(number)
        exponent = decimal.adjusted()
        quantum = Decimal(1).scaleb(exponent - digits + 1)
        try:
            truncated = decimal.quantize(quantum, rounding="ROUND_DOWN")
        except InvalidOperation:
            # Extremely small subnormals can fall outside the default exponent
            # range. Retry with a wider exponent range.
            context.Emin = -999999
            context.Emax = 999999
            truncated = decimal.quantize(quantum, rounding="ROUND_DOWN")
        result = float(truncated)
    return normalize_float(result)


def ordered_dot(left: np.ndarray, right: np.ndarray) -> float:
    """Dot product with a fixed left-to-right accumulation order."""

    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("dot operands have incompatible sizes")
    total = 0.0
    for index in range(a.size):
        total += float(a[index]) * float(b[index])
    return total


def ordered_error_stats(difference: np.ndarray) -> dict[str, float | int]:
    """Compute max-abs, MAE, SSE and MSE in original C-order."""

    values = np.asarray(difference, dtype=np.float64).reshape(-1, order="C")
    maximum = 0.0
    absolute_sum = 0.0
    squared_sum = 0.0
    for item in values:
        absolute = abs(float(item))
        if absolute > maximum:
            maximum = absolute
        absolute_sum += absolute
        squared_sum += absolute * absolute
    count = int(values.size)
    return {
        "max_abs_error": normalize_float(maximum),
        "mean_abs_error": normalize_float(absolute_sum / count) if count else 0.0,
        "sum_squared_error": normalize_float(squared_sum),
        "mse": normalize_float(squared_sum / count) if count else 0.0,
        "element_count": count,
    }


def ordered_min_max(values: np.ndarray) -> tuple[float, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1, order="C")
    if flat.size == 0:
        return 0.0, 0.0
    minimum = float(flat[0])
    maximum = float(flat[0])
    for item in flat[1:]:
        number = float(item)
        if number < minimum:
            minimum = number
        if number > maximum:
            maximum = number
    return minimum, maximum


def ordered_abs_max(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1, order="C")
    maximum = 0.0
    for item in flat:
        absolute = abs(float(item))
        if absolute > maximum:
            maximum = absolute
    return maximum


def ordered_sum(values: list[float]) -> float:
    total = 0.0
    for item in values:
        total += float(item)
    return total


def to_builtin(value: Any) -> Any:
    """Convert numpy and tuples into JSON-safe builtin values recursively."""

    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite float cannot be serialized")
    return value
