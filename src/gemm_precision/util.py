from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RELATIVE_ERROR_DENOMINATOR_FLOOR = 1.0e-30


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        to_builtin(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finite_float(value: Any) -> float | str:
    converted = float(value)
    if math.isnan(converted):
        return "nan"  # type: ignore[return-value]
    if math.isinf(converted):
        return "inf" if converted > 0.0 else "-inf"  # type: ignore[return-value]
    return converted


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_builtin(value.item())
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return finite_float(value)
    return value


def build_index_bands(size: int, requested_count: int) -> list[dict[str, int]]:
    count = max(1, min(int(size), int(requested_count)))
    base, remainder = divmod(int(size), count)
    bands: list[dict[str, int]] = []
    start = 0
    for index in range(count):
        width = base + (1 if index < remainder else 0)
        end = start + width
        bands.append({"index": index, "start": start, "end": end})
        start = end
    return bands


def band_for_index(index: int, bands: Iterable[dict[str, int]]) -> dict[str, int]:
    for band in bands:
        if int(band["start"]) <= index < int(band["end"]):
            return band
    raise IndexError(f"Index {index} is outside band definitions.")


def yaml_like_json(data: Any) -> str:
    return json.dumps(to_builtin(data), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
