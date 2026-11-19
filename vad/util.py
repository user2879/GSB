"""Small deterministic utilities shared by the VAD pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PRODUCT_VERSIONS = {
    "policy": "1.0.0",
    "features": "1.0.0",
    "state_machine": "1.0.0",
    "endpoints": "1.0.0",
    "snapshot": "1.0.0",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON to a sibling temporary file and atomically replace target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync; unsupported on Windows."""

    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def round_significant(values: np.ndarray, digits: int) -> np.ndarray:
    """Round a float64 array to a fixed number of significant decimal digits."""

    array = np.asarray(values, dtype=np.float64)
    if digits < 1:
        raise ValueError("significant digits must be positive")
    result = np.zeros_like(array, dtype=np.float64)
    nonzero = np.isfinite(array) & (array != 0.0)
    if np.any(nonzero):
        selected = array[nonzero]
        exponent = np.floor(np.log10(np.abs(selected)))
        scale = np.power(10.0, float(digits - 1) - exponent)
        result[nonzero] = np.round(selected * scale) / scale
    return result


def round_float(value: float, digits: int) -> float:
    array = round_significant(np.asarray([value], dtype=np.float64), digits)
    result = float(array[0])
    return 0.0 if result == 0.0 else result


def safe_stat_summary(values: Iterable[float], digits: int) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round_float(float(np.mean(array)), digits),
        "std": round_float(float(np.std(array)), digits),
        "min": round_float(float(np.min(array)), digits),
        "max": round_float(float(np.max(array)), digits),
    }


def log_rms(frame: np.ndarray, floor: float) -> float:
    value = float(np.sqrt(np.mean(frame * frame, dtype=np.float64)))
    return math.log(max(value, floor))


def quantile(values: np.ndarray, q: float, method: str = "linear") -> float:
    """Deterministic NumPy quantile wrapper with compatibility fallback."""

    try:
        return float(np.quantile(values, q, method=method))
    except TypeError:  # NumPy < 1.22 uses interpolation=.
        return float(np.quantile(values, q, interpolation=method))
