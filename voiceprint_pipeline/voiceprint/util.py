"""Deterministic primitives: hashing, rounding, fixed-point helpers, atomic IO."""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Iterable

import numpy as np

CANONICAL_KW = dict(sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json(obj: Any) -> str:
    """Canonical JSON text used for fingerprints (key order independent)."""
    return json.dumps(obj, **CANONICAL_KW)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(obj: Any) -> str:
    return sha256_text(canonical_json(_jsonable(obj)))


def stable_seed(seed_text: str) -> int:
    """Map arbitrary text to a positive 31-bit integer seed (no runtime randomness)."""
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def dround_scalar(value: float) -> int:
    """Deterministic round-half-away-from-zero to a Python int."""
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def dround_array(values: np.ndarray) -> np.ndarray:
    """Vectorized deterministic round-half-away-from-zero, dtype int64."""
    out = np.empty(values.shape, dtype=np.int64)
    pos = values >= 0
    np.floor(values + 0.5, out=out, where=pos, casting="unsafe")
    np.ceil(values - 0.5, out=out, where=~pos, casting="unsafe")
    return out


def round_sigfigs(values: np.ndarray, sig_figs: int) -> np.ndarray:
    """Round float64 array to ``sig_figs`` significant decimal digits."""
    result = np.zeros_like(values, dtype=np.float64)
    nonzero = values != 0
    if np.any(nonzero):
        magnitude = np.floor(np.log10(np.abs(values[nonzero])))
        factor = np.power(10.0, sig_figs - 1 - magnitude)
        rounded = dround_array(values[nonzero] * factor).astype(np.float64) / factor
        result[nonzero] = rounded
    return result


def isqrt_scalar(value: int) -> int:
    return math.isqrt(int(value))


def diff_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Per-element absolute difference plus aggregate indicators (float64)."""
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "element_count": int(diff.size),
        "per_element_abs_diff": [float(x) for x in diff.tolist()],
        "max_abs_diff": float(diff.max(initial=0.0)),
        "mean_abs_diff": float(diff.mean(dtype=np.float64)) if diff.size else 0.0,
        "rms_diff": float(np.sqrt(np.mean(diff * diff, dtype=np.float64))) if diff.size else 0.0,
    }


def atomic_write_text(path: str, text: str) -> None:
    """Write to a temp file in the same directory then atomically replace."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{id(text)}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str, obj: Any, indent: int | None = 2) -> None:
    text = json.dumps(_jsonable(obj), sort_keys=True, indent=indent, ensure_ascii=False)
    atomic_write_text(path, text)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sorted_stable(keys: Iterable[str]) -> list[str]:
    return sorted(keys)
