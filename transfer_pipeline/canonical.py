"""Canonical serialization, content fingerprints, and atomic JSON writes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ALGORITHM_VERSION = "deterministic-transfer-v1"
SNAPSHOT_SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    """Return a byte-stable JSON representation for hashing and comparison."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint(value: Any, *, domain: str = "object") -> str:
    """Hash a JSON-compatible value with an explicit domain separator."""

    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(canonical_json(value).encode("utf-8"))
    return digest.hexdigest()


def array_fingerprint(array: np.ndarray) -> str:
    """Hash an array's shape and little-endian float64 bytes."""

    contiguous = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(b"\x00")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def parameters_fingerprint(parameters: Mapping[str, Any]) -> str:
    """Hash named parameter arrays in sorted-name order."""

    material = {
        name: array_fingerprint(np.asarray(parameters[name], dtype=np.float64))
        for name in sorted(parameters)
    }
    return fingerprint(material, domain="named-parameters")


def ensure_finite_number(value: Any, label: str) -> float:
    """Convert a JSON scalar to a finite float or raise a stable ValueError."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def json_ready(value: Any) -> Any:
    """Recursively convert NumPy values to JSON-native deterministic values."""

    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite float is not JSON serializable")
    return value


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
