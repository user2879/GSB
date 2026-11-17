"""Deterministic hashing, JSON encoding and atomic snapshot replacement."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Iterable

import numpy as np

# Rounding every persisted float keeps full vs incremental reports byte-identical
# regardless of minor evaluation-order differences across code paths.
FLOAT_DIGITS = 10


def native(value: Any) -> Any:
    """Convert numpy scalars/arrays into JSON-native python values."""
    if isinstance(value, (np.floating,)):
        return round(float(value), FLOAT_DIGITS)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [native(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, float):
        return round(value, FLOAT_DIGITS)
    return value


def rounded_float(value: Any) -> float:
    """Single rounding rule shared by in-memory and on-disk numeric values."""
    return round(float(value), FLOAT_DIGITS)


def ordered_sum(values: Iterable[float]) -> float:
    """Deterministic left-to-right python sum (no pairwise reordering)."""
    total = 0.0
    for value in values:
        total = total + rounded_float(value)
    return round(total, FLOAT_DIGITS)


def dumps_canonical(payload: Any) -> str:
    """Canonical JSON text: sorted keys, fixed separators, no whitespace drift."""
    return json.dumps(
        native(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_fingerprint(payload: Any) -> str:
    """SHA-256 fingerprint of the canonical JSON encoding of payload."""
    digest = hashlib.sha256()
    digest.update(dumps_canonical(payload).encode("utf-8"))
    return digest.hexdigest()


def derived_key(parts: Iterable[Any]) -> str:
    """Cache key derived deterministically from upstream fingerprints/params."""
    return content_fingerprint(list(parts))


def atomic_write_json(path: str, payload: Any) -> None:
    """Write to a temp file in the same directory then atomically replace."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(dumps_canonical(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def matrix_summary(matrix: np.ndarray) -> dict:
    """Stable summary of a numeric matrix used in reports and fingerprints."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.size == 0:
        return {
            "shape": list(arr.shape),
            "sum": 0.0,
            "sumsq": 0.0,
            "fingerprint": content_fingerprint([]),
        }
    return {
        "shape": list(arr.shape),
        "sum": ordered_sum(arr.flatten().tolist()),
        "sumsq": ordered_sum((arr * arr).flatten().tolist()),
        "fingerprint": content_fingerprint(native(arr)),
    }
