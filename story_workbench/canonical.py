"""Canonical UTF-8 serialization, hashing, and double-precision helpers."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # numpy is listed as a runtime dependency; this keeps diagnostics possible.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only in minimal containers
    _np = None


def double(value: Any) -> float:
    """Coerce one finite value to an IEEE-754 binary64 represented by float."""

    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric coefficient")
    if _np is not None:
        result = float(_np.float64(value))
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite double")
    return 0.0 if result == 0.0 else result


def add_double(left: float, right: float) -> float:
    if _np is not None:
        return double(_np.float64(left) + _np.float64(right))
    return double(left + right)


def multiply_double(left: float, right: float) -> float:
    if _np is not None:
        return double(_np.float64(left) * _np.float64(right))
    return double(left * right)


def negate_double(value: float) -> float:
    if _np is not None:
        return double(-_np.float64(value))
    return double(-value)


def numpy_available() -> bool:
    return _np is not None


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def pretty_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def state_bytes(components: Sequence[str], values: Sequence[float]) -> bytes:
    if len(components) != len(values):
        raise ValueError("component/state length mismatch")
    digest = hashlib.sha256()
    for component, value in zip(components, values):
        digest.update(component.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(struct.pack(">d", double(value)))
        digest.update(b"\x1f")
    return digest.digest()


def state_hash(components: Sequence[str], values: Sequence[float]) -> str:
    return state_bytes(components, values).hex()


def ordered_state(components: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    return {component: double(values[index]) for index, component in enumerate(components)}


def edge_sequence_hash(edge_ids: Iterable[str]) -> str:
    return sha256_text("\n".join(edge_ids))


def replay_hash(edge_ids: Iterable[str], components: Sequence[str], values: Sequence[float]) -> str:
    digest = hashlib.sha256()
    digest.update(edge_sequence_hash(edge_ids).encode("ascii"))
    digest.update(b"\x00")
    digest.update(state_bytes(components, values))
    return digest.hexdigest()


def write_utf8(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")
