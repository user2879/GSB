"""Determinism primitives shared by every component.

The whole diagnostic workbench is a pure function of its four inputs
(config / samples / scale descriptor / injection manifest).  To guarantee that
an incremental rebuild is byte-for-byte (field-for-field) identical to a cold
full rebuild we:

* serialize every canonical object with sorted keys and a fixed float format;
* forbid NaN/Infinity in JSON (they would violate strict JSON and break
  cross-platform hashes);
* derive all fingerprints from SHA-256 over those canonical bytes;
* use a single ``np.random.Generator`` seeded per sample so results do not
  depend on thread scheduling.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays into JSON-native objects."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def canonical_bytes(obj: Any) -> bytes:
    """Canonical JSON bytes: sorted keys, fixed separators, no whitespace."""
    return json.dumps(
        _to_native(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json(obj: Any, indent: int | None = 2) -> str:
    """Pretty (or compact) canonical JSON string used for report files."""
    return json.dumps(
        _to_native(obj),
        sort_keys=True,
        indent=indent,
        separators=(",", ": ") if indent is not None else (",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(obj: Any) -> str:
    """SHA-256 fingerprint of any JSON-serializable object."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def short_hash(obj: Any, length: int = 16) -> str:
    return content_hash(obj)[:length]


def float_to_str(value: float | np.floating, digits: int = 17) -> str:
    """Round-trip-stable float rendering.

    ``repr`` gives enough digits for an exact float64 round trip; the explicit
    fixed-:func:`format` keeps the report deterministic across platforms while
    retaining full precision (17 significant digits).
    """
    value = float(value)
    if value == 0.0:
        # collapse -0.0 / +0.0 so the sign bit cannot perturb equivalence
        return "0.0"
    if not math.isfinite(value):  # pragma: no cover - guarded earlier
        raise ValueError("non-finite value cannot be serialized as JSON number")
    return format(value, ".{}g".format(digits))


def stable_round(value: float | np.floating, digits: int = 12) -> float:
    """Round to a fixed number of significant decimals for compact display."""
    value = float(value)
    if value == 0.0 or not math.isfinite(value):
        return 0.0
    return float(format(value, ".{}g".format(digits)))


def rng_for(seed: int, *parts: Any) -> np.random.Generator:
    """Deterministic per-entity RNG derived from a seed plus stable parts."""
    material = canonical_bytes([seed, list(parts)])
    digest = hashlib.sha256(material).digest()
    child_seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(child_seed)


def stable_task_key(scheme: str, sample_id: str, stage: str) -> str:
    """Sorting/identity key for a (scheme, sample, stage) task."""
    return "{}|{}|{}".format(scheme, sample_id, stage)


def sorted_task_keys(keys: Iterable[str]) -> list[str]:
    return sorted(keys)


DTYPE_BY_NAME = {
    "float64": np.float64,
    "float32": np.float32,
}


def resolve_dtype(name: str) -> np.dtype:
    if name not in DTYPE_BY_NAME:
        raise ValueError("unsupported precision tier: {}".format(name))
    return np.dtype(DTYPE_BY_NAME[name])
