"""Canonical JSON, hashing, and numeric helpers.

All fingerprints and snapshot manifests use this module.  Floats are rounded
before serialization so that logically equal local computations produce the
same persisted bytes across a process restart.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


FLOAT_DIGITS = 12


def normalize(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be canonicalized")
        rounded = round(value, FLOAT_DIGITS)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def require_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    from .errors import E_CONFIG_INVALID, ValidationError

    if not is_int(value):
        raise ValidationError(E_CONFIG_INVALID, "must be an integer", path)
    if minimum is not None and value < minimum:
        raise ValidationError(E_CONFIG_INVALID, f"must be >= {minimum}", path)
    return value


def require_number(value: Any, path: str) -> float:
    from .errors import E_CONFIG_INVALID, ValidationError

    if not is_number(value):
        raise ValidationError(E_CONFIG_INVALID, "must be a finite number", path)
    return float(value)


def require_text(value: Any, path: str, *, allow_empty: bool = False) -> str:
    from .errors import E_CONFIG_INVALID, ValidationError

    if not isinstance(value, str):
        raise ValidationError(E_CONFIG_INVALID, "must be a string", path)
    if not allow_empty and not value.strip():
        raise ValidationError(E_CONFIG_INVALID, "must not be empty", path)
    return value
