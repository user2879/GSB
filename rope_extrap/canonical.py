from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_float(value: float) -> float:
    """Return a finite JSON-safe float or raise.

    Python's JSON encoder emits round-trip floats.  We nevertheless reject
    NaN/Infinity at every public boundary because those values do not have a
    portable deterministic representation.
    """

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite float")
    return value


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None

