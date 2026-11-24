"""Explicit stable serialization rules.

The canonical form is JSON with:

* object keys sorted by Unicode code point,
* no insignificant whitespace (``,`` and ``:`` separators only),
* floats rendered as fixed 6-decimal strings (NaN/Inf are rejected),
* numpy scalar/array values converted to plain Python values first.

Every identifier and ledger digest in batchdiag is derived from this exact
byte string, so two runs with the same logical data always agree byte for
byte regardless of dict insertion order or dispatch order.
"""

import hashlib
import math

import numpy as np


def _stable(value):
    if isinstance(value, dict):
        return {str(k): _stable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _stable(value.item())
    if isinstance(value, np.ndarray):
        return [_stable(v) for v in value.tolist()]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("non-finite float cannot be canonicalized")
        return format(round(value, 6), ".6f")
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"type not supported by canonical form: {type(value)!r}")


def canonical_float(value) -> str:
    if value is None:
        raise TypeError("cannot canonicalize None as float")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise ValueError("non-finite float cannot be canonicalized")
    return format(round(number, 6), ".6f")


def dumps(value) -> str:
    """Return the canonical JSON string for *value*."""
    import json

    return json.dumps(
        _stable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def dump_bytes(value) -> bytes:
    return dumps(value).encode("utf-8")


def digest(value) -> str:
    """SHA-256 hex digest of the canonical bytes of *value*."""
    return hashlib.sha256(dump_bytes(value)).hexdigest()


def short_digest(value, length: int = 16) -> str:
    return digest(value)[:length]
