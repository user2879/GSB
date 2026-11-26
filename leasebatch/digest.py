"""Byte-stable canonical serialization and integer fixed-point watermarks.

All hashing in the project goes through :func:`canonical_json` so that the
result is independent of dictionary insertion order, whitespace and platform
floating point behaviour.  Every value that reaches a digest is an integer,
string, bool, ``None`` or a (sorted) list/dict of those.
"""

import hashlib
import json


def canonical_json(obj):
    """Return a deterministic UTF-8 byte string for ``obj``."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(obj):
    """SHA-256 hex digest of the canonical encoding of ``obj``."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def short_digest(obj, length=16):
    return digest(obj)[:length]


def fixed_point_watermark(matrix):
    """Integer watermark shared by leases, receipts and invalidation.

    A pure function of the integer matrix contents using a canonical encoding,
    so identical fixed-point state always yields the same watermark regardless
    of traversal order.
    """
    rows = [[int(v) for v in row] for row in matrix]
    return digest({"wp": "fixed_point_watermark", "m": rows})
