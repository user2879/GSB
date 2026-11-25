"""Deterministic encoding, hashing and monotonic clock helpers."""

import hashlib
import json
import time


def canonical_json(value):
    """Serialize with sorted keys and stable separators for byte comparison."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def sha256_hex(value):
    payload = canonical_json(value) if not isinstance(value, (bytes, str)) else value
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def now_ms():
    """Monotonic clock in whole milliseconds (int64-safe on Windows)."""
    return int(time.monotonic() * 1000.0)


def wall_ms():
    """Wall clock milliseconds, used only for human diagnostics."""
    return int(time.time() * 1000.0)


def is_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_non_negative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0

