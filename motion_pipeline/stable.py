"""Stable serialization, fingerprints and fixed-point helpers."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def round_half_away(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("cannot round a non-finite value")
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def div_round_half_away(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("denominator must be non-zero")
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def trunc_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("denominator must be non-zero")
    sign = 1 if (numerator >= 0) == (denominator >= 0) else -1
    return sign * (abs(numerator) // abs(denominator))


def float_to_fixed(value: float, bits: int) -> tuple[int, float, str]:
    """Return fixed integer, residual in fixed units and rounding direction."""

    scale = 1 << bits
    scaled = value * scale
    fixed = round_half_away(scaled)
    residual = scaled - fixed
    if residual > 0:
        direction = "down"
    elif residual < 0:
        direction = "up"
    else:
        direction = "exact"
    return fixed, residual, direction


def fixed_to_float(value: int, bits: int) -> float:
    return value / float(1 << bits)


def digest_int_array(values: list[int] | tuple[int, ...] | np.ndarray) -> str:
    if isinstance(values, np.ndarray):
        raw = np.ascontiguousarray(values, dtype="<i8").tobytes()
    else:
        raw = np.asarray(values, dtype="<i8").tobytes()
    return hashlib.sha256(raw).hexdigest()


_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix64_numpy(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=False)
    values = values ^ (values >> np.uint64(30))
    values = values * np.uint64(0xBF58476D1CE4E5B9)
    values = values ^ (values >> np.uint64(27))
    values = values * np.uint64(0x94D049BB133111EB)
    values = values ^ (values >> np.uint64(31))
    return values


def deterministic_noise_pixels(
    *,
    sequence_id: str,
    step: int,
    height: int,
    width: int,
    noise_seed: int,
    amplitude: float,
    bits: int,
) -> np.ndarray:
    """Generate signed integer pixel noise without process-global RNG state."""

    scale = 1 << bits
    amplitude_fixed = min(
        255 * scale,
        round_half_away(max(0.0, amplitude) * scale),
    )
    yy, xx = np.indices((height, width), dtype=np.uint64)
    base = np.uint64(stable_seed(sequence_id) ^ (noise_seed & 0xFFFFFFFFFFFFFFFF))
    keys = (
        base
        ^ (yy * np.uint64(0x9E3779B185EBCA87))
        ^ (xx * np.uint64(0xC2B2AE3D27D4EB4F))
        ^ np.uint64((step + 1) * 0xD6E8FEB86659FD93 & 0xFFFFFFFFFFFFFFFF)
    )
    with np.errstate(over="ignore"):
        hashed = _mix64_numpy(keys)
    if amplitude_fixed == 0:
        return np.zeros((height, width), dtype=np.int32)
    span = 2 * amplitude_fixed + 1
    signed = (hashed % np.uint64(span)).astype(np.int64) - amplitude_fixed
    # Fixed-point rounding toward nearest pixel, deterministically away from zero.
    rounded = np.where(
        signed >= 0,
        (signed + scale // 2) // scale,
        -((-signed + scale // 2) // scale),
    )
    return rounded.astype(np.int32)


def public_failure(
    business_code: str,
    sequence_id: str,
    message: str,
    *,
    line: int | None = None,
    field: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "sequence_id": sequence_id,
        "business_code": business_code,
        "message": message,
    }
    if line is not None:
        entry["line"] = line
    if field is not None:
        entry["field"] = field
    return entry


def failure_sort_key(entry: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(entry.get("sequence_id", "")),
        int(entry.get("line", 0)),
        str(entry.get("business_code", "")),
        str(entry.get("field", "")),
    )


def sort_failures(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=failure_sort_key)
