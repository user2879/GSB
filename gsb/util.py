"""Deterministic integer and fingerprint utilities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_obj(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def short_fingerprint(value: str, size: int = 16) -> str:
    return value[:size]


def stable_key(*parts: object) -> str:
    return "|".join(str(part) for part in parts)


def task_sort_key(
    file_name: str,
    stage: str,
    column_index: int = 0,
    character_index: int = 0,
    extra: int = 0,
) -> str:
    return f"{file_name}|{stage}|{column_index:06d}|{character_index:06d}|{extra:010d}"


def floor_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return numerator // denominator


def round_half_up(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def ratio_fixed(numerator: int, denominator: int, bits: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if bits < 0:
        raise ValueError("bits must be non-negative")
    return round_half_up(numerator << bits, denominator)


def quantize_fixed(value: int, bits: int) -> int:
    if bits < 0:
        raise ValueError("bits must be non-negative")
    return round_half_up(value, 1 << bits) << bits


def fixed_to_rounded_int(value: int, bits: int) -> int:
    return round_half_up(value, 1 << bits)


def rounding_direction(raw_fixed: int, quantized_fixed: int) -> str:
    if quantized_fixed > raw_fixed:
        return "UP"
    if quantized_fixed < raw_fixed:
        return "DOWN"
    return "EXACT"


def residual_fixed(raw_fixed: int, quantized_fixed: int) -> int:
    return abs(raw_fixed - quantized_fixed)


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def error_measure(reference: int, observed: int, truth_zero: bool = False) -> dict[str, Any]:
    absolute = abs(observed - reference)
    if truth_zero or reference == 0:
        return {
            "branch": "absolute",
            "absolute_error": absolute,
            "relative_error_ppm": None,
        }
    return {
        "branch": "relative",
        "absolute_error": absolute,
        "relative_error_ppm": round_half_up(absolute * 1_000_000, abs(reference)),
    }


def first_exceeding_stage(
    stage_errors: Iterable[tuple[str, int]],
    tolerance: int,
) -> str | None:
    for stage, error in stage_errors:
        if error > tolerance:
            return stage
    return None


def safe_relative_path(base: Path, value: str) -> Path:
    candidate = (base / value).resolve()
    root = base.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes base directory: {value}")
    return candidate


def json_dump_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
