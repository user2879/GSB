"""Canonical JSON, hashing, fixed-point arithmetic, and atomic file helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def clamp_int(value: int, low: int, high: int) -> int:
    return low if value < low else high if value > high else value


def round_div(numerator: int, denominator: int) -> int:
    """Round a rational to nearest integer, ties to even, without floats."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    twice = remainder * 2
    if twice > denominator or (twice == denominator and quotient % 2 == 1):
        quotient += 1
    return sign * quotient


def decimal_to_fixed(value: Any, decimal_places: int) -> int:
    """Convert a JSON scalar to a deterministic fixed-point integer."""
    decimal = Decimal(str(value))
    scaled = decimal.scaleb(decimal_places)
    return int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))


def fixed_to_float(value: int, decimal_places: int) -> float:
    return value / float(10**decimal_places)


def quantize_levels(value: int, scale: int, levels: int) -> int:
    """Snap a fixed value in [0, scale] to one of ``levels`` evenly spaced codes."""
    if levels < 2:
        raise ValueError("levels must be at least 2")
    bounded = clamp_int(value, 0, scale)
    level = round_div(bounded * (levels - 1), scale)
    level = clamp_int(level, 0, levels - 1)
    return round_div(level * scale, levels - 1)


def mul_fixed(a: int, b: int, scale: int) -> int:
    return round_div(a * b, scale)


def normalize_fixed(value: float, low: float, high: float, scale: int) -> int:
    if high <= low:
        raise ValueError("feature range high must exceed low")
    ratio = (float(value) - low) / (high - low)
    ratio = 0.0 if ratio < 0.0 else 1.0 if ratio > 1.0 else ratio
    return clamp_int(int(round(ratio * scale)), 0, scale)


def stable_softmax_fixed(scores: Iterable[tuple[str, int]], scale: int, levels: int) -> dict[str, int]:
    """Stable softmax over integer scores, preserving lexicographic tie behavior."""
    ordered = sorted(scores, key=lambda item: item[0])
    if not ordered:
        return {}
    maximum = max(score for _, score in ordered)
    exponentials: list[tuple[str, float]] = []
    for label, score in ordered:
        exponentials.append((label, math.exp((score - maximum) / float(scale))))
    denominator = math.fsum(value for _, value in exponentials)
    result: dict[str, int] = {}
    for label, value in exponentials:
        fixed = round_div(int(round((value / denominator) * scale)), 1)
        result[label] = quantize_levels(fixed, scale, levels)
    return result


def stable_sort_tasks(tasks: Iterable[Any], key: str = "sort_key") -> list[Any]:
    if hasattr(tasks, "__iter__") and not isinstance(tasks, list):
        tasks = list(tasks)
    return sorted(tasks, key=lambda item: item[key])


def safe_relative_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes sample root: {relative}") from exc
    return candidate
