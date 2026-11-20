"""Deterministic integer, hashing, JSON and file helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_digest(value: Any, length: int = 20) -> str:
    return sha256_json(value)[:length]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -((-numerator) // denominator)


def fixed_from_ns(delay_ns: int, scale_power: int) -> int:
    """Floor an exact nanosecond value to integer fixed-point milliseconds."""

    scale = 10**scale_power
    if delay_ns >= 0:
        return (delay_ns * scale) // 1_000_000
    return -ceil_div(-delay_ns * scale, 1_000_000)


def fixed_ceil_from_ns(value_ns: int, scale_power: int) -> int:
    """Ceil a nonnegative nanosecond magnitude into fixed-point units."""

    if value_ns < 0:
        raise ValueError("value_ns must be nonnegative")
    return ceil_div(value_ns * (10**scale_power), 1_000_000)


def format_fixed(value: int, scale_power: int) -> str:
    scale = 10**scale_power
    sign = "-" if value < 0 else ""
    whole, fraction = divmod(abs(value), scale)
    if scale_power == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{fraction:0{scale_power}d}"


def atomic_write_bytes(path: str | Path, data: bytes, *, read_only: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        if read_only:
            os.chmod(destination, 0o444)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: str | Path, text: str, *, read_only: bool = False) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), read_only=read_only)


def atomic_write_json(
    path: str | Path, value: Any, *, pretty: bool = True, read_only: bool = False
) -> None:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    else:
        text = canonical_json(value) + "\n"
    atomic_write_text(path, text, read_only=read_only)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_writable_by_mode(path: str | Path) -> bool:
    mode = os.stat(path).st_mode
    return bool(mode & 0o222)


def first_sequence_gap(seqs: Iterable[int], start: int = 1) -> int | None:
    expected = start
    for seq in sorted(seqs):
        if seq != expected:
            return seq
        expected += 1
    return None


def deep_first_diff(expected: Any, actual: Any, path: str = "$") -> dict[str, Any] | None:
    if type(expected) is not type(actual):
        return {"path": path, "expected": expected, "actual": actual, "kind": "type"}
    if isinstance(expected, dict):
        expected_keys = sorted(expected)
        actual_keys = sorted(actual)
        for key in expected_keys:
            if key not in actual:
                return {
                    "path": f"{path}.{key}",
                    "expected": expected[key],
                    "actual": "<missing>",
                    "kind": "missing",
                }
        for key in actual_keys:
            if key not in expected:
                return {
                    "path": f"{path}.{key}",
                    "expected": "<missing>",
                    "actual": actual[key],
                    "kind": "extra",
                }
        for key in expected_keys:
            diff = deep_first_diff(expected[key], actual[key], f"{path}.{key}")
            if diff is not None:
                return diff
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return {
                "path": f"{path}.length",
                "expected": len(expected),
                "actual": len(actual),
                "kind": "length",
            }
        for index, (left, right) in enumerate(zip(expected, actual)):
            diff = deep_first_diff(left, right, f"{path}[{index}]")
            if diff is not None:
                return diff
        return None
    if expected != actual:
        return {"path": path, "expected": expected, "actual": actual, "kind": "value"}
    return None


def bytes_to_human_size(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    units = ("KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        amount /= 1024.0
        if amount < 1024.0:
            return f"{amount:.1f} {unit}"
    return f"{amount:.1f} PiB"


def safe_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def python_major_minor() -> tuple[int, int]:
    import sys

    return sys.version_info.major, sys.version_info.minor


def power_ten_scale(scale_power: int) -> int:
    if not isinstance(scale_power, int) or scale_power < 0:
        raise ValueError("scale_power must be a nonnegative integer")
    return int(math.pow(10, scale_power))
