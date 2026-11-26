from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def canonical_dumps(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(value: Any, length: int | None = None) -> str:
    raw = canonical_dumps(value).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return digest if length is None else digest[:length]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def round_div(numerator: int, denominator: int) -> tuple[int, str]:
    """Floor-compatible deterministic nearest division.

    Ties are rounded toward zero.  The direction is relative to the exact
    real quotient after rounding to the returned integer.
    """
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    absolute = abs(numerator)
    quotient, remainder = divmod(absolute, denominator)
    twice = remainder * 2
    if twice > denominator:
        quotient += 1
    result = sign * quotient
    if remainder == 0:
        direction = "exact"
    elif result * denominator > numerator:
        direction = "up"
    else:
        direction = "down"
    return result, direction


def quantize_fixed(value: int, shift: int) -> tuple[int, str]:
    """Quantize an integer fixed-point value with half-step ties toward zero."""
    if shift < 0:
        raise ValueError("shift must be non-negative")
    if shift == 0:
        return value, "exact"
    step = 1 << shift
    half = step >> 1
    modulus = value % step
    lower = value - modulus
    if modulus < half:
        result = lower
    elif modulus > half:
        result = lower + step
    else:
        result = lower if value >= 0 else lower + step
    if result == value:
        direction = "exact"
    elif result > value:
        direction = "up"
    else:
        direction = "down"
    return result, direction


def stable_sorted(items: Iterable[Any], key) -> list[Any]:
    return sorted(items, key=key)


def coordinate_key(point: dict[str, int] | tuple[int, int]) -> tuple[int, int]:
    if isinstance(point, tuple):
        x, y = point
    else:
        x, y = point["x"], point["y"]
    return int(x), int(y)
