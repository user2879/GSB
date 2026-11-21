"""Exact fixed-point helpers and small filesystem primitives."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from decimal import Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path
from typing import Any, Mapping


ZERO = Decimal("0")
ONE = Decimal("1")


def as_decimal(value: Any, *, field: str = "value") -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite decimal, not bool")
    try:
        dec = Decimal(str(value))
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise ValueError(f"{field} is not decimal-compatible") from exc
    if not dec.is_finite():
        raise ValueError(f"{field} must be finite")
    return dec


def decimal_text(value: Decimal) -> str:
    """Return a stable, non-exponent decimal representation."""
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return text


def jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return decimal_text(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def quantize_fixed(value: Decimal | int | str, scale: int, quantum: int) -> int:
    """Quantize a decimal into integer minor units at a configured step."""
    if scale <= 0 or quantum <= 0:
        raise ValueError("scale and quantum must be positive")
    dec = as_decimal(value)
    raw_units = dec * Decimal(scale)
    steps = (raw_units / Decimal(quantum)).to_integral_value(rounding=ROUND_HALF_UP)
    return int(steps) * quantum


def quantize_units(value: Decimal | int | str, quantum: int) -> int:
    """Quantize a value that is already expressed in fixed minor units."""
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    dec = as_decimal(value)
    steps = (dec / Decimal(quantum)).to_integral_value(rounding=ROUND_HALF_UP)
    return int(steps) * quantum


def decay_fixed(delta_seconds: int, half_life: int, scale: int, quantum: int) -> int:
    """Return floor-independent, quantized 0.5**(delta/half_life) in fixed units."""
    if delta_seconds <= 0:
        return scale
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    ratio = Decimal(delta_seconds) / Decimal(half_life)
    if ratio > Decimal("20000"):
        return 0
    with localcontext() as ctx:
        ctx.prec = 90
        coefficient = (Decimal(1) / Decimal(2)) ** ratio
    return quantize_fixed(coefficient, scale, quantum)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def atomic_write_bytes(path: str | Path, payload: bytes, temp_dir: str | Path | None = None) -> None:
    """Write a file in a sibling temp directory and publish it with os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(temp_dir) if temp_dir is not None else target.parent
    temp_root.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(temp_root))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def atomic_write_text(path: str | Path, text: str, temp_dir: str | Path | None = None) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), temp_dir=temp_dir)


def same_volume(first: str | Path, second: str | Path) -> bool:
    first_path = Path(first)
    second_path = Path(second)
    first_path.mkdir(parents=True, exist_ok=True)
    second_path.mkdir(parents=True, exist_ok=True)
    return os.stat(first_path).st_dev == os.stat(second_path).st_dev


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any, temp_dir: str | Path | None = None, *, pretty: bool = True) -> None:
    text = json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    if pretty:
        text += "\n"
    atomic_write_text(path, text, temp_dir=temp_dir)
