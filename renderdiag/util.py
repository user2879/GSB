from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

from .errors import E_CONFIG_INVALID, fail


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def fingerprint(value: Any, domain: str = "renderdiag-v1") -> str:
    return sha256_text(domain + "\x00" + canonical_json(value))


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(
    stage: str,
    upstream_fingerprint: str,
    parameter_fingerprint: str,
    quantization_level: int,
    snapshot_version: int,
) -> str:
    material = {
        "domain": "renderdiag-cache-key-v1",
        "stage": stage,
        "upstream_fingerprint": upstream_fingerprint,
        "parameter_fingerprint": parameter_fingerprint,
        "quantization_level": int(quantization_level),
        "snapshot_version": int(snapshot_version),
    }
    return fingerprint(material, "renderdiag-cache-key")


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def to_int(value: Any, entry: str, *, minimum: int | None = None) -> int:
    if not is_int(value):
        fail(E_CONFIG_INVALID, entry, "expected an integer", detail={"value": repr(value)})
    result = int(value)
    if minimum is not None and result < minimum:
        fail(
            E_CONFIG_INVALID,
            entry,
            f"expected integer >= {minimum}",
            detail={"value": result},
        )
    return result


def to_decimal(value: Any, entry: str, *, nonnegative: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        fail(E_CONFIG_INVALID, entry, "expected a finite decimal", detail={"value": repr(value)})
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError):
        fail(E_CONFIG_INVALID, entry, "expected a finite decimal", detail={"value": repr(value)})
    if not dec.is_finite():
        fail(E_CONFIG_INVALID, entry, "expected a finite decimal", detail={"value": str(dec)})
    if nonnegative and dec < 0:
        fail(E_CONFIG_INVALID, entry, "expected a non-negative decimal", detail={"value": str(dec)})
    return dec


def decimal_text(value: Any) -> str:
    return format(Decimal(str(value)).normalize(), "f")


def decimal_to_fixed(value: Any, frac_bits: int, entry: str) -> int:
    dec = to_decimal(value, entry)
    scaled = dec * (1 << frac_bits)
    try:
        return int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
    except (ValueError, OverflowError):
        fail(E_CONFIG_INVALID, entry, "decimal cannot be represented", detail={"value": str(dec)})
    raise AssertionError("unreachable")


def round_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("round_div denominator is zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    a, b = abs(int(numerator)), abs(int(denominator))
    q, r = divmod(a, b)
    if 2 * r >= b:
        q += 1
    return sign * q


def quantize(value: int, step: int) -> tuple[int, int, str]:
    if step <= 0:
        raise ValueError("quantization step must be positive")
    quantized = round_div(int(value), step) * step
    residual = quantized - int(value)
    direction = "none" if residual == 0 else ("up" if residual > 0 else "down")
    return quantized, residual, direction


def clamp_int(value: int, low: int, high: int) -> int:
    return low if value < low else high if value > high else value


def stable_uint(*parts: Any, bits: int = 64) -> int:
    data = canonical_json(list(parts)).encode("utf-8")
    digest = hashlib.sha256(data).digest()
    value = int.from_bytes(digest[: max(1, bits // 8)], "big")
    if bits < 64:
        value &= (1 << bits) - 1
    return value


def stable_signed(*parts: Any, limit: int) -> int:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return stable_uint(*parts, bits=64) % (2 * limit + 1) - limit


def deterministic_ints(key: Any, count: int, low: int, high: int) -> list[int]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if high < low:
        raise ValueError("high must be >= low")
    if count == 0:
        return []
    span = high - low + 1
    material = canonical_json(key).encode("utf-8")
    out: list[int] = []
    counter = 0
    while len(out) < count:
        digest = hashlib.sha256(material + b"\x00" + counter.to_bytes(8, "big")).digest()
        for offset in range(0, 32, 4):
            word = int.from_bytes(digest[offset : offset + 4], "big")
            out.append(low + (word % span))
            if len(out) == count:
                break
        counter += 1
    return out


def finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number)


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
        os.replace(tmp_name, target)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    atomic_write_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def load_json(path: str | os.PathLike[str], entry: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        from .errors import E_CONFIG_INVALID
        fail(E_CONFIG_INVALID, entry, "file does not exist", detail={"path": str(path)})
    except json.JSONDecodeError as exc:
        from .errors import E_CONFIG_INVALID
        fail(
            E_CONFIG_INVALID,
            entry,
            "invalid JSON",
            detail={"path": str(path), "line": exc.lineno, "column": exc.colno},
        )
    raise AssertionError("unreachable")


def stable_sort_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in sorted(value)}


def mean_ints(values: Iterable[int]) -> int:
    items = list(values)
    if not items:
        return 0
    return round_div(sum(items), len(items))
