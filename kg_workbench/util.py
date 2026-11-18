"""Small deterministic helpers shared by all stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from decimal import Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path
from typing import Any, Iterable


ZERO = Decimal("0")
ONE = Decimal("1")
DECIMAL_PRECISION = 60


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle, parse_float=Decimal, parse_int=int)


def decimal_string(value: Decimal) -> str:
    if value == 0:
        return "0"
    normalized = value.normalize()
    return format(normalized, "f")


def json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return decimal_string(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dumps_report(value: Any) -> str:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, dumps_report(value))


def as_decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, str, float)):
        from .errors import ConfigError

        raise ConfigError(f"{field} must be a decimal number")
    try:
        if isinstance(value, float):
            return Decimal(str(value))
        return Decimal(value)
    except Exception as exc:  # noqa: BLE001 - converted to a business error
        from .errors import ConfigError

        raise ConfigError(f"{field} is not a valid decimal") from exc


def require_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    from .errors import ConfigError

    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{field} must be >= {minimum}")
    return value


def round_div_half_up(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def to_fixed(value: Decimal, decimals: int) -> int:
    scale = 10**decimals
    return int((value * scale).to_integral_value(rounding=ROUND_HALF_UP))


def quantize_units(value: Decimal, decimals: int, quantization_decimals: int) -> int:
    """Round to the configured fixed-point grid and return integer units."""

    if quantization_decimals < 0 or quantization_decimals > decimals:
        raise ValueError("invalid quantization decimals")
    scaled = to_fixed(value, decimals)
    quantum = 10 ** (decimals - quantization_decimals)
    if scaled >= 0:
        return ((scaled + quantum // 2) // quantum) * quantum
    return -(((-scaled + quantum // 2) // quantum) * quantum)


def quantize_probability(value: Decimal, decimals: int, quantization_decimals: int) -> int:
    value = max(ZERO, min(ONE, value))
    return quantize_units(value, decimals, quantization_decimals)


def units_decimal(units: int, decimals: int) -> Decimal:
    return Decimal(units) / (Decimal(10) ** decimals)


def format_fixed(units: int, decimals: int) -> str:
    scale = 10**decimals
    sign = "-" if units < 0 else ""
    absolute = abs(units)
    whole, fraction = divmod(absolute, scale)
    if decimals == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{fraction:0{decimals}d}"


def sigmoid_decimal(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        if value >= 0:
            z = (-value).exp()
            return ONE / (ONE + z)
        z = value.exp()
        return z / (ONE + z)


def sigmoid_from_units(units: int, decimals: int) -> Decimal:
    return sigmoid_decimal(units_decimal(units, decimals))


def permille(units: int, decimals: int) -> int:
    scale = 10**decimals
    value = max(0, min(scale, units))
    return max(0, min(1000, round_div_half_up(value * 1000, scale)))


def fixed_hash_bucket(key: str, buckets: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % buckets


def ordered_unique(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))
