from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import yaml


ENGINE_VERSION = "probcal-engine-v1"
STAGE_ORDER = ("data", "model", "calibration", "metrics")
SCHEME_ORDER = ("raw", "platt", "isotonic", "histogram")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def round_half_up_ratio(numerator: int, denominator: int) -> int:
    if denominator == 0:
        return 0
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    numerator = abs(numerator)
    denominator = abs(denominator)
    return sign * ((2 * numerator + denominator) // (2 * denominator))


def decimal_to_fixed(value: Any, scale: int) -> int:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"not a decimal value: {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"not a finite decimal value: {value!r}")
    return int((parsed * scale).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def fixed_to_decimal(value: int, scale: int, decimal_places: int) -> str:
    scaled = Decimal(value) / Decimal(scale)
    quantum = Decimal(1).scaleb(-decimal_places)
    return format(scaled.quantize(quantum, rounding=ROUND_HALF_UP), "f")


def quantize_probability(value: float, levels: int, scale: int) -> int:
    clipped = min(1.0, max(0.0, float(value)))
    level = int(math.floor(clipped * levels + 0.5))
    level = min(levels, max(0, level))
    return min(scale, max(0, round_half_up_ratio(level * scale, levels)))


def fixed_probability_to_float(value: int, scale: int) -> float:
    return min(1.0, max(0.0, value / scale))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(value: Any) -> str:
    return yaml.safe_dump(
        value,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def stage_rank(stage: str) -> int:
    return STAGE_ORDER.index(stage)


def scheme_rank(scheme: str) -> int:
    return SCHEME_ORDER.index(scheme)


def failure(code: str, stage: str, field: str, message: str) -> dict[str, str]:
    return {
        "code": code,
        "stage": stage,
        "field": field,
        "message": message,
    }


def sort_failures(failures: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        failures,
        key=lambda item: (
            stage_rank(item.get("stage", "")) if item.get("stage") in STAGE_ORDER else -1,
            item.get("field", ""),
            item.get("code", ""),
            item.get("message", ""),
        ),
    )

