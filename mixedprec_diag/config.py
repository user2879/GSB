from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import E_CONFIG, DiagnosticError


SUPPORTED_BLOCK_KB = frozenset({4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 3072})
SUPPORTED_PRECISIONS = frozenset({"float32", "fixed"})
SUPPORTED_PREALLOCATION = frozenset({"per_batch", "block_pool"})
SUPPORTED_ROUNDING = frozenset({"reject", "floor_to_block"})
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "memory_budget_mib",
        "block_size_kb",
        "precision",
        "quant_scale_power",
        "batch_size_limit",
        "preallocation",
        "budget_rounding",
        "pressure_threshold_ms",
        "ui_query_budget_ms",
    }
)


@dataclass(frozen=True)
class InferenceConfig:
    schema_version: int
    memory_budget_mib: int
    block_size_kb: int
    precision: str
    quant_scale_power: int
    batch_size_limit: int
    preallocation: str
    budget_rounding: str
    pressure_threshold_ms: float
    ui_query_budget_ms: float

    @property
    def block_size_bytes(self) -> int:
        return self.block_size_kb * 1024

    @property
    def requested_budget_bytes(self) -> int:
        return self.memory_budget_mib * 1024 * 1024

    @property
    def effective_budget_bytes(self) -> int:
        return (self.requested_budget_bytes // self.block_size_bytes) * self.block_size_bytes

    @property
    def total_blocks(self) -> int:
        return self.effective_budget_bytes // self.block_size_bytes

    @property
    def quantization_scale(self) -> float:
        return math.pow(10.0, self.quant_scale_power)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memory_budget_mib": self.memory_budget_mib,
            "block_size_kb": self.block_size_kb,
            "block_size_bytes": self.block_size_bytes,
            "precision": self.precision,
            "quant_scale_power": self.quant_scale_power,
            "quantization_scale": self.quantization_scale,
            "batch_size_limit": self.batch_size_limit,
            "preallocation": self.preallocation,
            "budget_rounding": self.budget_rounding,
            "requested_budget_bytes": self.requested_budget_bytes,
            "effective_budget_bytes": self.effective_budget_bytes,
            "total_blocks": self.total_blocks,
            "pressure_threshold_ms": self.pressure_threshold_ms,
            "ui_query_budget_ms": self.ui_query_budget_ms,
        }

    def version_fields(self) -> dict[str, Any]:
        return {
            "batch_size_limit": self.batch_size_limit,
            "precision": self.precision,
            "quant_scale_power": self.quant_scale_power,
            "memory_budget_mib": self.memory_budget_mib,
            "block_size_kb": self.block_size_kb,
            "preallocation": self.preallocation,
        }


def _fail(path: str, message: str, **details: Any) -> None:
    raise DiagnosticError(E_CONFIG, message, path, details)


def _require_int(data: dict[str, Any], key: str, *, minimum: int | None = None) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"$.{key}", "must be an integer", received=type(value).__name__)
    if minimum is not None and value < minimum:
        _fail(f"$.{key}", f"must be >= {minimum}", received=value)
    return value


def _require_number(data: dict[str, Any], key: str, *, positive: bool = False) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"$.{key}", "must be a number", received=type(value).__name__)
    number = float(value)
    if not math.isfinite(number):
        _fail(f"$.{key}", "must be finite", received=value)
    if positive and number <= 0:
        _fail(f"$.{key}", "must be > 0", received=value)
    if not positive and number < 0:
        _fail(f"$.{key}", "must be >= 0", received=value)
    return number


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"$.{key}", "duplicate JSON key", key=key)
        result[key] = value
    return result


def parse_config(data: Any, source: str = "<memory>") -> InferenceConfig:
    if not isinstance(data, dict):
        _fail("$", "configuration root must be an object", source=source)

    missing = sorted(REQUIRED_FIELDS - set(data))
    if missing:
        _fail(f"$.{missing[0]}", "required field is missing", missing=missing)
    unknown = sorted(set(data) - REQUIRED_FIELDS)
    if unknown:
        _fail(f"$.{unknown[0]}", "unknown field", unknown=unknown)

    schema_version = _require_int(data, "schema_version", minimum=1)
    if schema_version != 1:
        _fail("$.schema_version", "unsupported schema_version", supported=[1])

    budget_mib = _require_int(data, "memory_budget_mib", minimum=1)
    block_kb = _require_int(data, "block_size_kb", minimum=1)
    if block_kb not in SUPPORTED_BLOCK_KB:
        _fail(
            "$.block_size_kb",
            "unsupported block size",
            received=block_kb,
            supported=sorted(SUPPORTED_BLOCK_KB),
        )

    precision = data["precision"]
    if not isinstance(precision, str) or precision not in SUPPORTED_PRECISIONS:
        _fail("$.precision", "unsupported precision mode", supported=sorted(SUPPORTED_PRECISIONS))

    power = _require_int(data, "quant_scale_power")
    if power < -9 or power > 9:
        _fail("$.quant_scale_power", "must be in the safe integer power range [-9, 9]", received=power)

    batch_limit = _require_int(data, "batch_size_limit", minimum=1)

    preallocation = data["preallocation"]
    if not isinstance(preallocation, str) or preallocation not in SUPPORTED_PREALLOCATION:
        _fail(
            "$.preallocation",
            "unsupported preallocation strategy",
            supported=sorted(SUPPORTED_PREALLOCATION),
        )

    rounding = data["budget_rounding"]
    if not isinstance(rounding, str) or rounding not in SUPPORTED_ROUNDING:
        _fail("$.budget_rounding", "unsupported rounding rule", supported=sorted(SUPPORTED_ROUNDING))

    pressure_ms = _require_number(data, "pressure_threshold_ms")
    ui_budget_ms = _require_number(data, "ui_query_budget_ms", positive=True)

    budget_bytes = budget_mib * 1024 * 1024
    block_bytes = block_kb * 1024
    if budget_bytes % block_bytes != 0 and rounding != "floor_to_block":
        _fail(
            "$.memory_budget_mib",
            "budget is not an integer multiple of block size and budget_rounding is reject",
            requested_budget_bytes=budget_bytes,
            block_size_bytes=block_bytes,
            required_rule="floor_to_block",
        )
    if budget_bytes // block_bytes < 1:
        _fail(
            "$.memory_budget_mib",
            "budget contains fewer than one block",
            requested_budget_bytes=budget_bytes,
            block_size_bytes=block_bytes,
        )

    return InferenceConfig(
        schema_version=schema_version,
        memory_budget_mib=budget_mib,
        block_size_kb=block_kb,
        precision=precision,
        quant_scale_power=power,
        batch_size_limit=batch_limit,
        preallocation=preallocation,
        budget_rounding=rounding,
        pressure_threshold_ms=pressure_ms,
        ui_query_budget_ms=ui_budget_ms,
    )


def load_config(path: str | Path) -> InferenceConfig:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DiagnosticError(E_CONFIG, f"cannot read configuration: {exc}", "$", {"source": str(source)}) from exc
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicate_pairs)
    except DiagnosticError:
        raise
    except json.JSONDecodeError as exc:
        raise DiagnosticError(
            E_CONFIG,
            "invalid JSON",
            f"$[line {exc.lineno}, column {exc.colno}]",
            {"source": str(source)},
        ) from exc
    return parse_config(data, str(source))
