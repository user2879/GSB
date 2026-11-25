from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .errors import E_INDEX_INPUT, Failure, InputValidationError
from .model import FIELD_ORDER
from .util import read_json, sha256_json


@dataclass(frozen=True)
class FixedConfig:
    decimal_places: int
    quantum: int

    @property
    def scale(self) -> int:
        return 10**self.decimal_places

    def to_dict(self) -> dict[str, int]:
        return {"decimal_places": self.decimal_places, "quantum": self.quantum}


@dataclass(frozen=True)
class BM25Config:
    k1: float
    b: float

    def to_dict(self) -> dict[str, float]:
        return {"k1": self.k1, "b": self.b}


@dataclass(frozen=True)
class PrototypeConfig:
    fields: tuple[str, ...]
    field_weights: dict[str, float]
    bm25: BM25Config
    fixed: FixedConfig
    allowed_decimal_places: tuple[int, ...]
    allowed_quantums: tuple[int, ...]
    top_k: int
    max_workers: int
    batch_point_budget: int
    per_query_point_quota: int
    stage_tolerance_units: float
    schema_version: str = "search-proto-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fields": {
                name: {"weight": self.field_weights[name]} for name in self.fields
            },
            "bm25": self.bm25.to_dict(),
            "fixed": self.fixed.to_dict(),
            "allowed_decimal_places": list(self.allowed_decimal_places),
            "allowed_quantums": list(self.allowed_quantums),
            "top_k": self.top_k,
            "max_workers": self.max_workers,
            "batch_point_budget": self.batch_point_budget,
            "per_query_point_quota": self.per_query_point_quota,
            "stage_tolerance_units": self.stage_tolerance_units,
        }

    def semantic_dict(self) -> dict[str, Any]:
        value = self.to_dict()
        for key in (
            "max_workers",
            "batch_point_budget",
            "per_query_point_quota",
        ):
            value.pop(key, None)
        return value

    @property
    def semantic_fingerprint(self) -> str:
        return sha256_json(self.semantic_dict())

    @property
    def full_fingerprint(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def field_config_fingerprint(self) -> str:
        return sha256_json(
            {
                "fields": list(self.fields),
                "field_weights": {name: self.field_weights[name] for name in self.fields},
            }
        )

    @property
    def bm25_fingerprint(self) -> str:
        return sha256_json(self.bm25.to_dict())

    @property
    def fixed_fingerprint(self) -> str:
        return sha256_json(self.fixed.to_dict())

    def with_field_weights(self, weights: dict[str, float]) -> "PrototypeConfig":
        return replace(self, field_weights={name: weights[name] for name in self.fields})

    def with_bm25(self, k1: float, b: float) -> "PrototypeConfig":
        return replace(self, bm25=BM25Config(k1=k1, b=b))

    def with_fixed(self, decimal_places: int, quantum: int) -> "PrototypeConfig":
        return replace(
            self,
            fixed=FixedConfig(decimal_places=decimal_places, quantum=quantum),
        )

    def with_top_k(self, top_k: int) -> "PrototypeConfig":
        return replace(self, top_k=top_k)


def _failure(message: str, detail: str, context: dict[str, Any] | None = None) -> Failure:
    return Failure(
        code=E_INDEX_INPUT,
        source="config",
        line=0,
        entity_id="config",
        message=message,
        detail_code=detail,
        context=context,
    )


def validate_config(value: dict[str, Any]) -> PrototypeConfig:
    failures: list[Failure] = []

    fields_value = value.get("fields", value.get("field_weights"))
    if isinstance(fields_value, dict):
        field_names = tuple(name for name in FIELD_ORDER if name in fields_value)
        field_weights: dict[str, float] = {}
        for name, spec in fields_value.items():
            if name not in FIELD_ORDER:
                failures.append(_failure("字段名必须限定为 title/body", "FIELD_NAME", {"field": name}))
                continue
            weight = spec.get("weight") if isinstance(spec, dict) else spec
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) <= 0
            ):
                failures.append(
                    _failure("字段权重必须为有限正数", "FIELD_WEIGHT", {"field": name})
                )
                continue
            field_weights[name] = float(weight)
    else:
        field_names = ()
        field_weights = {}
        failures.append(_failure("缺少 fields/field_weights 映射", "FIELDS_MISSING"))

    if not field_names:
        failures.append(_failure("至少需要配置一个字段", "FIELDS_EMPTY"))

    bm25_value = value.get("bm25", {})
    k1 = bm25_value.get("k1")
    b = bm25_value.get("b")
    for name, item in (("k1", k1), ("b", b)):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0
        ):
            failures.append(
                _failure("词频饱和参数必须为有限正数", "BM25_PARAM", {"parameter": name})
            )

    fixed_value = value.get("fixed", {})
    decimal_places = fixed_value.get("decimal_places")
    quantum = fixed_value.get("quantum")
    allowed_decimal_places_raw = value.get("allowed_decimal_places", [2, 3, 4, 5])
    allowed_quantums_raw = value.get("allowed_quantums", [1, 2, 5, 10])

    if not isinstance(allowed_decimal_places_raw, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0
        for item in allowed_decimal_places_raw
    ):
        failures.append(
            _failure("allowed_decimal_places 必须为正整数枚举", "FIXED_ENUM")
        )
        allowed_decimal_places: tuple[int, ...] = ()
    else:
        allowed_decimal_places = tuple(sorted(set(allowed_decimal_places_raw)))

    if not isinstance(allowed_quantums_raw, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0
        for item in allowed_quantums_raw
    ):
        failures.append(_failure("allowed_quantums 必须为正整数枚举", "FIXED_ENUM"))
        allowed_quantums: tuple[int, ...] = ()
    else:
        allowed_quantums = tuple(sorted(set(allowed_quantums_raw)))

    if (
        not isinstance(decimal_places, int)
        or isinstance(decimal_places, bool)
        or decimal_places not in allowed_decimal_places
    ):
        failures.append(
            _failure(
                "定点小数位宽不在允许枚举内",
                "FIXED_DECIMAL_PLACES",
                {"value": decimal_places},
            )
        )
    if (
        not isinstance(quantum, int)
        or isinstance(quantum, bool)
        or quantum not in allowed_quantums
    ):
        failures.append(
            _failure(
                "量化档位不在允许枚举内",
                "FIXED_QUANTUM",
                {"value": quantum},
            )
        )

    def _positive_int(name: str, default: int | None = None) -> int:
        item = value.get(name, default)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            failures.append(_failure(f"{name} 必须为正整数", "POSITIVE_INT", {"field": name}))
            return 1
        return item

    top_k = _positive_int("top_k", 5)
    max_workers = _positive_int("max_workers", 1)
    batch_point_budget = _positive_int("batch_point_budget", 1000)
    per_query_point_quota = _positive_int("per_query_point_quota", 1000)
    tolerance = value.get("stage_tolerance_units", 0.5)
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0
    ):
        failures.append(
            _failure("stage_tolerance_units 必须为非负有限数", "TOLERANCE")
        )
        tolerance = 0.5

    if failures:
        raise InputValidationError(failures)

    return PrototypeConfig(
        fields=field_names,
        field_weights=field_weights,
        bm25=BM25Config(k1=float(k1), b=float(b)),
        fixed=FixedConfig(
            decimal_places=int(decimal_places), quantum=int(quantum)
        ),
        allowed_decimal_places=allowed_decimal_places,
        allowed_quantums=allowed_quantums,
        top_k=int(top_k),
        max_workers=int(max_workers),
        batch_point_budget=int(batch_point_budget),
        per_query_point_quota=int(per_query_point_quota),
        stage_tolerance_units=float(tolerance),
    )


def load_config(path: Path) -> PrototypeConfig:
    try:
        value = read_json(path)
    except (OSError, ValueError) as exc:
        raise InputValidationError(
            [_failure(f"无法读取配置: {exc}", "CONFIG_READ")]
        ) from exc
    if not isinstance(value, dict):
        raise InputValidationError([_failure("配置根节点必须为对象", "CONFIG_SHAPE")])
    return validate_config(value)

