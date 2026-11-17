"""Fault-injection parsing and real input mutation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import canonical_scheme
from .errors import (
    ConfigError,
    E_BASE_ZERO,
    E_NONFINITE_INPUT,
    E_POSITION_OUT_OF_RANGE,
    E_SCALE_INVALID,
)
from .samples import Sample
from .util import hash_object, read_json


FAULT_ALIASES = {
    "nonfinitefeature": "nonfinite_feature",
    "nonfiniteinput": "nonfinite_feature",
    "nanfeature": "nonfinite_feature",
    "positionoutofrange": "position_out_of_range",
    "keypositionoutofrange": "position_out_of_range",
    "basezero": "base_zero",
    "scalenonpositive": "scale_nonpositive",
    "scaleinvalid": "scale_nonpositive",
    "nonpositivescale": "scale_nonpositive",
}

FAULT_CODES = {
    "nonfinite_feature": E_NONFINITE_INPUT,
    "position_out_of_range": E_POSITION_OUT_OF_RANGE,
    "base_zero": E_BASE_ZERO,
    "scale_nonpositive": E_SCALE_INVALID,
}

FAULT_DEFAULT_STAGE = {
    "nonfinite_feature": "encoding",
    "position_out_of_range": "scoring",
    "base_zero": "encoding",
    "scale_nonpositive": "encoding",
}


def _canonical_fault(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError("injection fault must be a string", {"value": value})
    key = "".join(value.strip().lower().split()).replace("_", "").replace("-", "")
    try:
        return FAULT_ALIASES[key]
    except KeyError as exc:
        raise ConfigError(
            "unsupported injection fault",
            {"value": value, "supported": sorted(FAULT_CODES)},
        ) from exc


def _nonfinite_value(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"nan", "+nan", "-nan"}:
            return float("nan")
        if normalized in {"inf", "+inf", "+infinity", "infinity"}:
            return float("inf")
        if normalized in {"-inf", "-infinity"}:
            return float("-inf")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ConfigError("non-finite injection value must be nan/inf or numeric", {"value": value})


@dataclass(frozen=True)
class Injection:
    injection_id: str
    injection_index: int
    sample_id: str
    stage: str
    fault: str
    scheme: str | None
    raw: dict[str, Any]

    def applies_to(self, scheme: str) -> bool:
        return self.scheme is None or self.scheme == scheme

    @property
    def code(self) -> str:
        return FAULT_CODES[self.fault]


def parse_injections(raw: Any) -> list[Injection]:
    records = raw.get("injections") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ConfigError("injection manifest must contain an injections list")
    result: list[Injection] = []
    seen: set[str] = set()
    for index, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            raise ConfigError("injection item must be an object", {"injection_index": index})
        injection_id = item.get("injection_id", item.get("id"))
        if not isinstance(injection_id, str) or not injection_id:
            raise ConfigError(
                "injection requires a non-empty injection_id",
                {"injection_index": index},
            )
        if injection_id in seen:
            raise ConfigError("duplicate injection_id", {"injection_id": injection_id})
        seen.add(injection_id)
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ConfigError(
                "injection requires a non-empty sample_id",
                {"injection_id": injection_id},
            )
        fault = _canonical_fault(item.get("fault", item.get("type")))
        stage = item.get("stage", FAULT_DEFAULT_STAGE[fault])
        if not isinstance(stage, str) or not stage:
            raise ConfigError(
                "injection stage must be a non-empty string",
                {"injection_id": injection_id},
            )
        scheme_raw = item.get("scheme")
        scheme = canonical_scheme(scheme_raw) if scheme_raw is not None else None
        result.append(
            Injection(
                injection_id=injection_id,
                injection_index=index,
                sample_id=sample_id,
                stage=stage,
                fault=fault,
                scheme=scheme,
                raw=copy.deepcopy(item),
            )
        )
    result.sort(key=lambda item: item.injection_id)
    return result


def load_injections(path: str | Path) -> tuple[list[Injection], dict[str, Any], str]:
    raw = read_json(Path(path))
    injections = parse_injections(raw)
    return injections, raw, hash_object(raw)


def _target_kind(injection: Injection, default: str = "query") -> str:
    raw = injection.raw
    target = raw.get("target", raw.get("feature_target", raw.get("location", default)))
    if not isinstance(target, str):
        raise ConfigError(
            "injection target must be a string",
            {"injection_id": injection.injection_id, "value": target},
        )
    normalized = target.strip().lower().replace("-", "_")
    if normalized in {"query", "query_features", "query_feature"}:
        return "query"
    if normalized in {"key", "keys", "key_features", "key_feature"}:
        return "key"
    if normalized in {"key_positions", "key_position", "position"}:
        return "key_position"
    if normalized in {"query_position", "query"}:
        return "query_position"
    raise ConfigError(
        "unsupported injection target",
        {"injection_id": injection.injection_id, "target": target},
    )


def _index(raw: dict[str, Any], injection_id: str, default: int = 0) -> int:
    value = raw.get("index", raw.get("key_index", default))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(
            "injection index must be a non-negative integer",
            {"injection_id": injection_id, "value": value},
        )
    return value


def _defect(injection: Injection, message: str) -> dict[str, Any]:
    return {
        "injection_id": injection.injection_id,
        "injection_index": injection.injection_index,
        "fault": injection.fault,
        "code": injection.code,
        "stage": injection.stage,
        "message": message,
    }


def apply_injections(
    sample: Sample,
    scheme: str,
    base: float,
    scaling_factor: float,
    injections: list[Injection],
    target_length: int,
) -> tuple[Sample, float, float, list[dict[str, Any]]]:
    """Apply every matching injection to a copy of the real input."""

    effective = sample.copy()
    effective_base = float(base)
    effective_scale = float(scaling_factor)
    defects: list[dict[str, Any]] = []
    for injection in sorted(injections, key=lambda item: item.injection_id):
        if injection.sample_id != sample.sample_id or not injection.applies_to(scheme):
            continue
        if injection.fault == "nonfinite_feature":
            kind = _target_kind(injection)
            index = _index(injection.raw, injection.injection_id)
            value = _nonfinite_value(injection.raw.get("value"))
            if not math.isfinite(value):
                if kind == "query":
                    if index >= len(effective.query_features):
                        raise ConfigError(
                            "non-finite injection query index is out of range",
                            {"injection_id": injection.injection_id, "index": index},
                        )
                    effective.query_features[index] = value
                elif kind == "key":
                    if index >= len(effective.key_features):
                        raise ConfigError(
                            "non-finite injection key index is out of range",
                            {"injection_id": injection.injection_id, "index": index},
                        )
                    feature_index = injection.raw.get("feature_index", 0)
                    if (
                        isinstance(feature_index, bool)
                        or not isinstance(feature_index, int)
                        or feature_index < 0
                        or feature_index >= len(effective.key_features[index])
                    ):
                        raise ConfigError(
                            "non-finite injection feature_index is out of range",
                            {
                                "injection_id": injection.injection_id,
                                "feature_index": feature_index,
                            },
                        )
                    effective.key_features[index][feature_index] = value
                else:
                    raise ConfigError(
                        "nonfinite_feature injection requires query or key target",
                        {"injection_id": injection.injection_id, "target": kind},
                    )
            else:
                raise ConfigError(
                    "nonfinite_feature injection must use a non-finite value",
                    {"injection_id": injection.injection_id, "value": value},
                )
            defects.append(_defect(injection, "injected non-finite feature value"))
        elif injection.fault == "position_out_of_range":
            kind = _target_kind(injection, default="key_position")
            position = injection.raw.get("position", injection.raw.get("value", target_length))
            if isinstance(position, bool) or not isinstance(position, int):
                raise ConfigError(
                    "position injection requires an integer position",
                    {"injection_id": injection.injection_id, "value": position},
                )
            if kind in {"key_position", "key"}:
                index = _index(injection.raw, injection.injection_id)
                if index >= len(effective.key_positions):
                    raise ConfigError(
                        "position injection key index is out of range",
                        {"injection_id": injection.injection_id, "index": index},
                    )
                effective.key_positions[index] = position
            elif kind == "query_position":
                effective.query_position = position
            else:
                raise ConfigError(
                    "position_out_of_range injection requires a key/query position target",
                    {"injection_id": injection.injection_id, "target": kind},
                )
            defects.append(_defect(injection, "injected position outside configured target range"))
        elif injection.fault == "base_zero":
            effective_base = 0.0
            defects.append(_defect(injection, "injected base=0"))
        elif injection.fault == "scale_nonpositive":
            value = injection.raw.get("value", injection.raw.get("scaling_factor", 0.0))
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(
                    "scale_nonpositive injection requires a numeric value",
                    {"injection_id": injection.injection_id, "value": value},
                )
            effective_scale = float(value)
            if effective_scale > 0.0 or not math.isfinite(effective_scale):
                raise ConfigError(
                    "scale_nonpositive injection value must be finite and <= 0",
                    {"injection_id": injection.injection_id, "value": value},
                )
            defects.append(_defect(injection, "injected scaling factor <= 0"))
    return effective, effective_base, effective_scale, defects
