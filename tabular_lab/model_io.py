"""Model exchange versions v1/v2, their common representation, and sidecars."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import (
    E_CONFIG_INVALID,
    E_FIELD_TRUNCATED,
    E_MIX_RANGE,
    E_SCALE_RANGE,
    E_SCHEME_UNKNOWN,
    fail,
)
from .util import (
    atomic_write_json,
    atomic_write_text,
    milli_to_float,
    read_json,
    round_half_up_milli,
    safe_filename,
    sha256_text,
    short_hash,
    to_jsonable,
)


NAME_WIDTH = 20
FEATURE_WIDTH = 4
COEFFICIENT_WIDTH = 12
CODE_WIDTH = 2
INTERCEPT_WIDTH = 12
V1_LINE_WIDTH = NAME_WIDTH + FEATURE_WIDTH + COEFFICIENT_WIDTH + CODE_WIDTH + INTERCEPT_WIDTH
MILLI_SCALE_MAX = 999_999_999

SCHEME_ALIASES = {
    "l2": "l2",
    "ridge": "l2",
    "l1": "l1",
    "lasso": "l1",
    "elastic_net": "elastic_net",
    "elasticnet": "elastic_net",
    "enet": "elastic_net",
}
SCHEME_BASE = {"l2": 1, "l1": 2, "elastic_net": 3}
STATUS_UNIT = {"converged": 1, "max_epochs": 2, "budget_exhausted": 3, "interrupted": 4}

CODE_TO_NAMES = {
    SCHEME_BASE[scheme] * 10 + STATUS_UNIT[status]: (scheme, status)
    for scheme in SCHEME_BASE
    for status in STATUS_UNIT
}
LEGACY_CODE_TO_SCHEME = {1: "l2", 2: "l1", 3: "elastic_net"}
VALID_SCHEMES = set(SCHEME_BASE)
VALID_STATUSES = set(STATUS_UNIT)


@dataclass
class ModelRecord:
    name: str
    scheme: str
    coefficients: list[float]
    intercept: float
    feature_indices: list[int]
    convergence_status: str
    precision_label: str
    penalty: float = 0.0
    mix: float | None = None
    feature_names: list[str] | None = None
    penalty_code: int | None = None
    source_version: str = "internal"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            fail(E_CONFIG_INVALID, "model name is required")
        self.scheme = normalize_scheme(self.scheme)
        if self.convergence_status not in VALID_STATUSES:
            fail(
                E_CONFIG_INVALID,
                "unknown named convergence status",
                model=self.name,
                status=self.convergence_status,
            )
        if self.mix is not None and (
            not math.isfinite(float(self.mix)) or self.mix < 0.0 or self.mix > 1.0
        ):
            fail(E_MIX_RANGE, "mix must be in [0, 1]", model=self.name, value=self.mix)
        if self.penalty < 0.0 or not math.isfinite(self.penalty):
            fail(E_CONFIG_INVALID, "penalty must be finite and non-negative", model=self.name)
        if len(self.coefficients) != len(self.feature_indices):
            fail(
                E_CONFIG_INVALID,
                "coefficient and feature index counts differ",
                model=self.name,
                coefficients=len(self.coefficients),
                feature_indices=len(self.feature_indices),
            )
        if not self.coefficients:
            fail(E_CONFIG_INVALID, "a model must contain at least one coefficient", model=self.name)
        if any(index < 0 for index in self.feature_indices):
            fail(E_CONFIG_INVALID, "feature indices must be non-negative", model=self.name)
        if len(set(self.feature_indices)) != len(self.feature_indices):
            fail(E_CONFIG_INVALID, "feature indices must be unique", model=self.name)
        if any(not math.isfinite(float(value)) for value in self.coefficients):
            fail(E_CONFIG_INVALID, "coefficients must be finite", model=self.name)
        if not math.isfinite(float(self.intercept)):
            fail(E_CONFIG_INVALID, "intercept must be finite", model=self.name)
        if self.feature_names is not None and len(self.feature_names) != len(self.coefficients):
            fail(E_CONFIG_INVALID, "feature name count differs from coefficient count", model=self.name)

    def canonical(self, *, include_names: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "scheme": self.scheme,
            "coefficients": [float(value) for value in self.coefficients],
            "intercept": float(self.intercept),
            "feature_indices": [int(value) for value in self.feature_indices],
            "convergence_status": self.convergence_status,
            "precision_label": self.precision_label,
            "penalty": float(self.penalty),
            "mix": self.mix,
            "penalty_code": self.penalty_code,
            "source_version": self.source_version,
        }
        if include_names and self.feature_names is not None:
            payload["feature_names"] = list(self.feature_names)
        if self.metadata:
            payload["metadata"] = self.metadata
        return to_jsonable(payload)


def normalize_scheme(value: Any) -> str:
    key = str(value).strip().lower() if value is not None else ""
    if key not in SCHEME_ALIASES:
        fail(E_SCHEME_UNKNOWN, f"unknown regularization scheme: {value!r}", scheme=value)
    return SCHEME_ALIASES[key]


def penalty_code(scheme: str, status: str) -> int:
    scheme = normalize_scheme(scheme)
    if status not in STATUS_UNIT:
        fail(E_CONFIG_INVALID, f"unknown convergence status: {status!r}", status=status)
    return SCHEME_BASE[scheme] * 10 + STATUS_UNIT[status]


def decode_penalty_code(code: int) -> tuple[str, str]:
    if code in CODE_TO_NAMES:
        return CODE_TO_NAMES[code]
    if code in LEGACY_CODE_TO_SCHEME:
        return LEGACY_CODE_TO_SCHEME[code], "converged"
    fail(E_SCHEME_UNKNOWN, f"unknown integer penalty code: {code}", penalty_code=code)


def _validate_milli(value: int, field_name: str, model: str) -> int:
    if abs(value) > MILLI_SCALE_MAX:
        fail(
            E_SCALE_RANGE,
            f"{field_name} exceeds the supported milli-scale range",
            model=model,
            value=value,
            limit=MILLI_SCALE_MAX,
        )
    return value


def _parse_int_field(text: str, field_name: str, model: str | None = None) -> int:
    stripped = text.strip().lstrip("+")
    if not stripped:
        fail(E_FIELD_TRUNCATED, f"missing {field_name} field", model=model)
    try:
        return int(stripped, 10)
    except ValueError:
        fail(E_CONFIG_INVALID, f"invalid integer in {field_name} field", model=model, value=text)


def parse_v1_text(text: str) -> list[ModelRecord]:
    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if len(raw_line) < V1_LINE_WIDTH:
            fail(
                E_FIELD_TRUNCATED,
                "v1 line is shorter than the declared field widths",
                expected_width=V1_LINE_WIDTH,
                actual_width=len(raw_line),
                line=raw_line,
            )
        if raw_line[V1_LINE_WIDTH:].strip():
            fail(
                E_CONFIG_INVALID,
                "v1 line contains non-whitespace content after the final field",
                line=raw_line,
            )
        name = raw_line[0:NAME_WIDTH].strip()
        if not name:
            fail(E_CONFIG_INVALID, "v1 model-name field is empty", line=raw_line)
        feature_text = raw_line[NAME_WIDTH : NAME_WIDTH + FEATURE_WIDTH]
        coefficient_text = raw_line[
            NAME_WIDTH + FEATURE_WIDTH : NAME_WIDTH + FEATURE_WIDTH + COEFFICIENT_WIDTH
        ]
        code_text = raw_line[
            NAME_WIDTH
            + FEATURE_WIDTH
            + COEFFICIENT_WIDTH : NAME_WIDTH
            + FEATURE_WIDTH
            + COEFFICIENT_WIDTH
            + CODE_WIDTH
        ]
        intercept_text = raw_line[
            NAME_WIDTH + FEATURE_WIDTH + COEFFICIENT_WIDTH + CODE_WIDTH : V1_LINE_WIDTH
        ]
        feature_index = _parse_int_field(feature_text, "feature-index", name)
        coefficient_milli = _validate_milli(
            _parse_int_field(coefficient_text, "coefficient", name), "coefficient", name
        )
        code = _parse_int_field(code_text, "penalty-code", name)
        intercept_milli = _validate_milli(
            _parse_int_field(intercept_text, "intercept", name), "intercept", name
        )
        scheme, status = decode_penalty_code(code)
        if feature_index < 0:
            fail(E_CONFIG_INVALID, "v1 feature index must be non-negative", model=name)

        if name not in rows:
            rows[name] = {
                "coefficients": {},
                "intercept_milli": intercept_milli,
                "code": code,
                "scheme": scheme,
                "status": status,
            }
            order.append(name)
        model_row = rows[name]
        if model_row["intercept_milli"] != intercept_milli:
            fail(E_CONFIG_INVALID, "inconsistent intercept within one v1 model", model=name)
        if model_row["code"] != code:
            fail(E_CONFIG_INVALID, "inconsistent penalty code within one v1 model", model=name)
        existing = model_row["coefficients"].get(feature_index)
        if existing is not None and existing != coefficient_milli:
            fail(E_CONFIG_INVALID, "duplicate feature index has conflicting coefficients", model=name, feature_index=feature_index)
        model_row["coefficients"][feature_index] = coefficient_milli

    records: list[ModelRecord] = []
    for name in order:
        row = rows[name]
        coefficient_map: dict[int, int] = row["coefficients"]
        indices = sorted(coefficient_map)
        records.append(
            ModelRecord(
                name=name,
                scheme=row["scheme"],
                coefficients=[milli_to_float(coefficient_map[index]) for index in indices],
                intercept=milli_to_float(row["intercept_milli"]),
                feature_indices=indices,
                convergence_status=row["status"],
                precision_label="v1_milli_exact_to_float64",
                penalty_code=row["code"],
                source_version="v1",
                metadata={
                    "penalty_code": row["code"],
                    "coefficient_milli": [coefficient_map[index] for index in indices],
                    "intercept_milli": row["intercept_milli"],
                },
            )
        )
    if not records:
        fail(E_CONFIG_INVALID, "v1 model text contains no records")
    return records


def parse_v2_payload(payload: Any) -> list[ModelRecord]:
    if isinstance(payload, list):
        models = payload
    elif isinstance(payload, dict):
        models = payload.get("models", [payload])
    else:
        fail(E_CONFIG_INVALID, "v2 JSON root must be an object or array")
    if not isinstance(models, list) or not models:
        fail(E_CONFIG_INVALID, "v2 JSON must contain a non-empty models list")

    records: list[ModelRecord] = []
    seen: set[str] = set()
    for raw in models:
        if not isinstance(raw, dict):
            fail(E_CONFIG_INVALID, "each v2 model must be an object")
        name = str(raw.get("name", "")).strip()
        if not name:
            fail(E_CONFIG_INVALID, "v2 model name is required")
        if name in seen:
            fail(E_CONFIG_INVALID, "duplicate v2 model name", model=name)
        seen.add(name)
        scheme_value = raw.get("scheme", raw.get("regularization"))
        if scheme_value is None and raw.get("penalty_code") is not None:
            scheme_value, _ = decode_penalty_code(int(raw["penalty_code"]))
        scheme = normalize_scheme(scheme_value)
        coefficients_raw = raw.get("coefficients")
        if not isinstance(coefficients_raw, list):
            fail(E_CONFIG_INVALID, "v2 coefficients must be a list", model=name)
        coefficients: list[float] = []
        for value in coefficients_raw:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                fail(E_CONFIG_INVALID, "v2 coefficient must be numeric", model=name)
            float_value = float(value)
            if not math.isfinite(float_value):
                fail(E_CONFIG_INVALID, "v2 coefficient must be finite", model=name)
            coefficients.append(float_value)

        feature_indices_raw = raw.get("feature_indices")
        if feature_indices_raw is None:
            feature_indices_raw = list(range(len(coefficients)))
        if not isinstance(feature_indices_raw, list):
            fail(E_CONFIG_INVALID, "v2 feature_indices must be a list", model=name)
        feature_indices = []
        for value in feature_indices_raw:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                fail(E_CONFIG_INVALID, "v2 feature indices must be non-negative integers", model=name)
            feature_indices.append(value)

        mix_raw = raw.get("mix", raw.get("l1_ratio"))
        mix: float | None = None
        if mix_raw is not None:
            if isinstance(mix_raw, bool) or not isinstance(mix_raw, (int, float)):
                fail(E_CONFIG_INVALID, "v2 mix must be numeric", model=name)
            mix = float(mix_raw)
            if not math.isfinite(mix) or mix < 0.0 or mix > 1.0:
                fail(E_MIX_RANGE, "mix must be in [0, 1]", model=name, value=mix)
        penalty_raw = raw.get("penalty", raw.get("lambda", 0.0))
        if isinstance(penalty_raw, bool) or not isinstance(penalty_raw, (int, float)):
            fail(E_CONFIG_INVALID, "v2 penalty must be numeric", model=name)
        penalty = float(penalty_raw)
        if penalty < 0.0 or not math.isfinite(penalty):
            fail(E_CONFIG_INVALID, "v2 penalty must be finite and non-negative", model=name)

        status = str(raw.get("convergence_status", raw.get("convergence", "converged")))
        if status not in VALID_STATUSES:
            fail(E_CONFIG_INVALID, "unknown v2 convergence status", model=name, status=status)
        precision = str(raw.get("precision_label", raw.get("precision", "float64")))
        feature_names_raw = raw.get("feature_names")
        feature_names: list[str] | None = None
        if feature_names_raw is not None:
            if not isinstance(feature_names_raw, list):
                fail(E_CONFIG_INVALID, "v2 feature_names must be a list", model=name)
            feature_names = [str(value) for value in feature_names_raw]
        intercept_raw = raw.get("intercept", 0.0)
        if isinstance(intercept_raw, bool) or not isinstance(intercept_raw, (int, float)):
            fail(E_CONFIG_INVALID, "v2 intercept must be numeric", model=name)
        intercept = float(intercept_raw)
        if not math.isfinite(intercept):
            fail(E_CONFIG_INVALID, "v2 intercept must be finite", model=name)
        if feature_indices:
            canonical_order = sorted(range(len(feature_indices)), key=lambda index: feature_indices[index])
            coefficients = [coefficients[index] for index in canonical_order]
            feature_indices = [feature_indices[index] for index in canonical_order]
            if feature_names is not None:
                feature_names = [feature_names[index] for index in canonical_order]
        record = ModelRecord(
            name=name,
            scheme=scheme,
            coefficients=coefficients,
            intercept=intercept,
            feature_indices=feature_indices,
            convergence_status=status,
            precision_label=precision,
            penalty=penalty,
            mix=mix,
            feature_names=feature_names,
            penalty_code=penalty_code(scheme, status),
            source_version="v2",
            metadata={"source_precision_label": precision},
        )
        record.validate()
        records.append(record)
    return records


def _v1_line(record: ModelRecord, feature_index: int, coefficient: float) -> str:
    record.validate()
    if len(record.name) > NAME_WIDTH:
        fail(E_FIELD_TRUNCATED, "model name does not fit the v1 name field", model=record.name)
    if feature_index < 0 or feature_index >= 10**FEATURE_WIDTH:
        fail(E_FIELD_TRUNCATED, "feature index does not fit the v1 feature field", model=record.name)
    code = penalty_code(record.scheme, record.convergence_status)
    coefficient_milli = _validate_milli(
        round_half_up_milli(coefficient), "coefficient", record.name
    )
    intercept_milli = _validate_milli(
        round_half_up_milli(record.intercept), "intercept", record.name
    )
    if code < 0 or code >= 10**CODE_WIDTH:
        fail(E_FIELD_TRUNCATED, "penalty code does not fit the v1 code field", model=record.name)
    return (
        f"{record.name:<{NAME_WIDTH}}"
        f"{feature_index:>{FEATURE_WIDTH}d}"
        f"{coefficient_milli:>{COEFFICIENT_WIDTH}d}"
        f"{code:>{CODE_WIDTH}d}"
        f"{intercept_milli:>{INTERCEPT_WIDTH}d}"
    )


def records_to_v1_lines(records: Iterable[ModelRecord]) -> list[str]:
    lines: list[str] = []
    for record in records:
        record.validate()
        for feature_index, coefficient in zip(record.feature_indices, record.coefficients):
            lines.append(_v1_line(record, feature_index, coefficient))
    return lines


def record_to_v1_text(record: ModelRecord) -> str:
    return "\n".join(records_to_v1_lines([record])) + "\n"


def write_v1(records: Iterable[ModelRecord], path: str | Path) -> None:
    lines = records_to_v1_lines(records)
    if not lines:
        fail(E_CONFIG_INVALID, "cannot write an empty v1 model set")
    atomic_write_text(path, "\n".join(lines) + "\n")


def record_to_v2_payload(record: ModelRecord) -> dict[str, Any]:
    record.validate()
    payload: dict[str, Any] = {
        "name": record.name,
        "scheme": record.scheme,
        "coefficients": [float(value) for value in record.coefficients],
        "feature_indices": [int(value) for value in record.feature_indices],
        "intercept": float(record.intercept),
        "penalty": float(record.penalty),
        "mix": record.mix,
        "convergence_status": record.convergence_status,
        "precision_label": record.precision_label,
    }
    if record.feature_names is not None:
        payload["feature_names"] = list(record.feature_names)
    return to_jsonable(payload)


def records_to_v2_payload(records: Iterable[ModelRecord]) -> dict[str, Any]:
    return {
        "schema_version": "v2",
        "precision_policy": "IEEE-754 float64; precision_label is preserved per model",
        "models": [record_to_v2_payload(record) for record in records],
    }


def write_v2(records: Iterable[ModelRecord], path: str | Path) -> None:
    payload = records_to_v2_payload(records)
    if not payload["models"]:
        fail(E_CONFIG_INVALID, "cannot write an empty v2 model set")
    atomic_write_json(path, payload)


def _downgrade_sidecar(record: ModelRecord) -> dict[str, Any]:
    coefficient_milli: list[int] = []
    coefficient_residual: list[float] = []
    for coefficient in record.coefficients:
        milli = round_half_up_milli(coefficient)
        coefficient_milli.append(milli)
        coefficient_residual.append(float(coefficient) - milli_to_float(milli))
    intercept_milli = round_half_up_milli(record.intercept)
    intercept_residual = float(record.intercept) - milli_to_float(intercept_milli)
    return {
        "schema_version": "v1-downgrade-sidecar/1",
        "model": record.name,
        "source_version": "v2",
        "target_version": "v1",
        "precision_label": record.precision_label,
        "coefficients_float64": [float(value) for value in record.coefficients],
        "coefficients_milli": coefficient_milli,
        "coefficients_residual": coefficient_residual,
        "intercept_float64": float(record.intercept),
        "intercept_milli": intercept_milli,
        "intercept_residual": intercept_residual,
        "discarded_names": {
            "scheme": record.scheme,
            "convergence_status": record.convergence_status,
            "penalty": float(record.penalty),
            "mix": record.mix,
            "feature_names": list(record.feature_names) if record.feature_names else None,
        },
        "fidelity": {
            "max_abs_coefficient_residual": max((abs(value) for value in coefficient_residual), default=0.0),
            "abs_intercept_residual": abs(intercept_residual),
        },
    }


def sidecar_filename(model_name: str) -> str:
    return f"{safe_filename(model_name)}.{sha256_text(model_name)[:8]}.downgrade.json"


def write_downgrade_sidecar(record: ModelRecord, sidecar_dir: str | Path) -> Path:
    destination = Path(sidecar_dir) / sidecar_filename(record.name)
    atomic_write_json(destination, _downgrade_sidecar(record))
    return destination


def load_sidecars(sidecar_dir: str | Path | None) -> dict[str, dict[str, Any]]:
    if sidecar_dir is None:
        return {}
    directory = Path(sidecar_dir)
    if not directory.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.downgrade.json")):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("model"):
            result[str(payload["model"])] = payload
    return result


def attach_sidecar(record: ModelRecord, sidecar: dict[str, Any] | None) -> ModelRecord:
    if not sidecar:
        return record
    record.metadata["downgrade_sidecar"] = to_jsonable(sidecar)
    discarded = sidecar.get("discarded_names") or {}
    if isinstance(discarded, dict):
        if discarded.get("feature_names") is not None:
            names = [str(value) for value in discarded["feature_names"]]
            if len(names) == len(record.coefficients):
                record.feature_names = names
        if sidecar.get("precision_label"):
            record.metadata["original_precision_label"] = sidecar["precision_label"]
            record.precision_label = str(sidecar["precision_label"])
        if record.scheme == "elastic_net" and discarded.get("mix") is not None:
            try:
                mix = float(discarded["mix"])
            except (TypeError, ValueError):
                mix = None
            if mix is not None and 0.0 <= mix <= 1.0:
                record.mix = mix
        if discarded.get("penalty") is not None:
            try:
                penalty = float(discarded["penalty"])
            except (TypeError, ValueError):
                penalty = None
            if penalty is not None and math.isfinite(penalty) and penalty >= 0.0:
                record.penalty = penalty
    return record


def read_model_file(
    path: str | Path,
    *,
    sidecar_dir: str | Path | None = None,
    known_feature_names: Iterable[str] | None = None,
) -> list[ModelRecord]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(E_CONFIG_INVALID, f"model file not found: {source}")
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            fail(E_CONFIG_INVALID, f"invalid v2 JSON: {exc}", path=str(source))
        records = parse_v2_payload(payload)
        if known_feature_names is not None:
            validate_records_against_schema(records, list(known_feature_names))
        return records
    records = parse_v1_text(text)
    if sidecar_dir is None:
        for candidate_dir in (source.parent / "sidecars", source.parent):
            if candidate_dir.is_dir() and any(candidate_dir.glob("*.downgrade.json")):
                sidecar_dir = candidate_dir
                break
    sidecars = load_sidecars(sidecar_dir)
    if sidecars:
        for record in records:
            attach_sidecar(record, sidecars.get(record.name))
    if known_feature_names is not None:
        validate_records_against_schema(records, list(known_feature_names))
    return records


def validate_records_against_schema(
    records: Iterable[ModelRecord], known_feature_names: list[str]
) -> None:
    known = set(known_feature_names)
    for record in records:
        if record.feature_names is not None:
            unknown = sorted(set(record.feature_names) - known)
            if unknown:
                fail(
                    E_CONFIG_INVALID,
                    "model references unknown feature column(s)",
                    model=record.name,
                    unknown_columns=unknown,
                )
        if record.feature_indices and max(record.feature_indices) >= len(known_feature_names):
            fail(
                E_CONFIG_INVALID,
                "model feature index is outside the known feature schema",
                model=record.name,
                feature_index=max(record.feature_indices),
                feature_count=len(known_feature_names),
            )


def conversion_artifact(record: ModelRecord, target_version: str) -> dict[str, Any]:
    if target_version == "v1":
        sidecar = _downgrade_sidecar(record)
        text = record_to_v1_text(record)
        return {
            "version": "v1",
            "model": record.name,
            "text": text,
            "lines": text.splitlines(),
            "sidecar": sidecar,
            "payload_hash": short_hash("conv", {"text": text, "sidecar": sidecar}),
        }
    if target_version == "v2":
        payload = record_to_v2_payload(record)
        return {
            "version": "v2",
            "model": record.name,
            "payload": payload,
            "payload_hash": short_hash("conv", payload),
        }
    fail(E_CONFIG_INVALID, "target exchange version must be v1 or v2", version=target_version)


def compare_records(
    left: Iterable[ModelRecord], right: Iterable[ModelRecord]
) -> dict[str, Any]:
    left_by_name = {record.name: record for record in left}
    right_by_name = {record.name: record for record in right}
    if set(left_by_name) != set(right_by_name):
        return {
            "equal_names": False,
            "missing_left": sorted(set(right_by_name) - set(left_by_name)),
            "missing_right": sorted(set(left_by_name) - set(right_by_name)),
        }
    comparisons: list[dict[str, Any]] = []
    max_coefficient_delta = 0.0
    max_intercept_delta = 0.0
    for name in sorted(left_by_name):
        left_record = left_by_name[name]
        right_record = right_by_name[name]
        coefficient_delta = 0.0
        if len(left_record.coefficients) != len(right_record.coefficients):
            coefficient_delta = float("inf")
        else:
            coefficient_delta = max(
                (abs(float(a) - float(b)) for a, b in zip(left_record.coefficients, right_record.coefficients)),
                default=0.0,
            )
        intercept_delta = abs(float(left_record.intercept) - float(right_record.intercept))
        max_coefficient_delta = max(max_coefficient_delta, coefficient_delta)
        max_intercept_delta = max(max_intercept_delta, intercept_delta)
        comparisons.append(
            {
                "model": name,
                "max_abs_coefficient_delta": coefficient_delta,
                "abs_intercept_delta": intercept_delta,
                "scheme_equal": left_record.scheme == right_record.scheme,
                "status_equal": left_record.convergence_status == right_record.convergence_status,
            }
        )
    return {
        "equal_names": True,
        "max_abs_coefficient_delta": max_coefficient_delta,
        "max_abs_intercept_delta": max_intercept_delta,
        "models": comparisons,
    }
