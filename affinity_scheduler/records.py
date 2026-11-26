from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import FormatError, WidthError
from .model import DEFAULT_TIE_BREAK, PLUGIN_NAMES, VETO_CODES, Spec
from .util import canonical_json, digest_value


V1_PREFIX = "v1"
V1_FIELD_WIDTHS: tuple[tuple[str, int], ...] = (
    ("pod_id", 18),
    ("node_name", 18),
    ("total_score", 6),
    ("veto_code", 4),
)
V1_WIDTH = len(V1_PREFIX) + sum(width for _, width in V1_FIELD_WIDTHS)
SUPPORTED_FORMAT_VERSIONS = frozenset({1, 2, 3})
CURRENT_FORMAT_VERSION = 3
SUPPORTED_VETO_CODES = frozenset(VETO_CODES)

V2_REQUIRED_FIELDS = frozenset(
    {
        "format_version",
        "pod_id",
        "node_name",
        "plugin_scores",
        "total_score",
        "veto_code",
        "tie_break",
    }
)
V3_REQUIRED_FIELDS = frozenset(
    {
        "format_version",
        "pod_id",
        "node_name",
        "plugin_scores",
        "plugin_contributions",
        "total_score",
        "veto_code",
        "tie_break",
        "parameter_version",
        "error_bound",
    }
)


@dataclass(frozen=True)
class NormalizationParams:
    parameter_version: int = 0
    scale_exponent: int = 0
    weights_myriad: Mapping[str, int] = field(
        default_factory=lambda: {plugin: 2500 for plugin in PLUGIN_NAMES}
    )

    @property
    def scale_factor(self) -> int:
        return 10**self.scale_exponent

    @classmethod
    def from_spec(cls, spec: Spec) -> "NormalizationParams":
        return cls(
            parameter_version=spec.parameter_version,
            scale_exponent=spec.scale_exponent,
            weights_myriad=dict(spec.weights_myriad),
        )


@dataclass(frozen=True)
class NormalizedRecord:
    source_format_version: int
    source_line: int
    pod_id: str
    node_name: str
    total_score: int
    veto_code: int
    plugin_scores: Mapping[str, int]
    plugin_contributions: Mapping[str, int]
    plugin_weights_myriad: Mapping[str, int]
    parameter_version: int
    scale_exponent: int
    scale_factor: int
    error_bound: int
    tie_break: tuple[str, ...]
    filled_fields: tuple[str, ...]
    completion_basis: Mapping[str, Mapping[str, str]]
    source_total_score: float | int

    def to_v3_dict(self) -> dict[str, Any]:
        return {
            "format_version": 3,
            "pod_id": self.pod_id,
            "node_name": self.node_name,
            "plugin_scores": dict(self.plugin_scores),
            "plugin_weights_myriad": dict(self.plugin_weights_myriad),
            "plugin_contributions": dict(self.plugin_contributions),
            "total_score": self.total_score,
            "veto_code": self.veto_code,
            "tie_break": list(self.tie_break),
            "parameter_version": self.parameter_version,
            "scale_exponent": self.scale_exponent,
            "scale_factor": self.scale_factor,
            "error_bound": self.error_bound,
            "normalization": {
                "source_format_version": self.source_format_version,
                "filled_fields": list(self.filled_fields),
                "basis": {
                    key: dict(value)
                    for key, value in sorted(self.completion_basis.items())
                },
            },
        }

    def to_v2_dict(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "pod_id": self.pod_id,
            "node_name": self.node_name,
            "plugin_scores": {
                name: float(self.plugin_scores[name]) for name in PLUGIN_NAMES
            },
            "total_score": float(self.total_score),
            "veto_code": self.veto_code,
            "tie_break": list(self.tie_break),
        }

    def conclusion(self, *, include_parameter: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "pod_id": self.pod_id,
            "node_name": self.node_name,
            "total_score": self.total_score,
            "veto_code": self.veto_code,
            "plugin_scores": {name: self.plugin_scores[name] for name in PLUGIN_NAMES},
            "plugin_contributions": {
                name: self.plugin_contributions[name] for name in PLUGIN_NAMES
            },
            "error_bound": self.error_bound,
            "tie_break": list(self.tie_break),
        }
        if include_parameter:
            value["parameter_version"] = self.parameter_version
            value["scale_exponent"] = self.scale_exponent
            value["scale_factor"] = self.scale_factor
        return value

    def conclusion_bytes(self, *, include_parameter: bool = True) -> bytes:
        return canonical_json(
            self.conclusion(include_parameter=include_parameter)
        ).encode("utf-8")

    @property
    def conclusion_digest(self) -> str:
        return digest_value(self.conclusion())


@dataclass(frozen=True)
class RecordBatch:
    source_path: str
    records: tuple[NormalizedRecord, ...]
    format_versions: tuple[int, ...]

    @property
    def version_set(self) -> list[str]:
        return [str(version) for version in self.format_versions]


def _half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _require_plain_int(
    value: Any,
    field_name: str,
    *,
    source_line: int,
    source_path: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormatError(
            f"{field_name} must be an integer",
            field_path=field_name,
            source_line=source_line,
            source_path=source_path,
        )
    if minimum is not None and value < minimum:
        raise FormatError(
            f"{field_name} must be >= {minimum}",
            field_path=field_name,
            source_line=source_line,
            source_path=source_path,
        )
    if maximum is not None and value > maximum:
        raise FormatError(
            f"{field_name} must be <= {maximum}",
            field_path=field_name,
            source_line=source_line,
            source_path=source_path,
        )
    return value


def _require_number(
    value: Any,
    field_name: str,
    *,
    source_line: int,
    source_path: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormatError(
            f"{field_name} must be numeric",
            field_path=field_name,
            source_line=source_line,
            source_path=source_path,
        )
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise FormatError(
            f"{field_name} out of range [{minimum}, {maximum}]",
            field_path=field_name,
            source_line=source_line,
            source_path=source_path,
        )
    return result


def _validate_common(
    data: Mapping[str, Any],
    *,
    source_line: int,
    source_path: str,
) -> tuple[str, str, int]:
    pod_id = data.get("pod_id")
    node_name = data.get("node_name")
    if not isinstance(pod_id, str) or not pod_id.strip():
        raise FormatError(
            "pod_id must be a non-empty string",
            field_path="pod_id",
            source_line=source_line,
            source_path=source_path,
        )
    if not isinstance(node_name, str) or not node_name.strip():
        raise FormatError(
            "node_name must be a non-empty string",
            field_path="node_name",
            source_line=source_line,
            source_path=source_path,
        )
    veto_code = _require_plain_int(
        data.get("veto_code"),
        "veto_code",
        source_line=source_line,
        source_path=source_path,
    )
    if veto_code not in SUPPORTED_VETO_CODES:
        raise FormatError(
            f"unsupported veto_code {veto_code}",
            field_path="veto_code",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        )
    return pod_id, node_name, veto_code


def _parse_plugin_scores(
    value: Any,
    *,
    integer_only: bool,
    source_line: int,
    source_path: str,
    pod_id: str,
) -> dict[str, float | int]:
    if not isinstance(value, dict):
        raise FormatError(
            "plugin_scores must be an object",
            field_path="plugin_scores",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        )
    missing = sorted(set(PLUGIN_NAMES) - set(value))
    if missing:
        raise FormatError(
            f"plugin_scores missing fields: {', '.join(missing)}",
            field_path=f"plugin_scores.{missing[0]}",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        )
    extra = sorted(set(value) - set(PLUGIN_NAMES))
    if extra:
        raise FormatError(
            f"unknown plugin score field {extra[0]}",
            field_path=f"plugin_scores.{extra[0]}",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        )
    result: dict[str, float | int] = {}
    for name in PLUGIN_NAMES:
        if integer_only:
            result[name] = _require_plain_int(
                value[name],
                f"plugin_scores.{name}",
                source_line=source_line,
                source_path=source_path,
                minimum=0,
                maximum=100,
            )
        else:
            result[name] = _require_number(
                value[name],
                f"plugin_scores.{name}",
                source_line=source_line,
                source_path=source_path,
                minimum=0.0,
                maximum=100.0,
            )
    return result


def _parse_contributions(
    value: Any,
    *,
    source_line: int,
    source_path: str,
    pod_id: str,
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise FormatError(
            "plugin_contributions must be an object",
            field_path="plugin_contributions",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        )
    missing = sorted(set(PLUGIN_NAMES) - set(value))
    if missing:
        raise FormatError(
            f"plugin_contributions missing fields: {', '.join(missing)}",
            field_path=f"plugin_contributions.{missing[0]}",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        )
    extra = sorted(set(value) - set(PLUGIN_NAMES))
    if extra:
        raise FormatError(
            f"unknown plugin contribution field {extra[0]}",
            field_path=f"plugin_contributions.{extra[0]}",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        )
    return {
        name: _require_plain_int(
            value[name],
            f"plugin_contributions.{name}",
            source_line=source_line,
            source_path=source_path,
            minimum=0,
        )
        for name in PLUGIN_NAMES
    }


def _parse_tie_break(
    value: Any,
    *,
    source_line: int,
    source_path: str,
    pod_id: str,
) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        parts = tuple(value)
    else:
        raise FormatError(
            "tie_break must be a comma string or array of names",
            field_path="tie_break",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        )
    if not parts:
        raise FormatError(
            "tie_break must not be empty",
            field_path="tie_break",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        )
    return parts


def _contribution(score: int, weight: int, scale_factor: int) -> int:
    numerator = score * weight * scale_factor
    if numerator > 2**63 - 1:
        numerator = 2**63 - 1
    return numerator // 10000


def _filled_entry(rule: str, source: str) -> dict[str, str]:
    return {"rule": rule, "source": source}


def _normalize_v1(
    line: str,
    *,
    source_line: int,
    source_path: str,
    params: NormalizationParams,
) -> NormalizedRecord:
    if len(line) < V1_WIDTH:
        raise FormatError(
            f"v1 fixed-width line is too short: expected {V1_WIDTH}, got {len(line)}",
            field_path="<fixed-width>",
            source_line=source_line,
            source_path=source_path,
            details={"expected_width": V1_WIDTH, "actual_width": len(line)},
        )
    if len(line) > V1_WIDTH:
        raise WidthError(
            f"v1 fixed-width declaration mismatch: expected {V1_WIDTH}, got {len(line)}",
            source_line=source_line,
            source_path=source_path,
            details={"expected_width": V1_WIDTH, "actual_width": len(line)},
        )
    cursor = len(V1_PREFIX)
    fields: dict[str, str] = {}
    for name, width in V1_FIELD_WIDTHS:
        fields[name] = line[cursor : cursor + width]
        cursor += width
    pod_id = fields["pod_id"].strip()
    node_name = fields["node_name"].strip()
    if not pod_id or not node_name:
        raise FormatError(
            "v1 pod_id and node_name must be non-empty",
            field_path="pod_id" if not pod_id else "node_name",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id or None,
        )
    try:
        total_score = int(fields["total_score"].strip())
        veto_code = int(fields["veto_code"].strip())
    except ValueError as exc:
        raise FormatError(
            f"v1 integer field is invalid: {exc}",
            field_path="total_score/veto_code",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        ) from exc
    if total_score < 0 or total_score > 100:
        raise FormatError(
            f"total_score out of range [0, 100]: {total_score}",
            field_path="total_score",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        )
    if veto_code not in SUPPORTED_VETO_CODES:
        raise FormatError(
            f"unsupported veto_code {veto_code}",
            field_path="veto_code",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        )
    basis: dict[str, Mapping[str, str]] = {}
    for name in PLUGIN_NAMES:
        basis[f"plugin_scores.{name}"] = _filled_entry(
            "legacy_missing_plugin_default_zero", "v1"
        )
        basis[f"plugin_contributions.{name}"] = _filled_entry(
            "legacy_missing_contribution_default_zero", "v1"
        )
        basis[f"plugin_weights_myriad.{name}"] = _filled_entry(
            "normalization_parameter", "declared_spec"
        )
    basis["parameter_version"] = _filled_entry(
        "normalization_parameter", "declared_spec"
    )
    basis["scale_exponent"] = _filled_entry(
        "normalization_parameter", "declared_spec"
    )
    basis["scale_factor"] = _filled_entry(
        "normalization_parameter", "declared_spec"
    )
    basis["error_bound"] = _filled_entry(
        "legacy_integer_total_exact", "v1"
    )
    basis["tie_break"] = _filled_entry(
        "legacy_default_tie_break", "v1"
    )
    return NormalizedRecord(
        source_format_version=1,
        source_line=source_line,
        pod_id=pod_id,
        node_name=node_name,
        total_score=total_score,
        veto_code=veto_code,
        plugin_scores={name: 0 for name in PLUGIN_NAMES},
        plugin_contributions={name: 0 for name in PLUGIN_NAMES},
        plugin_weights_myriad={name: params.weights_myriad[name] for name in PLUGIN_NAMES},
        parameter_version=params.parameter_version,
        scale_exponent=params.scale_exponent,
        scale_factor=params.scale_factor,
        error_bound=0,
        tie_break=("total_score", "node_name"),
        filled_fields=tuple(sorted(basis)),
        completion_basis=basis,
        source_total_score=total_score,
    )


def _normalize_v2(
    data: Mapping[str, Any],
    *,
    source_line: int,
    source_path: str,
    params: NormalizationParams,
) -> NormalizedRecord:
    missing = sorted(V2_REQUIRED_FIELDS - set(data))
    if missing:
        raise FormatError(
            f"v2 required field missing: {missing[0]}",
            field_path=missing[0],
            source_line=source_line,
            source_path=source_path,
        )
    pod_id, node_name, veto_code = _validate_common(
        data, source_line=source_line, source_path=source_path
    )
    raw_scores = _parse_plugin_scores(
        data["plugin_scores"],
        integer_only=False,
        source_line=source_line,
        source_path=source_path,
        pod_id=pod_id,
    )
    total_float = _require_number(
        data["total_score"],
        "total_score",
        source_line=source_line,
        source_path=source_path,
        minimum=0.0,
        maximum=100.0,
    )
    tie_break = _parse_tie_break(
        data["tie_break"],
        source_line=source_line,
        source_path=source_path,
        pod_id=pod_id,
    )
    scores = {name: _half_up(float(raw_scores[name])) for name in PLUGIN_NAMES}
    contributions = {
        name: _contribution(scores[name], params.weights_myriad[name], params.scale_factor)
        for name in PLUGIN_NAMES
    }
    basis: dict[str, Mapping[str, str]] = {}
    for name in PLUGIN_NAMES:
        basis[f"plugin_scores.{name}"] = _filled_entry(
            "quantize_half_up_to_integer", "v2_float_plugin_score"
        )
        basis[f"plugin_contributions.{name}"] = _filled_entry(
            "integer_fixed_point_from_score_weight_scale", "v2_missing_contribution"
        )
        basis[f"plugin_weights_myriad.{name}"] = _filled_entry(
            "normalization_parameter", "declared_spec"
        )
    basis["parameter_version"] = _filled_entry(
        "normalization_parameter", "declared_spec"
    )
    basis["scale_exponent"] = _filled_entry(
        "normalization_parameter", "declared_spec"
    )
    basis["scale_factor"] = _filled_entry(
        "normalization_parameter", "declared_spec"
    )
    basis["error_bound"] = _filled_entry(
        "conservative_float_to_integer_bound", "v2_float_total_and_plugin_scores"
    )
    error_bound = 1 + (len(PLUGIN_NAMES) + params.scale_factor - 1) // params.scale_factor
    return NormalizedRecord(
        source_format_version=2,
        source_line=source_line,
        pod_id=pod_id,
        node_name=node_name,
        total_score=_half_up(total_float),
        veto_code=veto_code,
        plugin_scores=scores,
        plugin_contributions=contributions,
        plugin_weights_myriad={name: params.weights_myriad[name] for name in PLUGIN_NAMES},
        parameter_version=params.parameter_version,
        scale_exponent=params.scale_exponent,
        scale_factor=params.scale_factor,
        error_bound=error_bound,
        tie_break=tie_break,
        filled_fields=tuple(sorted(basis)),
        completion_basis=basis,
        source_total_score=total_float,
    )


def _normalize_v3(
    data: Mapping[str, Any],
    *,
    source_line: int,
    source_path: str,
    params: NormalizationParams,
) -> NormalizedRecord:
    missing = sorted(V3_REQUIRED_FIELDS - set(data))
    if missing:
        raise FormatError(
            f"v3 required field missing: {missing[0]}",
            field_path=missing[0],
            source_line=source_line,
            source_path=source_path,
        )
    pod_id, node_name, veto_code = _validate_common(
        data, source_line=source_line, source_path=source_path
    )
    scores_raw = _parse_plugin_scores(
        data["plugin_scores"],
        integer_only=True,
        source_line=source_line,
        source_path=source_path,
        pod_id=pod_id,
    )
    scores = {name: int(scores_raw[name]) for name in PLUGIN_NAMES}
    contributions = _parse_contributions(
        data["plugin_contributions"],
        source_line=source_line,
        source_path=source_path,
        pod_id=pod_id,
    )
    total_score = _require_plain_int(
        data["total_score"],
        "total_score",
        source_line=source_line,
        source_path=source_path,
        minimum=0,
        maximum=100,
    )
    parameter_version = _require_plain_int(
        data["parameter_version"],
        "parameter_version",
        source_line=source_line,
        source_path=source_path,
        minimum=1,
    )
    error_bound = _require_plain_int(
        data["error_bound"],
        "error_bound",
        source_line=source_line,
        source_path=source_path,
        minimum=0,
        maximum=100,
    )
    tie_break = _parse_tie_break(
        data["tie_break"],
        source_line=source_line,
        source_path=source_path,
        pod_id=pod_id,
    )
    if "scale_exponent" in data:
        scale_exponent = _require_plain_int(
            data["scale_exponent"],
            "scale_exponent",
            source_line=source_line,
            source_path=source_path,
            minimum=0,
            maximum=3,
        )
    elif "scale_factor" in data:
        scale_factor_raw = _require_plain_int(
            data["scale_factor"],
            "scale_factor",
            source_line=source_line,
            source_path=source_path,
        )
        if scale_factor_raw not in {1, 10, 100, 1000}:
            raise FormatError(
                "scale_factor must be 10**n for n in [0, 3]",
                field_path="scale_factor",
                source_line=source_line,
                source_path=source_path,
                pod_id=pod_id,
            )
        scale_exponent = {1: 0, 10: 1, 100: 2, 1000: 3}[scale_factor_raw]
    else:
        raise FormatError(
            "v3 required field missing: scale_exponent/scale_factor",
            field_path="scale_factor",
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
        )
    scale_factor = 10**scale_exponent
    if "scale_factor" in data:
        declared_scale = _require_plain_int(
            data["scale_factor"],
            "scale_factor",
            source_line=source_line,
            source_path=source_path,
        )
        if declared_scale != scale_factor:
            raise FormatError(
                "scale_factor disagrees with scale_exponent",
                field_path="scale_factor",
                source_line=source_line,
                source_path=source_path,
                pod_id=pod_id,
            )
    basis: dict[str, Mapping[str, str]] = {}
    if "plugin_weights_myriad" in data:
        raw_weights = data["plugin_weights_myriad"]
        if not isinstance(raw_weights, dict) or set(raw_weights) != set(PLUGIN_NAMES):
            raise FormatError(
                "plugin_weights_myriad must contain exactly four plugins",
                field_path="plugin_weights_myriad",
                source_line=source_line,
                source_path=source_path,
                pod_id=pod_id,
            )
        weights = {
            name: _require_plain_int(
                raw_weights[name],
                f"plugin_weights_myriad.{name}",
                source_line=source_line,
                source_path=source_path,
                minimum=0,
                maximum=10000,
            )
            for name in PLUGIN_NAMES
        }
        if sum(weights.values()) != 10000:
            raise FormatError(
                "plugin_weights_myriad must sum to 10000",
                field_path="plugin_weights_myriad",
                source_line=source_line,
                source_path=source_path,
                pod_id=pod_id,
            )
    else:
        weights = {name: params.weights_myriad[name] for name in PLUGIN_NAMES}
        basis["plugin_weights_myriad"] = _filled_entry(
            "normalization_parameter", "declared_spec"
        )
    return NormalizedRecord(
        source_format_version=3,
        source_line=source_line,
        pod_id=pod_id,
        node_name=node_name,
        total_score=total_score,
        veto_code=veto_code,
        plugin_scores=scores,
        plugin_contributions=contributions,
        plugin_weights_myriad=weights,
        parameter_version=parameter_version,
        scale_exponent=scale_exponent,
        scale_factor=scale_factor,
        error_bound=error_bound,
        tie_break=tie_break,
        filled_fields=tuple(sorted(basis)),
        completion_basis=basis,
        source_total_score=total_score,
    )


def _decode_json_record(
    data: Any,
    *,
    source_line: int,
    source_path: str,
    params: NormalizationParams,
) -> NormalizedRecord:
    if not isinstance(data, dict):
        raise FormatError(
            "record JSON must be an object",
            field_path="$",
            source_line=source_line,
            source_path=source_path,
        )
    version = data.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise FormatError(
            "format_version must be an integer",
            field_path="format_version",
            source_line=source_line,
            source_path=source_path,
        )
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise FormatError(
            f"unknown format_version {version}",
            field_path="format_version",
            source_line=source_line,
            source_path=source_path,
        )
    if version == 2:
        return _normalize_v2(
            data, source_line=source_line, source_path=source_path, params=params
        )
    if version == 3:
        return _normalize_v3(
            data, source_line=source_line, source_path=source_path, params=params
        )
    raise FormatError(
        "format_version 1 must use fixed-width text, not JSON",
        field_path="format_version",
        source_line=source_line,
        source_path=source_path,
    )


def detect_format(line: str, *, source_line: int, source_path: str) -> int:
    stripped = line.strip()
    if stripped.startswith(V1_PREFIX):
        return 1
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise FormatError(
                f"invalid record JSON: {exc.msg}",
                field_path="$",
                source_line=source_line,
                source_path=source_path,
            ) from exc
        version = data.get("format_version") if isinstance(data, dict) else None
        if isinstance(version, int) and not isinstance(version, bool):
            return version
        raise FormatError(
            "JSON record is missing integer format_version",
            field_path="format_version",
            source_line=source_line,
            source_path=source_path,
        )
    raise FormatError(
        "unknown scheduling record format",
        field_path="$",
        source_line=source_line,
        source_path=source_path,
    )


def decode_line(
    line: str,
    *,
    source_line: int,
    source_path: str,
    params: NormalizationParams | None = None,
) -> NormalizedRecord:
    params = params or NormalizationParams()
    version = detect_format(line, source_line=source_line, source_path=source_path)
    if version == 1:
        if not line.strip().startswith(V1_PREFIX):
            raise FormatError(
                "format_version 1 must use fixed-width text, not JSON",
                field_path="format_version",
                source_line=source_line,
                source_path=source_path,
            )
        return _normalize_v1(
            line,
            source_line=source_line,
            source_path=source_path,
            params=params,
        )
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise FormatError(
            f"unknown format_version {version}",
            field_path="format_version",
            source_line=source_line,
            source_path=source_path,
        )
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise FormatError(
            f"invalid record JSON: {exc.msg}",
            field_path="$",
            source_line=source_line,
            source_path=source_path,
        ) from exc
    return _decode_json_record(
        data,
        source_line=source_line,
        source_path=source_path,
        params=params,
    )


def load_records(
    path: str | Path,
    *,
    params: NormalizationParams | None = None,
) -> RecordBatch:
    source_path = str(path)
    text = Path(path).read_text(encoding="utf-8")
    params = params or NormalizationParams()
    records: list[NormalizedRecord] = []
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FormatError(
                f"invalid records array: {exc.msg}",
                field_path="$",
                source_line=exc.lineno,
                source_path=source_path,
            ) from exc
        if not isinstance(data, list):
            raise FormatError(
                "top-level records JSON must be an array",
                field_path="$",
                source_line=1,
                source_path=source_path,
            )
        for index, item in enumerate(data):
            records.append(
                _decode_json_record(
                    item,
                    source_line=1,
                    source_path=source_path,
                    params=params,
                )
            )
    elif stripped.startswith("{"):
        try:
            single = json.loads(text)
        except json.JSONDecodeError:
            single = None
        if isinstance(single, dict) and "format_version" in single:
            records.append(
                _decode_json_record(
                    single,
                    source_line=1,
                    source_path=source_path,
                    params=params,
                )
            )
        else:
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                records.append(
                    decode_line(
                        line,
                        source_line=line_number,
                        source_path=source_path,
                        params=params,
                    )
                )
    else:
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            records.append(
                decode_line(
                    line,
                    source_line=line_number,
                    source_path=source_path,
                    params=params,
                )
            )
    if not records:
        raise FormatError(
            "records file is empty",
            field_path="$",
            source_line=1,
            source_path=source_path,
        )
    versions = tuple(sorted({record.source_format_version for record in records}))
    return RecordBatch(
        source_path=source_path,
        records=tuple(records),
        format_versions=versions,
    )


def encode_v1(record: NormalizedRecord) -> str:
    def fixed(value: str, width: int, field_name: str) -> str:
        if len(value) > width:
            raise FormatError(
                f"{field_name} exceeds v1 width {width}",
                field_path=field_name,
            )
        return value.ljust(width)

    return (
        V1_PREFIX
        + fixed(record.pod_id, 18, "pod_id")
        + fixed(record.node_name, 18, "node_name")
        + str(record.total_score).rjust(6)
        + str(record.veto_code).rjust(4)
    )


def encode_v2(record: NormalizedRecord) -> str:
    return canonical_json(record.to_v2_dict())


def encode_v3(record: NormalizedRecord) -> str:
    return canonical_json(record.to_v3_dict())


def encode_current(record: NormalizedRecord) -> str:
    return encode_v3(record)


def encode_records(
    records: Iterable[NormalizedRecord], *, version: int = CURRENT_FORMAT_VERSION
) -> str:
    encoders = {1: encode_v1, 2: encode_v2, 3: encode_v3}
    if version not in encoders:
        raise FormatError(f"unsupported output format version {version}")
    encoder = encoders[version]
    return "".join(encoder(record) + "\n" for record in records)
