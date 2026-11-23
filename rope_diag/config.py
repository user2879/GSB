from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ErrorCode, Failure
from .util import is_finite_number, is_int, read_json, sha256_file, sha256_json


SCHEMES = ("rope", "linear_interpolation", "ntk_aware")
SCHEME_ALIASES = {
    "rope": "rope",
    "standard": "rope",
    "standard_rope": "rope",
    "linear": "linear_interpolation",
    "linear_interpolation": "linear_interpolation",
    "ntk": "ntk_aware",
    "ntk_aware": "ntk_aware",
}
PRECISION_ENUM = ("float64", "float32")
QUANTIZATION_ENUM = ("none", "int8_symmetric")


@dataclass(frozen=True)
class Tolerances:
    position_vector: float
    score: float
    attention: float


@dataclass(frozen=True)
class SensitivitySpec:
    base_step: float
    scale_step: float
    radius: int


@dataclass(frozen=True)
class DiagnosticConfig:
    global_version: int
    schemes: tuple[str, ...]
    base: float
    training_context_length: int
    extrapolation_target_length: int
    dimension: int
    scaling_factor: float
    precision: str
    quantization: str
    dimension_chunk_size: int
    point_budget: int
    concurrency: int
    snapshot_interval: int
    tolerances: Tolerances
    sensitivity: SensitivitySpec
    raw: dict[str, Any]
    source_path: Path
    config_hash: str

    @property
    def pair_count(self) -> int:
        return self.dimension // 2

    @property
    def extrapolation_ratio(self) -> float:
        return self.extrapolation_target_length / self.training_context_length

    def public_dict(self) -> dict[str, Any]:
        return {
            "global_version": self.global_version,
            "schemes": list(self.schemes),
            "base": self.base,
            "training_context_length": self.training_context_length,
            "extrapolation_target_length": self.extrapolation_target_length,
            "dimension": self.dimension,
            "scaling_factor": self.scaling_factor,
            "precision": self.precision,
            "quantization": self.quantization,
            "dimension_chunk_size": self.dimension_chunk_size,
            "point_budget": self.point_budget,
            "concurrency": self.concurrency,
            "snapshot_interval": self.snapshot_interval,
            "error_tolerances": {
                "position_vector": self.tolerances.position_vector,
                "score": self.tolerances.score,
                "attention": self.tolerances.attention,
            },
            "sensitivity": {
                "base_step": self.sensitivity.base_step,
                "scale_step": self.sensitivity.scale_step,
                "radius": self.sensitivity.radius,
            },
            "config_hash": self.config_hash,
        }


def _invalid(field: str, message: str, value: Any = None) -> Failure:
    details: dict[str, Any] = {"field": field}
    if value is not None:
        details["value"] = value
    return Failure(
        code=ErrorCode.CONFIG_INVALID,
        message=f"{field}: {message}",
        stage="config",
        details=details,
    )


def _canonical_scheme(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return SCHEME_ALIASES.get(value.strip().lower())


def validate_config_dict(
    raw: Any, *, source_path: Path, config_hash: str
) -> tuple[DiagnosticConfig | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(raw, dict):
        return None, [_invalid("config", "顶层必须是 JSON 对象")]

    try:
        version_value = raw.get("global_version", raw.get("run_version", raw.get("version")))
        if not is_int(version_value) or version_value <= 0:
            failures.append(_invalid("global_version", "必须为正整数", version_value))
            version_value = 1

        scheme_selection = raw.get("schemes", None)
        singular = raw.get("scheme", None)
        if scheme_selection is not None and singular is not None:
            failures.append(_invalid("schemes", "不得同时提供 schemes 与 scheme"))
        if scheme_selection is None and singular is not None:
            scheme_selection = [singular]
        if scheme_selection is None:
            scheme_selection = list(SCHEMES)
        if not isinstance(scheme_selection, list) or not scheme_selection:
            failures.append(_invalid("schemes", "必须为非空数组", scheme_selection))
            canonical_schemes: list[str] = []
        else:
            canonical_schemes = []
            for index, item in enumerate(scheme_selection):
                canonical = _canonical_scheme(item)
                if canonical is None:
                    failures.append(
                        _invalid(f"schemes[{index}]", f"必须属于 {list(SCHEMES)}", item)
                    )
                elif canonical not in canonical_schemes:
                    canonical_schemes.append(canonical)
            if len(canonical_schemes) != len(set(canonical_schemes)):
                failures.append(_invalid("schemes", "不得包含重复方案"))

        base = raw.get("base")
        if not is_finite_number(base) or float(base) <= 0.0:
            failures.append(_invalid("base", "必须为有限正数", base))

        training = raw.get("training_context_length")
        if not is_int(training) or training <= 0:
            failures.append(_invalid("training_context_length", "必须为正整数", training))

        target = raw.get("extrapolation_target_length")
        if not is_int(target) or target <= 0:
            failures.append(_invalid("extrapolation_target_length", "必须为正整数", target))
        if is_int(training) and is_int(target) and training >= target:
            failures.append(
                _invalid(
                    "extrapolation_target_length",
                    "必须严格大于 training_context_length",
                    target,
                )
            )

        dimension = raw.get("dimension")
        if not is_int(dimension) or dimension <= 0:
            failures.append(_invalid("dimension", "必须为正整数", dimension))
        elif dimension % 2 != 0:
            failures.append(_invalid("dimension", "必须为正偶数", dimension))

        scaling = raw.get("scaling_factor")
        if not is_finite_number(scaling) or float(scaling) <= 0.0:
            failures.append(_invalid("scaling_factor", "必须为有限正数", scaling))

        precision = raw.get("precision")
        if precision not in PRECISION_ENUM:
            failures.append(_invalid("precision", f"必须属于 {list(PRECISION_ENUM)}", precision))

        quantization = raw.get("quantization")
        if quantization not in QUANTIZATION_ENUM:
            failures.append(
                _invalid("quantization", f"必须属于 {list(QUANTIZATION_ENUM)}", quantization)
            )

        chunk_size = raw.get("dimension_chunk_size")
        if not is_int(chunk_size) or chunk_size <= 0:
            failures.append(_invalid("dimension_chunk_size", "必须为正整数", chunk_size))

        point_budget = raw.get("point_budget")
        if not is_int(point_budget) or point_budget <= 0:
            failures.append(_invalid("point_budget", "必须为正整数", point_budget))

        concurrency = raw.get("concurrency")
        if not is_int(concurrency) or concurrency <= 0:
            failures.append(_invalid("concurrency", "必须为正整数", concurrency))

        snapshot_interval = raw.get("snapshot_interval")
        if not is_int(snapshot_interval) or snapshot_interval <= 0:
            failures.append(_invalid("snapshot_interval", "必须为正整数", snapshot_interval))

        tolerance_raw = raw.get("error_tolerances")
        tolerance_values: dict[str, float] = {}
        if not isinstance(tolerance_raw, dict):
            failures.append(_invalid("error_tolerances", "必须为对象", tolerance_raw))
        else:
            for name in ("position_vector", "score", "attention"):
                value = tolerance_raw.get(name)
                if not is_finite_number(value) or float(value) <= 0.0:
                    failures.append(
                        _invalid(f"error_tolerances.{name}", "必须为有限正数", value)
                    )
                else:
                    tolerance_values[name] = float(value)

        sensitivity_raw = raw.get("sensitivity")
        sensitivity_values: dict[str, Any] = {}
        if not isinstance(sensitivity_raw, dict):
            failures.append(_invalid("sensitivity", "必须为对象", sensitivity_raw))
        else:
            for name in ("base_step", "scale_step"):
                value = sensitivity_raw.get(name)
                if not is_finite_number(value) or float(value) <= 0.0:
                    failures.append(
                        _invalid(f"sensitivity.{name}", "必须为有限正数", value)
                    )
                else:
                    sensitivity_values[name] = float(value)
            radius = sensitivity_raw.get("radius")
            if not is_int(radius) or radius < 0:
                failures.append(_invalid("sensitivity.radius", "必须为非负整数", radius))
            else:
                sensitivity_values["radius"] = radius

        if failures:
            return None, failures

        config = DiagnosticConfig(
            global_version=int(version_value),
            schemes=tuple(canonical_schemes),
            base=float(base),
            training_context_length=int(training),
            extrapolation_target_length=int(target),
            dimension=int(dimension),
            scaling_factor=float(scaling),
            precision=str(precision),
            quantization=str(quantization),
            dimension_chunk_size=int(chunk_size),
            point_budget=int(point_budget),
            concurrency=int(concurrency),
            snapshot_interval=int(snapshot_interval),
            tolerances=Tolerances(
                position_vector=tolerance_values["position_vector"],
                score=tolerance_values["score"],
                attention=tolerance_values["attention"],
            ),
            sensitivity=SensitivitySpec(
                base_step=sensitivity_values["base_step"],
                scale_step=sensitivity_values["scale_step"],
                radius=int(sensitivity_values["radius"]),
            ),
            raw=raw,
            source_path=source_path,
            config_hash=config_hash,
        )
        return config, []
    except (TypeError, ValueError, OverflowError) as exc:
        failures.append(_invalid("config", f"校验异常: {exc}"))
        return None, failures


def load_config(path: Path) -> tuple[DiagnosticConfig | None, list[Failure], str]:
    try:
        file_hash = sha256_file(path)
        raw = read_json(path)
    except (OSError, ValueError) as exc:
        failure = Failure(
            code=ErrorCode.IO_ERROR,
            message=f"无法读取配置: {exc}",
            stage="config",
            details={"path": path.as_posix()},
        )
        return None, [failure], ""
    config_hash = sha256_json(
        {"config_content": raw, "config_file_sha256": file_hash, "schema": "rope-diag-config-v1"}
    )
    config, failures = validate_config_dict(raw, source_path=path, config_hash=config_hash)
    return config, failures, config_hash
