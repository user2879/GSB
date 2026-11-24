"""Business errors and failure records used by the quantization pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_WEIGHT_PACK_INVALID = "E_WEIGHT_PACK_INVALID"
E_CALIBRATION_INVALID = "E_CALIBRATION_INVALID"
E_LAYER_CALIBRATION = "E_LAYER_CALIBRATION"
E_LAYER_QUANTIZATION = "E_LAYER_QUANTIZATION"
E_LAYER_DISTILLATION = "E_LAYER_DISTILLATION"
E_SATURATION_EXCEEDED = "E_SATURATION_EXCEEDED"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_POINT_QUOTA_EXCEEDED = "E_POINT_QUOTA_EXCEEDED"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_INJECTION_INVALID = "E_INJECTION_INVALID"
E_RUN_INTERRUPTED = "E_RUN_INTERRUPTED"
E_RUNTIME = "E_RUNTIME"


class PipelineError(Exception):
    """An expected, reportable pipeline error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str | None = None,
        layer: str | None = None,
        exception_type: str | None = None,
        injection_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.layer = layer
        self.exception_type = exception_type or type(self).__name__
        self.injection_id = injection_id
        self.details = dict(details or {})


class ConfigError(PipelineError):
    pass


class WeightPackError(PipelineError):
    pass


class CalibrationError(PipelineError):
    pass


@dataclass
class FailureRecord:
    code: str
    message: str
    stage: str | None = None
    layer: str | None = None
    exception_type: str = "PipelineError"
    injection_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "business_code": self.code,
            "stage": self.stage,
            "layer": self.layer,
            "exception_type": self.exception_type,
            "message": self.message,
        }
        if self.injection_id is not None:
            result["injection_id"] = self.injection_id
        if self.details:
            result["details"] = self.details
        return result

    @classmethod
    def from_exception(cls, exc: PipelineError) -> "FailureRecord":
        return cls(
            code=exc.code,
            message=exc.message,
            stage=exc.stage,
            layer=exc.layer,
            exception_type=exc.exception_type,
            injection_id=exc.injection_id,
            details=exc.details,
        )


def failure(
    code: str,
    message: str,
    *,
    stage: str | None = None,
    layer: str | None = None,
    exception_type: str = "ValidationError",
    injection_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> FailureRecord:
    return FailureRecord(
        code=code,
        message=message,
        stage=stage,
        layer=layer,
        exception_type=exception_type,
        injection_id=injection_id,
        details=dict(details or {}),
    )

