"""Business errors and deterministic failure records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STAGE_ORDER = {
    "config": 0,
    "load": 1,
    "feature": 2,
    "score": 3,
    "decision": 4,
    "metric": 5,
    "state": 6,
}


@dataclass(frozen=True)
class Failure:
    sample: str
    stage: str
    code: str
    exception_type: str
    message: str
    tier: str | None = None
    detail: dict[str, Any] | None = None

    def sort_key(self) -> tuple:
        return (
            self.sample,
            STAGE_ORDER.get(self.stage, 99),
            self.tier or "",
            self.code,
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sample": self.sample,
            "stage": self.stage,
            "code": self.code,
            "exception_type": self.exception_type,
            "message": self.message,
        }
        if self.tier is not None:
            result["tier"] = self.tier
        if self.detail:
            result["detail"] = self.detail
        return result


class PipelineAbort(Exception):
    """A run-level error that prevents further pipeline execution."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str = "config",
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.message = message
        self.detail = detail or {}


class ConfigValidationError(PipelineAbort):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(
            "E_CONFIG_INVALID",
            "configuration validation failed: " + "; ".join(errors),
            stage="config",
            detail={"errors": errors},
        )


class PlanValidationError(ConfigValidationError):
    pass


class InjectionValidationError(ConfigValidationError):
    pass


class SnapshotMismatchError(PipelineAbort):
    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(
            "E_CONFIG_MISMATCH",
            message,
            stage="state",
            detail=detail,
        )


class SampleStageError(Exception):
    """A failure confined to one sample and one stage."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}


class FeatureValidationError(SampleStageError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__("E_FEATURE_INVALID", message, detail=detail)


class ImageDecodeError(SampleStageError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__("E_SAMPLE_INVALID", message, detail=detail)
