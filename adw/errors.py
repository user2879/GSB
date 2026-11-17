"""Business errors and deterministic failure records."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Iterable


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_BUDGET_PENDING = "E_BUDGET_PENDING"
E_SAMPLE_STAGE = "E_SAMPLE_STAGE"
E_PARAM_INVALID = "E_PARAM_INVALID"
E_NON_FINITE_FEATURE = "E_NON_FINITE_FEATURE"
E_THRESHOLD_BOUNDARY = "E_THRESHOLD_BOUNDARY"
E_DATA_LIMIT = "E_DATA_LIMIT"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"

STAGE_ORDER = {
    "config": 0,
    "load": 1,
    "preprocess": 2,
    "score": 3,
    "decision": 4,
    "snapshot": 5,
}


@dataclass(frozen=True)
class Failure:
    code: str
    stage: str
    message: str
    field: str | None = None
    sample_id: str | None = None
    algorithm: str | None = None
    exception_type: str | None = None
    injection_id: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
            "sample_id": self.sample_id,
            "algorithm": self.algorithm,
            "exception_type": self.exception_type,
            "injection_id": self.injection_id,
        }
        if self.field is not None:
            out["field"] = self.field
        if self.details:
            out["details"] = self.details
        return out


def failure_sort_key(item: Failure | dict[str, Any]) -> tuple[Any, ...]:
    d = item.to_dict() if isinstance(item, Failure) else item
    return (
        STAGE_ORDER.get(str(d.get("stage") or ""), 99),
        str(d.get("sample_id") or ""),
        str(d.get("algorithm") or ""),
        str(d.get("code") or ""),
        str(d.get("injection_id") or ""),
        str(d.get("exception_type") or ""),
        str(d.get("message") or ""),
    )


def sort_failures(items: Iterable[Failure | dict[str, Any]]) -> list[dict[str, Any]]:
    converted = [x.to_dict() if isinstance(x, Failure) else x for x in items]
    return sorted(converted, key=failure_sort_key)


class ConfigValidationError(ValueError):
    def __init__(self, failures: list[Failure]):
        self.failures = sort_failures(failures)
        super().__init__("; ".join(f"{x['code']}: {x['message']}" for x in self.failures))


class SampleStageError(RuntimeError):
    """A deterministic, sample-scoped stage exception."""

    code = E_SAMPLE_STAGE
    stage = "unknown"

    def __init__(self, sample_id: str, message: str, *, injection_id: str | None = None):
        self.sample_id = sample_id
        self.injection_id = injection_id
        super().__init__(message)


class NonFiniteFeatureError(SampleStageError):
    code = E_NON_FINITE_FEATURE


class HyperparameterOutOfBoundsError(SampleStageError):
    code = E_PARAM_INVALID


class ThresholdBoundaryError(SampleStageError):
    code = E_THRESHOLD_BOUNDARY


class DataLimitError(RuntimeError):
    code = E_DATA_LIMIT


class SnapshotError(RuntimeError):
    code = E_SNAPSHOT_INVALID


def config_failure(field: str, message: str) -> Failure:
    return Failure(code=E_CONFIG_INVALID, stage="config", field=field, message=message)


def sample_failure(
    exc: BaseException,
    *,
    stage: str,
    sample_id: str,
    algorithm: str | None = None,
    injection_id: str | None = None,
    code: str | None = None,
) -> Failure:
    inferred = getattr(exc, "code", E_SAMPLE_STAGE)
    return Failure(
        code=code or inferred,
        stage=stage,
        sample_id=sample_id,
        algorithm=algorithm,
        exception_type=type(exc).__name__,
        injection_id=injection_id or getattr(exc, "injection_id", None),
        message=str(exc),
    )
