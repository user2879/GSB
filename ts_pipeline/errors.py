"""Stable business error codes used by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_SAMPLE_INVALID = "E_SAMPLE_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SERIES_QUOTA_EXHAUSTED = "E_SERIES_QUOTA_EXHAUSTED"
E_POINT_BUDGET_EXHAUSTED = "E_POINT_BUDGET_EXHAUSTED"
E_WINDOW_BUILD_FAILED = "E_WINDOW_BUILD_FAILED"
E_MODEL_FIT_FAILED = "E_MODEL_FIT_FAILED"
E_PREDICTION_FAILED = "E_PREDICTION_FAILED"
E_METRIC_FAILED = "E_METRIC_FAILED"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"


@dataclass
class Failure:
    code: str
    stage: str
    message: str
    sequence_id: str | None = None
    window_index: int | None = None
    model_name: str | None = None
    injection_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def sort_key(self) -> tuple:
        return (
            self.sequence_id if self.sequence_id is not None else "",
            self.window_index if self.window_index is not None else -1,
            self.model_name if self.model_name is not None else "",
            self.stage,
            self.code,
            self.injection_id if self.injection_id is not None else "",
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "sequence_id": self.sequence_id,
            "window_index": self.window_index,
            "model_name": self.model_name,
            "injection_id": self.injection_id,
            "message": self.message,
            "detail": self.detail,
        }


def sort_failures(failures: list[Failure]) -> list[Failure]:
    return sorted(failures, key=lambda item: item.sort_key())


class PipelineFatalError(RuntimeError):
    """Raised when no meaningful processing can continue."""

    def __init__(self, code: str, message: str, *, stage: str = "bootstrap"):
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage

    def as_failure(self) -> Failure:
        return Failure(code=self.code, stage=self.stage, message=self.message)
