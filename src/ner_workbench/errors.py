from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E_CORPUS_INVALID = "E_CORPUS_INVALID"
E_LABEL_LENGTH_MISMATCH = "E_LABEL_LENGTH_MISMATCH"
E_LABEL_UNKNOWN = "E_LABEL_UNKNOWN"
E_PARAM_SHAPE_MISMATCH = "E_PARAM_SHAPE_MISMATCH"
E_SAMPLE_QUOTA_EXCEEDED = "E_SAMPLE_QUOTA_EXCEEDED"
E_GLOBAL_BUDGET_EXHAUSTED = "E_GLOBAL_BUDGET_EXHAUSTED"
E_NONFINITE_SCORE = "E_NONFINITE_SCORE"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_MAX_ITERATIONS_EXCEEDED = "E_MAX_ITERATIONS_EXCEEDED"
E_INTERNAL_ERROR = "E_INTERNAL_ERROR"


STAGE_ORDER = {
    "load": 0,
    "schedule": 1,
    "emission": 2,
    "decode": 3,
    "merge": 4,
    "snapshot": 5,
}


@dataclass
class WorkbenchError(Exception):
    code: str
    stage: str
    detail: str
    exception_type: str | None = None
    context: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__init__(self.detail)
        if self.exception_type is None:
            self.exception_type = type(self).__name__

    def as_failure(self, sample_id: str, injection_id: str | None = None) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "stage": self.stage,
            "code": self.code,
            "exception_type": self.exception_type,
            "message": self.detail,
            "injection_id": injection_id,
        }


def failure_record(
    sample_id: str,
    stage: str,
    code: str,
    exception_type: str,
    message: str,
    injection_id: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "stage": stage,
        "code": code,
        "exception_type": exception_type,
        "message": message,
        "injection_id": injection_id,
    }


def failure_sort_key(record: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(record.get("sample_id", "")),
        STAGE_ORDER.get(str(record.get("stage", "")), 99),
        str(record.get("code", "")),
        str(record.get("injection_id") or ""),
    )
