from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_TEMPLATE_MISSING = "E_TEMPLATE_MISSING"
E_STAGE_FAILED = "E_STAGE_FAILED"


STAGE_ORDER = {
    "config": 0,
    "load": 1,
    "split": 2,
    "score": 3,
    "select": 4,
    "rewrite": 5,
    "report": 6,
}


class WorkbenchError(Exception):
    """A diagnosed failure that can be converted into a failure record."""

    def __init__(
        self,
        code: str,
        stage: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.details = details or {}


@dataclass
class FailureRecord:
    document_id: str | None
    stage: str
    code: str
    exception_type: str
    message: str
    sort_key: str
    injection_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def ordering_key(self) -> tuple[int, str, str, str, str]:
        return (
            STAGE_ORDER.get(self.stage, 99),
            self.document_id or "",
            self.sort_key,
            self.code,
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "stage": self.stage,
            "code": self.code,
            "exception_type": self.exception_type,
            "message": self.message,
            "sort_key": self.sort_key,
            "injection_ids": sorted(set(self.injection_ids)),
            "details": self.details,
        }


def failure_from_error(
    error: BaseException,
    *,
    document_id: str | None,
    stage: str,
    sort_key: str,
    injection_ids: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> FailureRecord:
    if isinstance(error, WorkbenchError):
        code = error.code
        message = str(error)
        merged_details = dict(error.details)
        if details:
            merged_details.update(details)
    else:
        code = E_STAGE_FAILED
        message = str(error) or error.__class__.__name__
        merged_details = dict(details or {})
    return FailureRecord(
        document_id=document_id,
        stage=stage,
        code=code,
        exception_type=error.__class__.__name__,
        message=message,
        sort_key=sort_key,
        injection_ids=list(injection_ids or []),
        details=merged_details,
    )
