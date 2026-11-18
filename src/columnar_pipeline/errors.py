from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BUSINESS_CODES = {
    "E_DATA",
    "E_SKEW",
    "E_BUDGET",
    "E_DIGEST",
    "E_REPLAY",
}

EXIT_CODES = {
    "E_DATA": 10,
    "E_SKEW": 20,
    "E_BUDGET": 30,
    "E_DIGEST": 40,
    "E_REPLAY": 50,
}


@dataclass
class PipelineError(Exception):
    code: str
    message: str
    location: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    field_name: str | None = None

    def __post_init__(self) -> None:
        if self.code not in BUSINESS_CODES:
            raise ValueError(f"unknown business code: {self.code}")
        Exception.__init__(self, self.message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.location is not None:
            payload["location"] = self.location
        if self.task_id is not None:
            payload["task_id"] = self.task_id
        if self.field_name is not None:
            payload["field"] = self.field_name
        if self.details:
            payload["details"] = self.details
        return payload

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.code]


def data_error(
    message: str,
    location: str,
    *,
    details: dict[str, Any] | None = None,
) -> PipelineError:
    return PipelineError("E_DATA", message, location=location, details=details or {})


def skew_error(
    message: str,
    *,
    task_id: str | None = None,
    location: str | None = None,
    details: dict[str, Any] | None = None,
) -> PipelineError:
    return PipelineError(
        "E_SKEW",
        message,
        location=location,
        details=details or {},
        task_id=task_id,
    )


def budget_error(
    message: str,
    *,
    task_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> PipelineError:
    return PipelineError("E_BUDGET", message, details=details or {}, task_id=task_id)


def digest_error(
    message: str,
    *,
    task_id: str | None = None,
    location: str | None = None,
    details: dict[str, Any] | None = None,
) -> PipelineError:
    return PipelineError(
        "E_DIGEST",
        message,
        location=location,
        details=details or {},
        task_id=task_id,
    )


def replay_error(
    message: str,
    *,
    task_id: str,
    field_name: str,
    details: dict[str, Any] | None = None,
) -> PipelineError:
    return PipelineError(
        "E_REPLAY",
        message,
        details=details or {},
        task_id=task_id,
        field_name=field_name,
    )
