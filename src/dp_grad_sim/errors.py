from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class SimError(Exception):
    """A business or runtime error with a stable machine-readable code."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "exception_type": type(self).__name__,
            "details": self.details,
        }


@dataclass
class FailureRecord:
    code: str
    window: int | None
    worker_id: int | None
    tensor: str | None
    exception_type: str
    message: str
    link: str | None = None
    sent_bytes: int | None = None
    requested_bytes: int | None = None
    budget_bytes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "window": self.window,
            "worker_id": self.worker_id,
            "tensor": self.tensor,
            "exception_type": self.exception_type,
            "message": self.message,
        }
        if self.link is not None:
            payload["link"] = self.link
        if self.sent_bytes is not None:
            payload["sent_bytes"] = self.sent_bytes
        if self.requested_bytes is not None:
            payload["requested_bytes"] = self.requested_bytes
        if self.budget_bytes is not None:
            payload["budget_bytes"] = self.budget_bytes
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (
            -1 if self.window is None else self.window,
            -1 if self.worker_id is None else self.worker_id,
            "" if self.tensor is None else self.tensor,
            self.code,
            "" if self.link is None else self.link,
            self.exception_type,
            self.message,
        )


def failure_from_error(
    error: BaseException,
    *,
    window: int | None = None,
    worker_id: int | None = None,
    tensor: str | None = None,
    code: str = "E_COMM_SEND_FAILED",
    **extra: Any,
) -> FailureRecord:
    sim_error = error if isinstance(error, SimError) else None
    return FailureRecord(
        code=sim_error.code if sim_error else code,
        window=window,
        worker_id=worker_id,
        tensor=tensor,
        exception_type=type(error).__name__,
        message=str(error),
        link=extra.pop("link", None),
        sent_bytes=extra.pop("sent_bytes", None),
        requested_bytes=extra.pop("requested_bytes", None),
        budget_bytes=extra.pop("budget_bytes", None),
        extra=extra,
    )

