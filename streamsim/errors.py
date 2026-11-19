"""Business errors returned by the simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    path: str | None = None
    details: dict | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_dict(self) -> dict:
        payload = {"code": self.code, "message": self.message}
        if self.path is not None:
            payload["path"] = self.path
        if self.details is not None:
            payload["details"] = self.details
        return payload


def error_response(error: BusinessError, **extra: object) -> dict:
    payload = {"ok": False, "error": error.as_dict()}
    payload.update(extra)
    return payload
