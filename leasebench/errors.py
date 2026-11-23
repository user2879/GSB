"""Domain errors with stable public error codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LeasebenchError(Exception):
    code: str
    field: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, f"{self.code}: {self.field}: {self.message}")

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "field": self.field,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


class DeclError(LeasebenchError):
    def __init__(self, field: str, message: str, **details: Any) -> None:
        super().__init__("E_DECL", field, message, dict(details))


class QuotaError(LeasebenchError):
    def __init__(self, field: str, message: str, **details: Any) -> None:
        super().__init__("E_QUOTA", field, message, dict(details))


class SnapshotError(LeasebenchError):
    def __init__(self, field: str, message: str, **details: Any) -> None:
        super().__init__("E_SNAPSHOT", field, message, dict(details))


class ReplayError(LeasebenchError):
    def __init__(
        self,
        field: str,
        message: str,
        *,
        event_seq: int | None = None,
        lease_id: str | None = None,
        **details: Any,
    ) -> None:
        payload = dict(details)
        if event_seq is not None:
            payload["event_seq"] = event_seq
        if lease_id is not None:
            payload["lease_id"] = lease_id
        super().__init__("E_REPLAY", field, message, payload)

