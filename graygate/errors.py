from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GatewayError(Exception):
    code: str
    field_path: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code} {self.field_path}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "field_path": self.field_path,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class SpecError(GatewayError):
    def __init__(self, field_path: str, message: str, details: dict[str, Any] | None = None):
        super().__init__("E_SPEC", field_path, message, details or {})


class SnapshotError(GatewayError):
    def __init__(
        self,
        field_path: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__("E_SNAPSHOT", field_path, message, details or {})


class ReplayError(GatewayError):
    def __init__(
        self,
        field_path: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__("E_REPLAY", field_path, message, details or {})


class LatencyError(GatewayError):
    def __init__(
        self,
        field_path: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__("E_LATENCY", field_path, message, details or {})


class RequestError(GatewayError):
    def __init__(
        self,
        code: str,
        field_path: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(code, field_path, message, details or {})

