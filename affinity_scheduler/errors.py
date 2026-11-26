from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_SPEC = "E_SPEC"
E_FORMAT = "E_FORMAT"
E_WIDTH = "E_WIDTH"
E_PORT = "E_PORT"
E_TIMEOUT = "E_TIMEOUT"
E_REPLY = "E_REPLY"
E_MIGRATE = "E_MIGRATE"
E_SNAPSHOT = "E_SNAPSHOT"
E_VERIFY = "E_VERIFY"


@dataclass
class SchedulerError(Exception):
    """A structured business error with a stable machine-readable code."""

    code: str
    message: str
    field_path: str | None = None
    source_line: int | None = None
    source_path: str | None = None
    pod_id: str | None = None
    node_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        for key in (
            "field_path",
            "source_line",
            "source_path",
            "pod_id",
            "node_name",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.details:
            payload["details"] = self.details
        return payload


class SpecError(SchedulerError):
    def __init__(
        self,
        message: str,
        *,
        field_path: str,
        source_line: int | None = None,
        source_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            E_SPEC,
            message,
            field_path=field_path,
            source_line=source_line,
            source_path=source_path,
            details=details or {},
        )


class FormatError(SchedulerError):
    def __init__(
        self,
        message: str,
        *,
        field_path: str | None = None,
        source_line: int | None = None,
        source_path: str | None = None,
        pod_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            E_FORMAT,
            message,
            field_path=field_path,
            source_line=source_line,
            source_path=source_path,
            pod_id=pod_id,
            details=details or {},
        )


class WidthError(SchedulerError):
    def __init__(
        self,
        message: str,
        *,
        source_line: int | None = None,
        source_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            E_WIDTH,
            message,
            source_line=source_line,
            source_path=source_path,
            details=details or {},
        )


class PortError(SchedulerError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(E_PORT, message, details=details or {})


class TimeoutError(SchedulerError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(E_TIMEOUT, message, details=details or {})


class ReplyError(SchedulerError):
    def __init__(
        self,
        message: str,
        *,
        pod_id: str | None = None,
        node_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            E_REPLY,
            message,
            pod_id=pod_id,
            node_name=node_name,
            details=details or {},
        )


class MigrateError(SchedulerError):
    def __init__(
        self,
        message: str,
        *,
        pod_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(E_MIGRATE, message, pod_id=pod_id, details=details or {})


class SnapshotError(SchedulerError):
    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(E_SNAPSHOT, message, details=details or {})


class VerifyError(SchedulerError):
    def __init__(
        self,
        message: str,
        *,
        pod_id: str | None = None,
        node_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            E_VERIFY,
            message,
            pod_id=pod_id,
            node_name=node_name,
            details=details or {},
        )
