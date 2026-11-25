"""Structured errors used at CLI and protocol boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


@dataclass
class EdgeError(Exception):
    code: str
    message: str
    line: int | None = None
    field: str | None = None
    event_seq: int | None = None
    node_id: str | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.line is not None:
            result["line"] = self.line
        if self.field is not None:
            result["field"] = self.field
        if self.event_seq is not None:
            result["event_seq"] = self.event_seq
        if self.node_id is not None:
            result["node_id"] = self.node_id
        if self.details:
            result.update(self.details)
        return result


class SpecError(EdgeError):
    def __init__(
        self,
        message: str,
        *,
        line: int | None = None,
        field: str | None = None,
        path: str | None = None,
    ) -> None:
        details = {"path": path} if path else {}
        super().__init__("E_SPEC", message, line=line, field=field, details=details)


class LeaseError(EdgeError):
    def __init__(self, message: str, *, event_seq: int | None = None, node_id: str | None = None) -> None:
        super().__init__("E_LEASE", message, event_seq=event_seq, node_id=node_id)


class PortError(EdgeError):
    def __init__(self, message: str, *, node_id: str | None = None) -> None:
        super().__init__("E_PORT", message, node_id=node_id)


class HandshakeTimeout(EdgeError):
    def __init__(self, message: str, *, node_id: str | None = None) -> None:
        super().__init__("E_TIMEOUT", message, node_id=node_id)


class NodeError(EdgeError):
    def __init__(self, message: str, *, node_id: str | None = None, code: str = "E_NODE") -> None:
        super().__init__(code, message, node_id=node_id)


class ReplayError(EdgeError):
    def __init__(
        self,
        message: str,
        *,
        event_seq: int | None = None,
        node_id: str | None = None,
        field: str | None = None,
    ) -> None:
        super().__init__("E_REPLAY", message, event_seq=event_seq, node_id=node_id, field=field)


class UiUnavailable(EdgeError):
    def __init__(self, message: str) -> None:
        super().__init__("E_UI", message)
