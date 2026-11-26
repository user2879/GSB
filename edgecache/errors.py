"""Structured errors used by the CLI, TCP protocol and replay verifier."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


@dataclass
class EdgeError(Exception):
    code: str
    message: str
    field: str | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)
    exit_code: int = 2

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.field is not None:
            payload["field"] = self.field
        if self.details:
            payload["details"] = self.details
        return payload


class SpecError(EdgeError):
    def __init__(self, field: str, message: str, **details: Any) -> None:
        super().__init__("E_SPEC", message, field=field, details=details, exit_code=3)


class ConserveError(EdgeError):
    def __init__(
        self,
        batch_id: str,
        node_diff: int,
        global_diff: int,
        quota_version: int | str,
        **details: Any,
    ) -> None:
        super().__init__(
            "E_CONSERVE",
            f"eviction conservation failed for batch {batch_id}",
            details={
                "batch_id": batch_id,
                "node_diff": node_diff,
                "global_diff": global_diff,
                "quota_version": quota_version,
                **details,
            },
            exit_code=4,
        )


class LeaseError(EdgeError):
    def __init__(self, lease_id: str, expected_version: int, actual_version: int | None) -> None:
        super().__init__(
            "E_LEASE",
            f"optimistic lease renewal rejected for {lease_id}",
            details={
                "lease_id": lease_id,
                "expected_version": expected_version,
                "actual_version": actual_version,
            },
            exit_code=5,
        )


class KilledError(EdgeError):
    def __init__(self, node_id: str, **details: Any) -> None:
        super().__init__(
            "E_KILLED",
            f"node process {node_id} is not running",
            details={"node_id": node_id, **details},
            exit_code=6,
        )


class PortError(EdgeError):
    def __init__(self, node_id: str, host: str, port: int, **details: Any) -> None:
        super().__init__(
            "E_PORT",
            f"loopback port {host}:{port} is unavailable for node {node_id}",
            details={"node_id": node_id, "host": host, "port": port, **details},
            exit_code=7,
        )


class TimeoutErrorEdge(EdgeError):
    def __init__(self, node_id: str, timeout_ms: int, **details: Any) -> None:
        super().__init__(
            "E_TIMEOUT",
            f"real socket timeout after {timeout_ms} ms for node {node_id}",
            details={"node_id": node_id, "timeout_ms": timeout_ms, **details},
            exit_code=8,
        )


def protocol_error(code: str, message: str, **details: Any) -> EdgeError:
    return EdgeError(code, message, details=details, exit_code=9)
