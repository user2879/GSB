"""Business errors used by the workbench and command line interface."""

from __future__ import annotations

from typing import Any, Mapping


class WorkbenchError(Exception):
    """An error with a stable business code and reviewable details."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class SpecError(WorkbenchError):
    def __init__(
        self,
        message: str,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        path: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        payload = dict(details or {})
        if entity_type is not None:
            payload["entity_type"] = entity_type
        if entity_id is not None:
            payload["entity_id"] = entity_id
        if path is not None:
            payload["path"] = path
        super().__init__("E_SPEC", message, details=payload)


class CycleError(WorkbenchError):
    def __init__(
        self,
        message: str,
        *,
        row_key: str,
        cycle: list[int],
        path: str | None = None,
    ) -> None:
        details: dict[str, Any] = {
            "entity_type": "row_key",
            "entity_id": row_key,
            "row_key": row_key,
            "cycle": cycle,
        }
        if path is not None:
            details["path"] = path
        super().__init__("E_CYCLE", message, details=details)


class DivergenceError(WorkbenchError):
    def __init__(
        self,
        message: str,
        *,
        query_id: str,
        row_key: str,
        field: str,
        left: Any,
        right: Any,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        payload = dict(details or {})
        payload.update(
            {
                "query_id": query_id,
                "row_key": row_key,
                "field": field,
                "left": left,
                "right": right,
            }
        )
        super().__init__("E_DIVERGE", message, details=payload)


class SnapshotError(WorkbenchError):
    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__("E_SNAPSHOT", message, details=details)


class LedgerError(WorkbenchError):
    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__("E_LEDGER", message, details=details)


class SchedulerError(WorkbenchError):
    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__("E_SCHEDULER", message, details=details)
