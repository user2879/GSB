"""Business error types and exit codes."""

from __future__ import annotations

from typing import Any


ERROR_EXIT_CODE = 2


class DocDBError(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def payload(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        result.update(self.details)
        return result


class DocumentError(DocDBError):
    def __init__(
        self,
        message: str,
        *,
        primary_key: Any = None,
        field_path: str = "$",
        **details: Any,
    ) -> None:
        super().__init__(
            "E_DOC",
            message,
            primary_key=primary_key,
            field_path=field_path,
            **details,
        )


class SegmentError(DocDBError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_SEGMENT", message, **details)


class BoundError(DocDBError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_BOUND", message, **details)


class LedgerError(DocDBError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_LEDGER", message, **details)


class ReplayError(DocDBError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_REPLAY", message, **details)


class StaleError(DocDBError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_STALE", message, **details)


class StateError(DocDBError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_STATE", message, **details)


class SnapshotError(DocDBError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_SNAPSHOT", message, **details)
