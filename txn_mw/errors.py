"""Structured error types used by the middleware and its tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MWError(Exception):
    """An error that is safe to expose as a structured protocol response."""

    code: str
    message: str
    path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.path:
            return f"{self.code}: {self.message} (path={self.path})"
        return f"{self.code}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            result["path"] = self.path
        if self.details:
            result["details"] = self.details
        return result


class ProtocolError(MWError):
    def __init__(self, message: str, path: str = "$", details: dict[str, Any] | None = None):
        super().__init__("E_TXN", message, path, details or {})


class ConflictError(MWError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("E_CONFLICT", message, None, details or {})


class QueryExhausted(MWError):
    def __init__(self, txn_id: str, used: int):
        super().__init__(
            "E_QUERY",
            "transaction query limit exhausted; manual review required",
            "$.txn_id",
            {"txn_id": txn_id, "query_count": used},
        )


class StateError(MWError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("E_STATE", message, None, details or {})


class DuplicateTxn(MWError):
    def __init__(self, txn_id: str):
        super().__init__(
            "E_DUP",
            "transaction identifier is already registered",
            "$.txn_id",
            {"txn_id": txn_id},
        )

