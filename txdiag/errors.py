"""Typed, machine-readable diagnostics errors."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


@dataclass
class TxDiagError(Exception):
    code: str
    message: str
    txid: str | None = None
    row: str | None = None
    field: str | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.txid is not None:
            data["txid"] = self.txid
        if self.row is not None:
            data["row"] = self.row
        if self.field is not None:
            data["field"] = self.field
        if self.details:
            data["details"] = self.details
        return data

    def __str__(self) -> str:
        parts = [f"{self.code}: {self.message}"]
        if self.txid is not None:
            parts.append(f"txid={self.txid}")
        if self.row is not None:
            parts.append(f"row={self.row}")
        if self.field is not None:
            parts.append(f"field={self.field}")
        return " ".join(parts)


class SpecError(TxDiagError):
    def __init__(self, message: str, txid: str | None = None, field: str | None = None,
                 row: str | None = None, details: dict[str, Any] | None = None) -> None:
        super().__init__("E_SPEC", message, txid=txid, row=row, field=field,
                         details=details or {})


class TimeoutErrorDiagnostic(TxDiagError):
    def __init__(self, message: str, txid: str | None = None, row: str | None = None,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__("E_TIMEOUT", message, txid=txid, row=row,
                         field="wait_timeout_ms", details=details or {})


class DivergenceError(TxDiagError):
    def __init__(self, message: str, txid: str | None = None, row: str | None = None,
                 field: str | None = None, details: dict[str, Any] | None = None) -> None:
        super().__init__("E_DIVERGE", message, txid=txid, row=row, field=field,
                         details=details or {})


class ReplayError(TxDiagError):
    def __init__(self, message: str, txid: str | None = None, row: str | None = None,
                 field: str | None = None, details: dict[str, Any] | None = None) -> None:
        super().__init__("E_REPLAY", message, txid=txid, row=row, field=field,
                         details=details or {})
