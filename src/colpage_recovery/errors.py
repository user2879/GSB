from __future__ import annotations

from typing import Any


class DiagnosticError(Exception):
    """Base class for business-level diagnostic failures."""

    code = "E_DIAG"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        lsn: int | None = None,
        page_id: str | None = None,
        tx_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.lsn = lsn
        self.page_id = page_id
        self.tx_id = tx_id
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.lsn is not None:
            payload["lsn"] = self.lsn
        if self.page_id is not None:
            payload["page_id"] = self.page_id
        if self.tx_id is not None:
            payload["tx_id"] = self.tx_id
        if self.field is not None:
            payload["field"] = self.field
        if self.details:
            payload["details"] = self.details
        return payload


class WalValidationError(DiagnosticError):
    code = "E_WAL"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        if self.code != "E_WAL":
            payload["wal_category"] = "E_WAL"
        return payload


class CommitMarkerConflictError(WalValidationError):
    code = "E_COMMIT"


class RangeOutOfBoundsError(WalValidationError):
    code = "E_RANGE"


class SnapshotMismatchError(DiagnosticError):
    code = "E_SNAPSHOT"


class ReplayMismatchError(DiagnosticError):
    code = "E_REPLAY"


class BudgetExceededError(DiagnosticError):
    code = "E_BUDGET"


class LedgerStateError(DiagnosticError):
    code = "E_LEDGER"
