from __future__ import annotations

from typing import Any


class GSBError(Exception):
    """Base error carrying a stable machine-readable code."""

    code = "E_GSB"

    def __init__(self, message: str, *, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or []

    def to_dict(self) -> dict[str, Any]:
        return {"ok": False, "error_code": self.code, "message": self.message, "details": self.details}


class SchemaViolation(GSBError):
    code = "E_SCHEMA_VIOLATION"

    def __init__(self, issues: list[dict[str, Any]]):
        self.issues = issues
        first = issues[0] if issues else {}
        where = first.get("id") or first.get("entity") or "input"
        field = first.get("field") or "<unknown>"
        super().__init__(
            f"E_SCHEMA_VIOLATION at {where}.{field}: {first.get('message', 'invalid schema')}",
            details=issues,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["issues"] = self.issues
        return payload


class SnapshotUnavailable(GSBError):
    code = "E_SNAPSHOT_UNAVAILABLE"


class SnapshotVersionMismatch(GSBError):
    code = "E_SNAPSHOT_VERSION_MISMATCH"


class BudgetPending(GSBError):
    code = "E_SCOPE_BUDGET_PENDING"
