from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class VScanError(Exception):
    """Base error carrying a stable, machine-readable error code."""

    code = "E_VSCAN"

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.detail}


@dataclass
class SpecIssue:
    segment_id: str
    field: str
    message: str


class SpecError(VScanError):
    code = "E_SPEC"

    def __init__(self, issue: SpecIssue) -> None:
        super().__init__(
            issue.message,
            segment_id=issue.segment_id,
            field=issue.field,
        )
        self.issue = issue


class DivergenceError(VScanError):
    code = "E_DIVERGE"

    def __init__(
        self,
        field: str,
        *,
        segment_id: str | None = None,
        column: str | None = None,
        batch_no: int | None = None,
        expected: Any = None,
        actual: Any = None,
        message: str | None = None,
    ) -> None:
        location = []
        if segment_id is not None:
            location.append(f"segment={segment_id}")
        if column is not None:
            location.append(f"column={column}")
        if batch_no is not None:
            location.append(f"batch={batch_no}")
        text = message or f"incremental/full rebuild diverged at field {field}"
        if location:
            text += " (" + ", ".join(location) + ")"
        super().__init__(
            text,
            field=field,
            segment_id=segment_id,
            column=column,
            batch_no=batch_no,
            expected=expected,
            actual=actual,
        )


class BudgetError(VScanError):
    code = "E_BUDGET"


class LedgerError(VScanError):
    code = "E_LEDGER"


@dataclass
class RecoveryBatch:
    query_id: str
    segment_id: str
    batch_no: int
    state: str
    detail: str = ""


@dataclass
class RecoveryReport:
    unknown: list[RecoveryBatch] = field(default_factory=list)
    committed: list[RecoveryBatch] = field(default_factory=list)
    backup_path: str | None = None
    replaced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "unknown": [batch.__dict__ for batch in self.unknown],
            "committed": [batch.__dict__ for batch in self.committed],
            "backup_path": self.backup_path,
            "replaced": self.replaced,
        }
