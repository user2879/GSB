from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class ErrorRecord:
    code: str
    message: str
    path: str = ""
    line: int | None = None
    business_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "line": self.line,
        }
        if self.business_code:
            record["business_code"] = self.business_code
        if self.details:
            record["details"] = self.details
        return record


class BusinessError(Exception):
    """A structured, expected business failure."""

    def __init__(self, records: Iterable[ErrorRecord] | ErrorRecord, message: str | None = None):
        if isinstance(records, ErrorRecord):
            normalized = [records]
        else:
            normalized = list(records)
        if not normalized:
            raise ValueError("BusinessError requires at least one ErrorRecord")
        self.records = normalized
        self.code = normalized[0].code
        super().__init__(message or normalized[0].message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "errors": [record.to_dict() for record in self.records],
        }


class SpecValidationError(BusinessError):
    """The complete specification is rejected; no event has been processed."""

    def __init__(self, records: Iterable[ErrorRecord] | ErrorRecord):
        super().__init__(records, "specification rejected with E_SPEC")
        self.code = "E_SPEC"


class VersionMismatchError(BusinessError):
    def __init__(self, message: str, *, path: str = "ledger.parameter_version", details: dict[str, Any] | None = None):
        record = ErrorRecord(
            code="E_VERSION",
            message=message,
            path=path,
            business_code="E_VERSION",
            details=details or {},
        )
        super().__init__([record], message)


class ReplayMismatchError(BusinessError):
    def __init__(self, record: ErrorRecord):
        super().__init__([record], record.message)


def issue_record(
    code: str,
    message: str,
    *,
    event_seq: int | None = None,
    pool_id: str | None = None,
    connection_id: str | None = None,
    severity: str = "ERROR",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic runtime issue record used by logs and audits."""

    return {
        "code": code,
        "severity": severity,
        "message": message,
        "event_seq": event_seq,
        "pool_id": pool_id,
        "connection_id": connection_id,
        "details": details or {},
    }
