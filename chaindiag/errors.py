from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


E_CHAIN_FORMAT = "E_CHAIN_FORMAT"
E_OP_QUOTA_EXCEEDED = "E_OP_QUOTA_EXCEEDED"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_INJECTED_FAULT = "E_INJECTED_FAULT"
E_SNAPSHOT_RECOVERY = "E_SNAPSHOT_RECOVERY"


@dataclass
class Issue:
    code: str
    message: str
    location: str | None = None
    line: int | None = None
    field: str | None = None
    severity: str = "error"
    context: dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "line": self.line,
            "field": self.field,
            "severity": self.severity,
            "context": self.context,
        }


class ChainFormatError(ValueError):
    def __init__(self, issues: Issue | list[Issue]):
        self.issues = [issues] if isinstance(issues, Issue) else list(issues)
        super().__init__("; ".join(f"{i.location or '<root>'}: {i.message}" for i in self.issues))

    def to_report(self) -> dict[str, Any]:
        return {
            "status": "error",
            "error": {
                "code": E_CHAIN_FORMAT,
                "issues": [issue.to_dict() for issue in self.issues],
            },
        }


class InjectedFault(RuntimeError):
    def __init__(self, injection_id: str, fault_type: str, message: str, **context: Any):
        self.injection_id = injection_id
        self.fault_type = fault_type
        self.context = context
        super().__init__(f"[{injection_id}] {fault_type}: {message}")
