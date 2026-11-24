from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


E_CORPUS_INVALID = "E_CORPUS_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_RECEIPT_INVALID = "E_RECEIPT_INVALID"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_VECTOR_NONFINITE = "E_VECTOR_NONFINITE"
E_BOUNDARY_FLIP = "E_BOUNDARY_FLIP"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_SNAPSHOT_RECOVERY = "E_SNAPSHOT_RECOVERY"
E_INJECTION_FAILED = "E_INJECTION_FAILED"


@dataclass
class Failure:
    code: str
    message: str
    path: Optional[str] = None
    line: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    field: Optional[str] = None
    injection_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "line": self.line,
            "field": self.field,
            "injection_id": self.injection_id,
            "details": self.details,
        }


class DiagnosticError(Exception):
    """An input or runtime fault carrying one stable business code."""

    def __init__(
        self,
        code: str,
        message: str,
        failures: Optional[List[Failure]] = None,
        injection_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.failures = failures or []
        self.injection_id = injection_id

    def to_report(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "injection_id": self.injection_id,
            "failures": [item.to_dict() for item in self.failures],
        }
