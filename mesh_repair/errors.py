"""Business errors and stable process exit codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    E_MESH_FORMAT = "E_MESH_FORMAT"
    E_BLOCK_KEY = "E_BLOCK_KEY"
    E_BOUNDARY_CONFLICT = "E_BOUNDARY_CONFLICT"
    E_BUDGET_DENIED = "E_BUDGET_DENIED"
    E_VOLUME_DEVIATE = "E_VOLUME_DEVIATE"
    E_RESIDUAL_TOPOLOGY = "E_RESIDUAL_TOPOLOGY"
    E_VERIFIER_MISMATCH = "E_VERIFIER_MISMATCH"
    E_INTERNAL = "E_INTERNAL"


EXIT_CODES: dict[ErrorCode, int] = {
    ErrorCode.E_MESH_FORMAT: 10,
    ErrorCode.E_BLOCK_KEY: 11,
    ErrorCode.E_BOUNDARY_CONFLICT: 12,
    ErrorCode.E_BUDGET_DENIED: 13,
    ErrorCode.E_VOLUME_DEVIATE: 14,
    ErrorCode.E_RESIDUAL_TOPOLOGY: 15,
    ErrorCode.E_VERIFIER_MISMATCH: 16,
    ErrorCode.E_INTERNAL: 1,
}


@dataclass
class BusinessError(Exception):
    code: ErrorCode
    message: str
    location: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.location:
            return f"{self.code.value}: {self.message} ({self.location})"
        return f"{self.code.value}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "fail",
            "business_code": self.code.value,
            "message": self.message,
            "location": self.location,
            "details": self.details,
            "exit_code": EXIT_CODES[self.code],
        }


def require(condition: bool, code: ErrorCode, message: str, **kwargs: Any) -> None:
    if not condition:
        raise BusinessError(code=code, message=message, **kwargs)
