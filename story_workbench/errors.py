"""Structured errors used by the CLI, tests, and the desktop UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_SPEC_INVALID = "E_SPEC_INVALID"
E_SPEC_IO = "E_SPEC_IO"
E_SEQUENCE_INVALID = "E_SEQUENCE_INVALID"
E_STATE_NONFINITE = "E_STATE_NONFINITE"
E_INJECTION_INVALID = "E_INJECTION_INVALID"
E_DEPENDENCY_INVALID = "E_DEPENDENCY_INVALID"
E_NONDETERMINISTIC = "E_NONDETERMINISTIC"
E_INTERNAL = "E_INTERNAL"


@dataclass
class WorkbenchError(Exception):
    """A stable machine-readable error."""

    code: str
    message: str
    location: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "details": self.details,
        }


class SpecInvalid(WorkbenchError):
    def __init__(
        self,
        message: str,
        *,
        location: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code=E_SPEC_INVALID,
            message=message,
            location=location or {},
            details=details or {},
        )


class SequenceInvalid(WorkbenchError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code=E_SEQUENCE_INVALID,
            message=message,
            location={},
            details=details or {},
        )


class InjectionInvalid(WorkbenchError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code=E_INJECTION_INVALID,
            message=message,
            location={},
            details=details or {},
        )
