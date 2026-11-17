"""Business error codes for the chaos engineering component.

All errors carry a stable machine-readable ``code`` so that callers (CLI,
GUI, embedded integrations) can branch on a contract rather than message text.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional


class ChaosError(Exception):
    """Base class for every business error raised by this package."""

    code = "E_INTERNAL"

    def __init__(self, message: str, *, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details if details is not None else {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class SpecError(ChaosError):
    """Raised when an experiment declaration fails strong validation.

    The declaration is rejected as a whole; every offending JSON path is
    reported so the caller can fix all problems in one pass. Nothing from the
    declaration is partially applied.
    """

    code = "E_SPEC"

    def __init__(self, paths: Iterable[str], message: str = "experiment declaration is invalid") -> None:
        paths = list(paths)
        super().__init__(message, details={"paths": paths})
        self.paths: List[str] = paths


class TimelineError(ChaosError):
    """Raised when a regenerated timeline digest deviates from the ledger."""

    code = "E_TIMELINE"

    def __init__(self, message: str, *, first_diverging_index: int, expected: Any, actual: Any) -> None:
        super().__init__(
            message,
            details={
                "first_diverging_index": first_diverging_index,
                "expected": expected,
                "actual": actual,
            },
        )
        self.first_diverging_index = first_diverging_index


class TargetError(ChaosError):
    """Raised when a loopback target service cannot be reached or bound."""

    code = "E_TARGET"


class ReplayError(ChaosError):
    """Raised when replay cannot reproduce a committed ledger entry."""

    code = "E_REPLAY"

    def __init__(self, message: str, *, basis: str, injection_index: int, field: str,
                 ledger_value: Any, replay_value: Any) -> None:
        super().__init__(
            message,
            details={
                "basis": basis,
                "injection_index": injection_index,
                "field": field,
                "ledger_value": ledger_value,
                "replay_value": replay_value,
            },
        )
        self.basis = basis
        self.injection_index = injection_index
        self.field = field
        self.ledger_value = ledger_value
        self.replay_value = replay_value


class LedgerError(ChaosError):
    """Raised when an on-disk ledger is corrupt or has an unknown format."""

    code = "E_LEDGER"
