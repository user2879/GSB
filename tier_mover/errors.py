"""Stable business error codes and structured exceptions for tier_mover.

Every failure raised by the package is a :class:`TierMoverError` carrying a
stable ``code`` plus a structured ``context`` mapping so that callers (and the
JSONL structured log) never have to parse human readable text.
"""

from __future__ import annotations

from typing import Any, Dict


class TierMoverError(Exception):
    """Base class carrying a stable business code and structured context."""

    code = "E_UNKNOWN"

    def __init__(self, message: str, context: Dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: Dict[str, Any] = dict(context or {})

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


class SpecError(TierMoverError):
    """The strategy/object manifest failed structural validation."""

    code = "E_SPEC"


class IntegrityError(TierMoverError):
    """An object digest mismatch or dual-ownership inconsistency was found."""

    code = "E_INTEGRITY"


class CapacityError(TierMoverError):
    """No stable placement/migration can keep every tier within capacity."""

    code = "E_CAPACITY"


class VersionConflictError(TierMoverError):
    """The persisted snapshot version conflicts with the session version."""

    code = "E_VERSION"


class ReplayError(TierMoverError):
    """The deterministic replay diverged from the recorded ledger."""

    code = "E_REPLAY"


#: Exit code used by the CLI for every handled business failure.
BUSINESS_EXIT_CODE = 2
