"""Domain error hierarchy.

Each error carries a stable machine code (E_*) and, where applicable, a
JSON-pointer-like field path so a bad cluster declaration is rejected as a
whole rather than partially accepted.
"""

from __future__ import annotations

from typing import Any, Iterable


class ScalerError(Exception):
    """Base class for all deterministic scaler failures."""

    code = "E_INTERNAL"

    def __init__(self, message: str, *, fields: Iterable[str] | None = None,
                 context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.fields: list[str] = list(fields or [])
        self.context: dict[str, Any] = dict(context or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "fields": self.fields,
            "context": self.context,
        }


class SpecError(ScalerError):
    """The cluster declaration is structurally or semantically invalid."""

    code = "E_SPEC"


class BudgetError(SpecError):
    """Budget / quota parameters are invalid or infeasible (E_BUDGET)."""

    code = "E_BUDGET"


class LoadError(ScalerError):
    """A per-minute load sequence has a gap or missing minute (E_LOAD)."""

    code = "E_LOAD"


class VersionConflictError(ScalerError):
    """A snapshot references a stale or unknown declaration version (E_VERSION)."""

    code = "E_VERSION"


class DivergenceError(ScalerError):
    """Fast cache result differs from the authoritative ledger replay (E_DIVERGE)."""

    code = "E_DIVERGE"


class ReplayError(ScalerError):
    """Recovered ledger rows fail deterministic re-derivation (E_REPLAY)."""

    code = "E_REPLAY"
