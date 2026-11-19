"""Session state machine with explicit, auditable transitions."""
from __future__ import annotations

from ..errors import E_STATE, ErrorDetail, PlatformError

# Allowed transitions; every state move is recorded in the ledger.
_TRANSITIONS: dict[str, set[str]] = {
    "INIT": {"COMPILED", "FAILED"},
    "COMPILED": {"DISPATCHING", "FAILED"},
    "DISPATCHING": {"CONVERGED", "DIVERGED", "FAILED"},
    "CONVERGED": {"PERSISTED", "FAILED"},
    "DIVERGED": {"PERSISTED", "FAILED"},
    "PERSISTED": {"RESTORED", "COMPILED", "FAILED"},
    "RESTORED": {"COMPILED", "FAILED"},
    "FAILED": {"INIT"},
}


class StateMachine:
    def __init__(self, initial: str = "INIT") -> None:
        self.state = initial
        self.history: list[str] = [initial]

    def transition(self, target: str) -> None:
        allowed = _TRANSITIONS.get(self.state, set())
        if target not in allowed:
            raise PlatformError(
                E_STATE,
                f"illegal state transition {self.state} -> {target}",
                [
                    ErrorDetail(
                        reason=f"allowed targets from {self.state}: {sorted(allowed)}",
                        source="state-machine",
                    )
                ],
            )
        self.state = target
        self.history.append(target)

    def snapshot(self) -> dict[str, str | list[str]]:
        return {"state": self.state, "history": list(self.history)}
