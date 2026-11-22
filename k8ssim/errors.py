"""Domain error types for the scheduling workbench.

Each error carries a stable code and a machine-readable location so the UI and
demo script can present deterministic, first-failure diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


class SimError(Exception):
    """Base class for all workbench errors."""

    code: str = "E_GENERIC"

    def __init__(self, message: str, path: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.path = path

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.path:
            return f"[{self.code}] {self.path}: {self.message}"
        return f"[{self.code}] {self.message}"


class SpecError(SimError):
    """Input specification is invalid; the whole document is rejected."""

    code = "E_SPEC"


class SnapshotError(SimError):
    """Snapshot format version or content digest does not match."""

    code = "E_SNAPSHOT"

    def __init__(self, message: str, mismatches: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.mismatches: List[str] = list(mismatches or [])


class ReplayError(SimError):
    """Replaying the ledger produced a different scheduling conclusion."""

    code = "E_REPLAY"

    def __init__(self, message: str, pod_id: str, field: str) -> None:
        super().__init__(message, path=f"pods[{pod_id}].{field}")
        self.pod_id = pod_id
        self.field = field


@dataclass
class ValidationIssue:
    """A single validation finding; only the first one aborts loading."""

    path: str
    message: str
    value: Any = field(default=None, repr=False)

    def render(self) -> str:
        return f"{self.path}: {self.message}"
