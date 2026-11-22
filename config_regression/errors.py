"""Stable platform error codes.

All user-facing validation / runtime failures are surfaced with one of the
stable ``E_*`` codes defined here.  Each error carries one or more
``(subject, reason)`` violation tuples so that problems can be listed
deterministically, independent of dict or hash iteration order.
"""

from __future__ import annotations

from dataclasses import dataclass, field


E_MANIFEST = "E_MANIFEST"
E_REGRESSION = "E_REGRESSION"
E_EXHAUST = "E_EXHAUST"
E_REPLAY = "E_REPLAY"
E_SNAPSHOT = "E_SNAPSHOT"


class PlatformError(Exception):
    """Base class for platform errors carrying a stable code."""

    code: str = "E_PLATFORM"

    def __init__(self, message: str, violations: list[tuple[str, str]] | None = None):
        super().__init__(message)
        self.message = message
        self.violations: list[tuple[str, str]] = sorted(violations or [])

    def details(self) -> list[dict[str, str]]:
        return [{"subject": subject, "reason": reason}
                for subject, reason in self.violations]


class ManifestError(PlatformError):
    code = E_MANIFEST


class RegressionError(PlatformError):
    code = E_REGRESSION


class ExhaustError(PlatformError):
    code = E_EXHAUST


class ReplayError(PlatformError):
    code = E_REPLAY


class SnapshotError(PlatformError):
    code = E_SNAPSHOT


@dataclass
class ViolationCollector:
    """Collect violations and raise :class:`ManifestError` as one batch."""

    violations: list[tuple[str, str]] = field(default_factory=list)

    def add(self, subject: str, reason: str) -> None:
        self.violations.append((subject, reason))

    def check(self) -> None:
        if self.violations:
            raise ManifestError("manifest validation failed", self.violations)
