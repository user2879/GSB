"""Stable error codes and exception hierarchy for the platform.

All externally reported failures carry one of the ``E_*`` codes required by the
specification.  Every exception also carries a *location* describing the first
offending position inside the input document (or ``"<runtime>"`` when no input
position applies) and an optional structured ``context`` dictionary.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class PlatformError(Exception):
    """Base class for every deterministic, code-bearing platform failure."""

    code = "E_PLATFORM"

    def __init__(
        self,
        message: str,
        location: str = "<runtime>",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.location = location
        self.context: Dict[str, Any] = dict(context or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "context": self.context,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code} @ {self.location}: {self.message}"


class ManifestError(PlatformError):
    """Schema / structural validation failure of a manifest input file."""

    code = "E_MANIFEST"


class CycleError(ManifestError):
    """Derived dependency graph contains a cycle (sub-code of manifest)."""

    code = "E_CYCLE"


class GridError(PlatformError):
    """Illegal fixed-point grid / scale parameter (not an integer power of 10)."""

    code = "E_GRID"


class PortError(PlatformError):
    """A node could not bind its loopback TCP port."""

    code = "E_PORT"


class HandshakeError(PlatformError):
    """TCP connect / handshake did not complete inside the socket timeout."""

    code = "E_HANDSHAKE"


class ReplyError(PlatformError):
    """A node reply was missing required fields or could not be parsed."""

    code = "E_REPLY"


class ConvergeError(PlatformError):
    """Expected and observed derived digests could not be reconciled."""

    code = "E_CONVERGE"


class ReplayError(PlatformError):
    """Ledger replay could not reproduce a committed receipt byte-for-byte."""

    code = "E_REPLAY"


class SnapshotError(PlatformError):
    """Snapshot format version / content digest is unreadable or unsupported."""

    code = "E_SNAPSHOT"
