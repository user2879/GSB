"""Business error codes and the single platform exception type.

Every recoverable, user-facing failure in the platform raises :class:`PlatformError`
carrying one of the documented ``E_*`` codes.  Codes are part of the contract and
must remain stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Stable business error code catalogue
# ---------------------------------------------------------------------------
E_MANIFEST = "E_MANIFEST"      # manifest / registry load-time hard validation
E_CYCLE = "E_CYCLE"            # derivation dependency graph contains a cycle
E_GRID = "E_GRID"              # illegal precision grid (scale) parameter
E_REF = "E_REF"                # deleting / referencing a key that is still in use
E_NODE = "E_NODE"              # node subprocess failed to start / died
E_HANDSHAKE = "E_HANDSHAKE"    # node handshake timed out or was malformed
E_PORT = "E_PORT"              # loopback port already occupied / cannot bind
E_CONVERGE = "E_CONVERGE"      # receipt summary mismatch or missing receipt
E_VIEW = "E_VIEW"              # interactive view disagrees with authoritative replay
E_REPLAY = "E_REPLAY"          # snapshot replay is not reproducible
E_STATE = "E_STATE"            # invalid state-machine transition
E_IO = "E_IO"                  # local file / sqlite IO failure


@dataclass
class ErrorDetail:
    """One machine-readable failure location.

    ``key`` is the configuration item key (or a node/request id) and ``source``
    identifies the trigger source (file, stage or component).  Structured logs
    always carry at least one of them.
    """

    reason: str
    key: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "key": self.key, "source": self.source}


@dataclass
class PlatformError(Exception):
    code: str
    message: str
    details: list[ErrorDetail] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": [d.to_dict() for d in self.details],
        }

    @staticmethod
    def raise_if(code: str, message: str, details: list[ErrorDetail]) -> None:
        if details:
            raise PlatformError(code, message, details)
