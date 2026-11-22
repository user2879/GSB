"""Structured business error codes for the lease diagnostic application.

Every validation or adjudication failure is reported with a stable business
code plus a list of human-readable locations so that a whole input set can be
rejected atomically (no partial loading).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Input declaration / strong validation errors.
E_POOL = "E_POOL"            # pool declaration or request-stream validation
E_ROLLBACK = "E_ROLLBACK"    # rollback target missing or rollback chain cyclic
# Runtime adjudication errors.
E_LEASE = "E_LEASE"          # conflicting lease ownership / loser of contention
E_TIMEOUT = "E_TIMEOUT"      # single grant socket timeout (retryable)
# Recovery / snapshot errors.
E_GAP = "E_GAP"              # renewal event referenced by ledger is missing
E_SNAPSHOT = "E_SNAPSHOT"    # snapshot version/digest mismatch
E_REPLAY = "E_REPLAY"        # deterministic replay diverges from ledger


@dataclass
class LeaseError(Exception):
    """An error carrying a business code and structured location details."""

    code: str
    message: str
    locations: list[str] = field(default_factory=list)
    extra: dict | None = None

    def __str__(self) -> str:
        head = f"{self.code}: {self.message}"
        if self.locations:
            head += " | at " + "; ".join(self.locations)
        return head

    def to_dict(self) -> dict:
        data = {"code": self.code, "message": self.message,
                "locations": list(self.locations)}
        if self.extra:
            data["extra"] = self.extra
        return data
