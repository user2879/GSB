"""Business error hierarchy for the obsflow package.

Every error carries a stable business code (``E_*``). Error codes are part of
the public contract and are used by the CLI, the structured logger and the
diagnostics UI.
"""
from __future__ import annotations

from typing import Any, Optional


class ObsFlowError(Exception):
    """Base class for all obsflow business errors."""

    code: str = "E_GENERIC"

    def __init__(self, message: str, *, field: Optional[str] = None,
                 line_no: Optional[int] = None, detail: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        # Name of the offending field (``None`` when not field specific).
        self.field = field
        # 1-based record line number within the offending stream file.
        self.line_no = line_no
        # Extra structured context that is safe to serialise to JSON.
        self.detail: dict[str, Any] = dict(detail or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "line_no": self.line_no,
            "detail": self.detail,
        }


class FormatError(ObsFlowError):
    """Record decoding / strong validation failure (business code E_FORMAT)."""

    code = "E_FORMAT"


class OrderError(FormatError):
    """Monotonic observation sequence regression (business code E_ORDER).

    It is a :class:`FormatError` so that strict load-time validation still
    reports ``E_FORMAT`` as required, while the order-injection pipeline can
    expose the more specific ``E_ORDER`` code.
    """

    code = "E_ORDER"


class IntegrityError(ObsFlowError):
    """Truncated / unreadable record stream (business code E_INTEGRITY)."""

    code = "E_INTEGRITY"


class SnapshotError(ObsFlowError):
    """Snapshot content digest / version mismatch (business code E_SNAPSHOT)."""

    code = "E_SNAPSHOT"


class LeaseError(ObsFlowError):
    """Lease optimistic-version conflict or lease not held (E_LEASE)."""

    code = "E_LEASE"


class ReplayError(ObsFlowError):
    """Ledger replay field-level mismatch (business code E_REPLAY)."""

    code = "E_REPLAY"
