"""Business error codes for the level constraint solver.

Every validation or runtime failure is represented by an :class:`SolverError`
carrying a stable ``E_xxx`` code so callers (and the JSON CLI layer) can react
programmatically.
"""

from __future__ import annotations


class SolverError(Exception):
    """Raised for all expected, business-level failures.

    Attributes:
        code: Stable ``E_xxx`` business code.
        message: Human readable explanation.
        details: Optional structured payload (JSON-serialisable).
    """

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        payload = {"ok": False, "error": {"code": self.code, "message": self.message}}
        if self.details:
            payload["error"]["details"] = self.details
        return payload


E_SCHEMA = "E_SCHEMA"
E_DOMAIN = "E_DOMAIN"
E_CONSTRAINT = "E_CONSTRAINT"
E_CONFLICT = "E_CONFLICT"
E_DIGEST = "E_DIGEST"
E_VERSION = "E_VERSION"
E_INTERRUPTED = "E_INTERRUPTED"
E_IO = "E_IO"
