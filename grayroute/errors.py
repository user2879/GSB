"""Domain error types.

Each error carries a stable ``code`` and enough locating context to point at
the first offending object and field.  Codes used across the system:

* ``E_SPEC``   - malformed JSON/NDJSON declaration or request stream
* ``E_FIELD``  - malformed receipt field
* ``E_VERSION``- parameter version conflict inside one batch
* ``E_REPLAY`` - post-recovery replay mismatch against the ledger
"""
from __future__ import annotations


class GrayRouteError(Exception):
    """Base class for all domain errors."""

    code = "E_GENERIC"

    def __init__(self, message: str, *, object_id: str | None = None,
                 field: str | None = None, **extra):
        super().__init__(message)
        self.message = message
        self.object_id = object_id
        self.field = field
        self.extra = extra

    def to_dict(self) -> dict:
        out = {"code": self.code, "message": self.message}
        if self.object_id is not None:
            out["object_id"] = self.object_id
        if self.field is not None:
            out["field"] = self.field
        if self.extra:
            out["detail"] = self.extra
        return out


class SpecError(GrayRouteError):
    code = "E_SPEC"


class FieldError(GrayRouteError):
    code = "E_FIELD"


class VersionConflictError(GrayRouteError):
    code = "E_VERSION"


class ReplayError(GrayRouteError):
    code = "E_REPLAY"


class SnapshotError(GrayRouteError):
    code = "E_SNAPSHOT"
