"""Diagnostic error taxonomy.

All spec/semantic failures raised by the diagnostic components use
:class:`DiagError` so an independent reviewer can distinguish the four
contracted causes from ordinary Python exceptions.
"""


class DiagError(Exception):
    """Base error carrying a stable code, offending object id and field."""

    code = "E_INTERNAL"

    def __init__(self, message, object_id=None, field=None, detail=None):
        super().__init__(message)
        self.object_id = object_id
        self.field = field
        self.detail = detail

    def as_dict(self):
        return {
            "code": self.code,
            "message": str(self),
            "object_id": self.object_id,
            "field": self.field,
            "detail": self.detail,
        }


class SpecError(DiagError):
    """Declarations failed strong validation."""

    code = "E_SPEC"


class UpstreamError(DiagError):
    """A shard stays frozen because an upstream outcome/summary is missing."""

    code = "E_UPSTREAM"


class LeaseError(DiagError):
    """Optimistic lease version check failed or the lease is not active."""

    code = "E_LEASE"


class ReplayError(DiagError):
    """Recomputed state disagrees with the committed ledger record."""

    code = "E_REPLAY"
