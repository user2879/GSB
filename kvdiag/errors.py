"""Business error types with stable error codes."""


class KvDiagError(Exception):
    """Base error carrying a stable business code and structured context."""

    code = "E_INTERNAL"

    def __init__(self, message, *, shard_id=None, field=None, details=None):
        super().__init__(message)
        self.message = message
        self.shard_id = shard_id
        self.field = field
        self.details = dict(details or {})

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "shard_id": self.shard_id,
            "field": self.field,
            "details": self.details,
        }


class SpecError(KvDiagError):
    """Declaration / sample validation failure."""

    code = "E_SPEC"


class LeaseError(KvDiagError):
    """Lease optimistic-version or lifecycle failure."""

    code = "E_LEASE"


class ChainError(KvDiagError):
    """Snapshot chain integrity failure."""

    code = "E_CHAIN"


class DivergeError(KvDiagError):
    """Index position vs lease expiry position divergence."""

    code = "E_DIVERGE"


class SnapshotError(KvDiagError):
    """Ledger/session snapshot version or digest mismatch."""

    code = "E_SNAPSHOT"
