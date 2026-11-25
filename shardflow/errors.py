"""Typed error codes used across the orchestration stack."""


class ShardflowError(Exception):
    """Base error carrying a stable machine readable code."""

    code = "E_INTERNAL"

    def __init__(self, message, code=None, object_id=None, field=None):
        super().__init__(message)
        if code is not None:
            self.code = code
        self.object_id = object_id
        self.field = field

    def to_dict(self):
        return {
            "code": self.code,
            "message": str(self),
            "object": self.object_id,
            "field": self.field,
        }


class SpecError(ShardflowError):
    """Declaration / event stream violates the specification."""

    code = "E_SPEC"


class PortError(ShardflowError):
    """A loopback port cannot be bound or connected to."""

    code = "E_PORT"


class HandshakeError(ShardflowError):
    """The node handshake did not complete inside the socket timeout."""

    code = "E_HANDSHAKE"


class ProtocolFieldError(ShardflowError):
    """A wire message is missing a required field or has a bad value."""

    code = "E_FIELD"


class LeaseError(ShardflowError):
    """Optimistic lease version mismatch or exhausted renewal budget."""

    code = "E_LEASE"


class DuplicateError(ShardflowError):
    """A cancellation targets a shard that was already cancelled."""

    code = "E_DUP"


class ReplayError(ShardflowError):
    """Independent replay derived a different value than the ledger."""

    code = "E_REPLAY"


class LedgerError(ShardflowError):
    """Ledger format/content identity is inconsistent."""

    code = "E_LEDGER"

