"""Structured error codes used across the shard quota ledger component."""


class LedgerError(Exception):
    """Base error carrying a stable machine-readable code and location info."""

    code = "E_INTERNAL"

    def __init__(self, message, entity=None, field=None, **extra):
        super().__init__(message)
        self.message = message
        self.entity = entity
        self.field = field
        self.extra = extra

    def to_dict(self):
        data = {"code": self.code, "message": self.message}
        if self.entity is not None:
            data["entity"] = self.entity
        if self.field is not None:
            data["field"] = self.field
        data.update(self.extra)
        return data


class SpecError(LedgerError):
    code = "E_SPEC"


class DependError(LedgerError):
    code = "E_DEPEND"


class CycleError(LedgerError):
    code = "E_CYCLE"


class PortError(LedgerError):
    code = "E_PORT"


class TimeoutError(LedgerError):
    code = "E_TIMEOUT"


class FieldError(LedgerError):
    code = "E_FIELD"


class LeaseError(LedgerError):
    code = "E_LEASE"


class ReplayError(LedgerError):
    code = "E_REPLAY"


class SnapshotError(LedgerError):
    code = "E_SNAPSHOT"
