"""Stable business and transport error codes for the local rate-limit chain."""

E_SPEC = "E_SPEC"
E_LEASE = "E_LEASE"
E_PORT = "E_PORT"
E_TIMEOUT = "E_TIMEOUT"
E_REPLY = "E_REPLY"
E_LEDGER = "E_LEDGER"
E_DIVERGENCE = "E_DIVERGENCE"

ALL_CODES = {
    E_SPEC,
    E_LEASE,
    E_PORT,
    E_TIMEOUT,
    E_REPLY,
    E_LEDGER,
    E_DIVERGENCE,
}


class RlcbError(Exception):
    """Base class carrying a stable code plus structured context."""

    code = "E_RLCB"

    def __init__(self, message: str, *, code: str | None = None, **context):
        super().__init__(message)
        if code is not None:
            self.code = code
        self.context = context

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": str(self),
            "context": self.context,
        }


class SpecError(RlcbError):
    code = E_SPEC

    def __init__(self, message: str, *, field: str, line: int | None, source: str):
        super().__init__(
            message, field=field, line=line, source=source
        )
        self.field = field
        self.line = line
        self.source = source


class LeaseError(RlcbError):
    code = E_LEASE


class PortError(RlcbError):
    code = E_PORT


class TimeoutError_(RlcbError):
    code = E_TIMEOUT


class ReplyError(RlcbError):
    code = E_REPLY


class LedgerError(RlcbError):
    code = E_LEDGER


class DivergenceError(RlcbError):
    code = E_DIVERGENCE
