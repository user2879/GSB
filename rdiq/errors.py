"""Domain errors with stable, machine-readable codes."""


class RDIQError(Exception):
    """Base class carrying an error code plus a structured context."""

    code = "E_RDIQ"

    def __init__(self, message, *, entry=None, field=None, **context):
        super().__init__(message)
        self.message = message
        self.entry = entry
        self.field = field
        self.context = context

    def to_dict(self):
        payload = {"code": self.code, "message": self.message}
        if self.entry is not None:
            payload["entry"] = self.entry
        if self.field is not None:
            payload["field"] = field_name(self.field)
        if self.context:
            payload["context"] = self.context
        return payload


def field_name(field):
    if isinstance(field, (list, tuple)):
        return ".".join(str(part) for part in field)
    return str(field)


class SpecError(RDIQError):
    """Raised on the first invalid local JSON specification entry."""

    code = "E_SPEC"


class GraphError(RDIQError):
    """Raised when the reference graph is incomplete or cyclic."""

    code = "E_GRAPH"


class ReplayError(RDIQError):
    """Raised on the first ledger-vs-recomputed mismatch during replay."""

    code = "E_REPLAY"


class LedgerError(RDIQError):
    """Raised on corrupt or unusable ledger state."""

    code = "E_LEDGER"
