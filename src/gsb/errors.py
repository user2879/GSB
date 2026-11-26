"""Stable error codes for the GSB event-bus subscription pipeline."""


class GSBError(Exception):
    """Base class carrying a stable machine-readable ``code``."""

    code = "E_GENERIC"

    def __init__(self, message, *, code=None, entry=None, field=None, **context):
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message
        self.entry = entry
        self.field = field
        self.context = context

    def to_dict(self):
        data = {"code": self.code, "message": self.message}
        if self.entry is not None:
            data["entry"] = self.entry
        if self.field is not None:
            data["field"] = self.field
        if self.context:
            data["context"] = self.context
        return data


class SpecError(GSBError):
    code = "E_SPEC"


class FormatError(GSBError):
    code = "E_FORMAT"


class BackfillError(GSBError):
    code = "E_BACKFILL"


class PortError(GSBError):
    code = "E_PORT"


class BusTimeoutError(GSBError):
    code = "E_TIMEOUT"


class LeaseError(GSBError):
    code = "E_LEASE"
