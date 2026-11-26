"""Typed error hierarchy for the lease-batch validation system."""


class SpecError(Exception):
    """Base error carrying a stable machine code and optional source line."""

    code = "E_SPEC"

    def __init__(self, message, *, field=None, line=None, code=None):
        self.field = field
        self.line = line
        if code is not None:
            self.code = code
        location = f" line={line}" if line is not None else ""
        field_part = f" field={field}" if field is not None else ""
        super().__init__(f"[{self.code}]{field_part}{location} {message}")


class FieldError(SpecError):
    code = "E_FIELD"


class RankError(SpecError):
    code = "E_RANK"


class RefError(SpecError):
    code = "E_REF"


class LeaseError(SpecError):
    code = "E_LEASE"


class ReplayError(SpecError):
    code = "E_REPLAY"
