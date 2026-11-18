"""Business errors with stable codes and structured context."""

from __future__ import annotations

from typing import Any

E_SAMPLE = "E_SAMPLE"
E_DEPTH = "E_DEPTH"
E_VERSION = "E_VERSION"
E_REPLAY = "E_REPLAY"
E_PARAM = "E_PARAM"
E_IO = "E_IO"


class SlowProfError(Exception):
    """A domain error suitable for CLI and JSONL logging.

    ``family`` allows an error to have a narrower business code while still
    belonging to a broader contract.  In particular frame-depth errors use
    ``E_DEPTH`` and family ``E_SAMPLE``: the whole sample load is rejected by
    the sample contract, and the narrower code identifies the failed field.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        line_no: int | None = None,
        family: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.line_no = line_no
        self.family = family
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.family is not None:
            result["family"] = self.family
        if self.line_no is not None:
            result["line"] = self.line_no
        if self.field is not None:
            result["field"] = self.field
        if self.details:
            result["details"] = self.details
        return result


class SampleValidationError(SlowProfError):
    """A strict NDJSON validation failure."""

    def __init__(
        self,
        message: str,
        *,
        line_no: int,
        field: str,
        code: str = E_SAMPLE,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code,
            message,
            field=field,
            line_no=line_no,
            family=E_SAMPLE,
            details=details,
        )


class VersionError(SlowProfError):
    def __init__(self, message: str, *, field: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(E_VERSION, message, field=field, details=details)


class ReplayError(SlowProfError):
    def __init__(self, message: str, *, field: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(E_REPLAY, message, field=field, details=details)


class ParameterError(SlowProfError):
    def __init__(self, message: str, *, field: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(E_PARAM, message, field=field, details=details)

