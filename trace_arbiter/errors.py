from __future__ import annotations

from typing import Any


EXIT_CODES = {
    "E_SPEC": 2,
    "E_PORT": 3,
    "E_SPAWN": 4,
    "E_HANDSHAKE": 5,
    "E_DIVERGE": 6,
    "E_REPLAY": 7,
    "E_SNAPSHOT": 8,
    "E_CHAIN": 9,
    "E_TIMEOUT": 10,
    "E_PROTOCOL": 11,
    "E_RECOVERY": 12,
    "E_IO": 13,
    "E_UI": 14,
}


class ArbiterError(Exception):
    """A structured, user-facing error.

    Required error classes use their fixed ``E_*`` code. Optional location
    fields make the first failing item easy to find in logs and in CLI output.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
        trace_id: str | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if code not in EXIT_CODES:
            raise ValueError(f"unknown error code: {code}")
        self.code = code
        self.message = message
        self.path = path
        self.line = line
        self.trace_id = trace_id
        self.field = field
        self.details = details or {}

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.code]

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"ok": False, "code": self.code, "message": self.message}
        for name in ("path", "line", "trace_id", "field"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.details:
            result["details"] = self.details
        return result


def spec_error(
    message: str,
    *,
    path: str,
    line: int | None = None,
    field: str | None = None,
    trace_id: str | None = None,
) -> ArbiterError:
    return ArbiterError(
        "E_SPEC",
        message,
        path=path,
        line=line,
        field=field,
        trace_id=trace_id,
    )
