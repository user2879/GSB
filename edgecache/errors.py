"""Stable business errors used by the CLI and the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ERROR_CODES = frozenset(
    {
        "E_SPEC",
        "E_CONTENT",
        "E_CYCLE",
        "E_PORT",
        "E_TIMEOUT",
        "E_PROTOCOL",
        "E_PROCESS",
        "E_STALE",
        "E_SNAPSHOT",
        "E_REPLAY",
        "E_STATE",
        "E_TK",
    }
)


@dataclass
class EdgeError(Exception):
    """A domain error with deterministic, machine-readable context.

    ``path`` is a content path when one is known.  ``field_path`` is a JSON-like
    location such as ``content[/data/root].ttl_ms``.  Both are deliberately
    stable across processes and runs.
    """

    code: str
    message: str
    path: str | None = None
    field_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    cause: str | None = None

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unknown business error code: {self.code}")
        super().__init__(self.message)

    def envelope(self, **extra: Any) -> dict[str, Any]:
        value: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            value["path"] = self.path
        if self.field_path is not None:
            value["field_path"] = self.field_path
        if self.details:
            value["details"] = self.details
        if self.cause is not None:
            value["cause"] = self.cause
        value.update(extra)
        return value

    @classmethod
    def from_exception(cls, exc: BaseException, *, code: str = "E_PROCESS") -> "EdgeError":
        if isinstance(exc, EdgeError):
            return exc
        return cls(code=code, message=str(exc), cause=type(exc).__name__)


def spec_error(
    message: str,
    *,
    path: str | None = None,
    field_path: str | None = None,
    **details: Any,
) -> EdgeError:
    return EdgeError(
        code="E_SPEC",
        message=message,
        path=path,
        field_path=field_path,
        details=details,
    )
