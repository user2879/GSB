from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArbiterError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": False, "code": self.code, "message": self.message}
        payload.update(self.details)
        return payload


class SpecError(ArbiterError):
    def __init__(
        self,
        field_name: str,
        line: int,
        message: str,
        *,
        file: str | None = None,
        value: Any = None,
    ) -> None:
        details: dict[str, Any] = {"field": field_name, "line": int(line)}
        if file is not None:
            details["file"] = file
        if value is not None:
            details["value"] = value
        super().__init__("E_SPEC", message, details)


class LeaseError(ArbiterError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_LEASE", message, details)


class StaleError(LeaseError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message, reason="stale", **details)
        self.code = "E_STALE"


class LedgerError(ArbiterError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_LEDGER", message, details)


class ReplayError(ArbiterError):
    def __init__(self, event_seq: int, field_name: str, expected: Any, actual: Any, message: str) -> None:
        super().__init__(
            "E_REPLAY",
            message,
            {
                "event_seq": int(event_seq),
                "field": field_name,
                "expected": expected,
                "actual": actual,
            },
        )


class DivergenceError(ArbiterError):
    def __init__(
        self,
        event_seq: int | None,
        field_name: str,
        sequential: Any,
        batch: Any,
        message: str,
    ) -> None:
        super().__init__(
            "E_DIVERGE",
            message,
            {
                "event_seq": event_seq,
                "field": field_name,
                "sequential": sequential,
                "batch": batch,
            },
        )


class TransportError(ArbiterError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        if code not in {"E_PORT", "E_TIMEOUT", "E_PROTOCOL"}:
            raise ValueError(f"unsupported transport error code: {code}")
        super().__init__(code, message, details)
