from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DriftError(Exception):
    code: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    violations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.message:
            self.message = self.code
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": False, "code": self.code, "message": self.message}
        if self.violations:
            payload["violations"] = self.violations
        if self.details:
            payload["details"] = self.details
        return payload

    @classmethod
    def snapshot(cls, violations: list[dict[str, Any]]) -> "DriftError":
        return cls(
            "E_SNAPSHOT",
            "deployment snapshot declarations were rejected",
            violations=violations,
        )

    @classmethod
    def agent(cls, violations: list[dict[str, Any]]) -> "DriftError":
        return cls(
            "E_AGENT",
            "observation agent declarations were rejected",
            violations=violations,
        )

    @classmethod
    def port(cls, message: str, **details: Any) -> "DriftError":
        return cls("E_PORT", message, details=details)

    @classmethod
    def timeout(cls, message: str, **details: Any) -> "DriftError":
        return cls("E_TIMEOUT", message, details=details)

    @classmethod
    def record(cls, message: str, **details: Any) -> "DriftError":
        return cls("E_RECORD", message, details=details)


def violation(path: str, reason: str, actual: Any = None, expected: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"path": path, "reason": reason}
    if actual is not None:
        item["actual"] = actual
    if expected is not None:
        item["expected"] = expected
    return item
