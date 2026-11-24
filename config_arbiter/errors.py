from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppError(Exception):
    code: str
    field: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"{self.code} at {self.field}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "field": self.field,
            "message": self.message,
        }
        if self.context:
            payload["context"] = self.context
        return payload


class SpecError(AppError):
    def __init__(self, field: str, message: str, **context: Any) -> None:
        super().__init__("E_SPEC", field, message, context)


class ProtocolError(AppError):
    def __init__(
        self, code: str, field: str, message: str, **context: Any
    ) -> None:
        super().__init__(code, field, message, context)


class ReplayError(AppError):
    def __init__(
        self,
        *,
        key: str,
        field: str,
        message: str,
        view_id: str | None = None,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        context: dict[str, Any] = {"key": key, "field": field}
        if view_id is not None:
            context["view_id"] = view_id
        if expected is not None:
            context["expected"] = expected
        if actual is not None:
            context["actual"] = actual
        super().__init__("E_REPLAY", field, message, context)

