from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


BUSINESS_CODES = {
    "OK",
    "E_SPEC",
    "E_FIELD",
    "E_WINDOW",
    "E_VERSION",
    "E_REPLAY",
    "E_VERIFY",
    "E_STATE",
    "E_IO",
}


@dataclass
class PipelineError(Exception):
    code: str
    message: str
    line: int | None = None
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in BUSINESS_CODES or self.code == "OK":
            raise ValueError(f"invalid error business code: {self.code}")
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "business_code": self.code,
            "message": self.message,
        }
        if self.line is not None:
            result["line"] = self.line
        if self.field is not None:
            result["field"] = self.field
        if self.details:
            result["details"] = self.details
        return result

    def __str__(self) -> str:
        location = []
        if self.line is not None:
            location.append(f"line={self.line}")
        if self.field is not None:
            location.append(f"field={self.field}")
        suffix = f" ({', '.join(location)})" if location else ""
        return f"{self.code}: {self.message}{suffix}"


def first_error(errors: list[PipelineError]) -> PipelineError | None:
    if not errors:
        return None
    return min(
        errors,
        key=lambda error: (
            error.line if error.line is not None else 2**63 - 1,
            error.field or "",
            error.code,
            error.message,
        ),
    )
