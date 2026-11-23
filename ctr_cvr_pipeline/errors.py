from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    request_id: str | None = None
    item_id: str | None = None
    sample_id: str | None = None
    field: str | None = None
    command_index: int | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)

    def sort_key(self) -> tuple[str, str, str, str, str, int]:
        return (
            self.code,
            self.request_id or "",
            self.item_id or "",
            self.field or "",
            self.sample_id or "",
            self.command_index if self.command_index is not None else -1,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.sample_id is not None:
            result["sample_id"] = self.sample_id
        if self.request_id is not None:
            result["request_id"] = self.request_id
        if self.item_id is not None:
            result["item_id"] = self.item_id
        if self.field is not None:
            result["field"] = self.field
        if self.command_index is not None:
            result["command_index"] = self.command_index
        if self.details:
            result["details"] = self.details
        return result


def sample_identity(request_id: Any, item_id: Any, line_no: int | None = None) -> str:
    req = str(request_id) if request_id not in (None, "") else "?"
    item = str(item_id) if item_id not in (None, "") else "?"
    suffix = f":line{line_no}" if line_no is not None else ""
    return f"{req}:{item}{suffix}"


class ConfigError(ValueError):
    def __init__(self, code: str, field: str, message: str):
        super().__init__(f"{code} {field}: {message}")
        self.code = code
        self.field = field
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "field": self.field, "message": self.message}
