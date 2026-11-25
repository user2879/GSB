from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E_INDEX_INPUT = "E_INDEX_INPUT"
E_QUERY_BUDGET_EXCEEDED = "E_QUERY_BUDGET_EXCEEDED"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"


@dataclass(frozen=True)
class Failure:
    """A stable, serialisable business failure record."""

    code: str
    source: str
    line: int
    entity_id: str
    message: str
    detail_code: str = ""
    context: dict[str, Any] | None = None

    def sort_key(self) -> tuple[str, int, str, str, str, str]:
        return (
            self.source,
            self.line,
            self.entity_id,
            self.code,
            self.detail_code,
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "code": self.code,
            "source": self.source,
            "line": self.line,
            "entity_id": self.entity_id,
            "message": self.message,
        }
        if self.detail_code:
            value["detail_code"] = self.detail_code
        if self.context:
            value["context"] = self.context
        return value


class InputValidationError(Exception):
    def __init__(self, failures: list[Failure]):
        self.failures = sorted(failures, key=Failure.sort_key)
        super().__init__("; ".join(item.message for item in self.failures))

