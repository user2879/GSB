"""Business error codes and isolated-failure records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

E_INPUT_INVALID = "E_INPUT_INVALID"
E_BUDGET_INVALID = "E_BUDGET_INVALID"
E_REQUEST_BUDGET_EXHAUSTED = "E_REQUEST_BUDGET_EXHAUSTED"
E_GLOBAL_BUDGET_PENDING = "E_GLOBAL_BUDGET_PENDING"
E_RESULT_STALE = "E_RESULT_STALE"
E_STAGE_EXCEPTION = "E_STAGE_EXCEPTION"
E_SNAPSHOT_INCOMPATIBLE = "E_SNAPSHOT_INCOMPATIBLE"

STAGE_ORDER = {
    "load": 0,
    "validation": 1,
    "normalize": 2,
    "budget": 3,
    "fusion": 4,
    "rerank": 5,
    "snapshot": 6,
}


@dataclass(frozen=True)
class Failure:
    code: str
    stage: str
    detail: str
    request_id: str | None = None
    item_id: str | None = None
    line: int | None = None
    injection_id: str | None = None

    def sort_key(self) -> tuple[Any, ...]:
        return (
            STAGE_ORDER.get(self.stage, 99),
            self.request_id or "",
            self.item_id or "",
            self.line if self.line is not None else 1 << 60,
            self.code,
            self.detail,
            self.injection_id or "",
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "stage": self.stage,
            "detail": self.detail,
        }
        if self.request_id is not None:
            out["request_id"] = self.request_id
        if self.item_id is not None:
            out["item_id"] = self.item_id
        if self.line is not None:
            out["line"] = self.line
        if self.injection_id is not None:
            out["injection_id"] = self.injection_id
        return out


def sorted_failures(items: list[Failure]) -> list[Failure]:
    return sorted(items, key=lambda f: f.sort_key())
