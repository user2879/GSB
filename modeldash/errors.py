"""Typed errors whose codes are part of the command-line contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WorkbenchError(Exception):
    code: str
    field: str
    detail: str
    context: dict[str, Any] | None = None

    def __str__(self) -> str:
        suffix = f" ({self.field})" if self.field else ""
        return f"{self.code}{suffix}: {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "field": self.field,
            "detail": self.detail,
        }
        if self.context:
            value["context"] = self.context
        return value


class SpecError(WorkbenchError):
    def __init__(self, field: str, detail: str) -> None:
        super().__init__("E_SPEC", field, detail)


class DispatchError(WorkbenchError):
    def __init__(
        self,
        code: str,
        node_id: str,
        detail: str,
        *,
        retryable: bool = True,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, f"node:{node_id}", detail, context)
        self.node_id = node_id
        self.retryable = retryable


class LedgerError(WorkbenchError):
    pass
