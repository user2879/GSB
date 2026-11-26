"""Diagnostics are kept as data so they can be counted and audited."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    source: str = ""
    entry_index: int | None = None
    field_name: str | None = None
    injection_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "source": self.source,
        }
        if self.entry_index is not None:
            value["entry_index"] = self.entry_index
        if self.field_name is not None:
            value["field"] = self.field_name
        if self.injection_id is not None:
            value["injection_id"] = self.injection_id
        if self.details:
            value["details"] = self.details
        return value


class DiagnosticBag:
    """Small deterministic collector shared by all validation layers."""

    def __init__(self) -> None:
        self._items: list[Diagnostic] = []

    def add(
        self,
        code: str,
        message: str,
        *,
        source: str = "",
        entry_index: int | None = None,
        field_name: str | None = None,
        injection_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> Diagnostic:
        item = Diagnostic(
            code=code,
            message=message,
            source=source,
            entry_index=entry_index,
            field_name=field_name,
            injection_id=injection_id,
            details=details or {},
        )
        self._items.append(item)
        return item

    def extend(self, items: Iterable[Diagnostic]) -> None:
        self._items.extend(items)

    @property
    def items(self) -> tuple[Diagnostic, ...]:
        return tuple(self._items)

    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(item.code for item in self._items).items()))

    def injection_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item.injection_id
                    for item in self._items
                    if item.injection_id is not None
                }
            )
        )

    def to_dicts(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]

    def __len__(self) -> int:
        return len(self._items)

