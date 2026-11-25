"""Stable business error codes and a deterministic failure ledger entry."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

E_INPUT_SCHEMA = "E_INPUT_SCHEMA"
E_RECEIPT_MALFORMED = "E_RECEIPT_MALFORMED"
E_ZERO_SCALE = "E_ZERO_SCALE"
E_QUOTA_EXCEEDED = "E_QUOTA_EXCEEDED"
E_BUDGET_PENDING = "E_BUDGET_PENDING"

ALL_CODES = (
    E_INPUT_SCHEMA,
    E_RECEIPT_MALFORMED,
    E_ZERO_SCALE,
    E_QUOTA_EXCEEDED,
    E_BUDGET_PENDING,
)


@dataclass(frozen=True)
class Failure:
    """One non-fatal validation/runtime failure.

    Failures never abort the rest of the run; they are collected into a
    deterministically ordered failure list.
    """

    code: str
    scope: str
    target: str
    message: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sort_key(self) -> tuple[str, str, str, str, str]:
        return (self.code, self.scope, self.target, self.message, self.detail)


class FailureList:
    def __init__(self) -> None:
        self._items: list[Failure] = []

    def add(
        self,
        code: str,
        scope: str,
        target: str,
        message: str,
        detail: str = "",
    ) -> None:
        if code not in ALL_CODES:
            raise ValueError(f"unknown business code: {code}")
        self._items.append(Failure(code, scope, target, message, detail))

    def extend(self, failures: "FailureList") -> None:
        self._items.extend(failures._items)

    def sorted(self) -> list[Failure]:
        return sorted(self._items, key=lambda f: f.sort_key)

    def to_list(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.sorted()]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)
