"""Structured business errors used by the command line and UI."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


@dataclass
class DomainError(Exception):
    """A user-visible business error with stable fields."""

    code: str
    message: str
    category: str | None = None
    page_id: int | None = None
    field: str | None = None
    event_seq: int | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)
    exit_code: int = 1

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)
        if self.category is None:
            self.category = self.code

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "category": self.category,
            "message": self.message,
            "page_id": self.page_id,
            "field": self.field,
            "event_seq": self.event_seq,
            "details": self.details,
        }


class EPage(DomainError):
    def __init__(
        self,
        message: str,
        *,
        page_id: int | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="E_PAGE",
            category="E_PAGE",
            message=message,
            page_id=page_id,
            field=field,
            details=details or {},
            exit_code=10,
        )


class ENullMap(DomainError):
    """A null bitmap fault. Its aggregate category remains E_PAGE."""

    def __init__(
        self,
        message: str,
        *,
        page_id: int | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="E_NULLMAP",
            category="E_PAGE",
            message=message,
            page_id=page_id,
            field=field,
            details=details or {},
            exit_code=11,
        )


class EBudget(DomainError):
    def __init__(self, budget: int, minimum_feasible: int = 1) -> None:
        super().__init__(
            code="E_BUDGET",
            category="E_BUDGET",
            message=(
                f"page cache slot budget must be >= {minimum_feasible}; "
                f"received {budget}"
            ),
            field="budget",
            details={
                "budget": budget,
                "minimum_feasible_slots": minimum_feasible,
            },
            exit_code=12,
        )


class ERoute(DomainError):
    def __init__(
        self,
        message: str,
        *,
        page_id: int | None = None,
        event_seq: int | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="E_ROUTE",
            category="E_ROUTE",
            message=message,
            page_id=page_id,
            event_seq=event_seq,
            field=field,
            details=details or {},
            exit_code=13,
        )


class EReplay(DomainError):
    def __init__(
        self,
        message: str,
        *,
        page_id: int | None = None,
        event_seq: int | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="E_REPLAY",
            category="E_REPLAY",
            message=message,
            page_id=page_id,
            event_seq=event_seq,
            field=field,
            details=details or {},
            exit_code=14,
        )
