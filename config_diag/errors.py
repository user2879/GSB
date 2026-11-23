from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


@dataclass
class DiagnosticError(Exception):
    """A structured, user-actionable diagnostic.

    The first four fields are intentionally stable because they are consumed by
    the CLI, JSONL logs, tests, and the Tkinter status area.
    """

    code: str
    detail: str
    key: str | None = None
    field: str | None = None
    context: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.detail)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "detail": self.detail,
        }
        if self.key is not None:
            payload["key"] = self.key
        if self.field is not None:
            payload["field"] = self.field
        if self.context:
            payload["context"] = self.context
        return payload


def spec_error(detail: str, key: str | None, field: str | None, **context: Any) -> DiagnosticError:
    return DiagnosticError("E_SPEC", detail, key=key, field=field, context=context)


def cycle_error(detail: str, key: str | None, cycle: list[str]) -> DiagnosticError:
    return DiagnosticError(
        "E_CYCLE",
        detail,
        key=key,
        field="refs",
        # The requirement names both "E_SPEC overall" and the explicit cycle
        # injection code E_CYCLE.  Keeping the category makes both contracts
        # machine-visible without weakening the specific diagnosis.
        context={"spec_category": "E_SPEC", "cycle": cycle},
    )
