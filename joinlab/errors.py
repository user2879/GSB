from __future__ import annotations

from dataclasses import asdict, dataclass, field as dc_field
from typing import Any


@dataclass
class LabError(Exception):
    """Stable machine-readable error contract used by the CLI and components."""

    code: str
    message: str
    table: str | None = None
    column: str | None = None
    field: str | None = None
    candidate_id: str | None = None
    operator_id: str | None = None
    expected: Any = None
    actual: Any = None
    context: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("context", None)
        if self.context:
            data["context"] = self.context
        return {key: value for key, value in data.items() if value is not None}


def schema_error(
    message: str,
    *,
    table: str | None = None,
    column: str | None = None,
    field: str | None = None,
    **context: Any,
) -> LabError:
    return LabError(
        "E_SCHEMA",
        message,
        table=table,
        column=column,
        field=field,
        context=context,
    )


def replay_error(
    message: str,
    *,
    candidate_id: str | None,
    operator_id: str | None,
    field: str,
    expected: Any,
    actual: Any,
    **context: Any,
) -> LabError:
    return LabError(
        "E_REPLAY",
        message,
        candidate_id=candidate_id,
        operator_id=operator_id,
        field=field,
        expected=expected,
        actual=actual,
        context=context,
    )
