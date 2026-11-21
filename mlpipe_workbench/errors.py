"""Business error types and stable serialisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkbenchError(Exception):
    """An error carrying the business code required by the ledger.

    The four fields below are deliberately explicit: the demo and UI must not
    have to infer why an operation failed from free-form exception text.
    """

    code: str
    message: str
    stage_id: str | None = None
    param: str | None = None
    param_value: Any = None
    affected_rows: int | None = None
    location: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "stage_id": self.stage_id,
            "param": self.param,
            "param_value": self.param_value,
            "affected_rows": self.affected_rows,
            "location": self.location,
        }
        if self.details:
            payload["details"] = self.details
        return payload


E_SPEC = "E_SPEC"
E_DATA = "E_DATA"
E_PARAM = "E_PARAM"
E_OUTLIER = "E_OUTLIER"
E_REPLAY = "E_REPLAY"
E_STATE = "E_STATE"
E_ENV = "E_ENV"

BUSINESS_CODES = (E_SPEC, E_DATA, E_PARAM, E_OUTLIER, E_REPLAY, E_STATE, E_ENV)
