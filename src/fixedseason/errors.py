"""Business error codes and deterministic validation issue records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


E_SERIES_INVALID = "E_SERIES_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_FIT_BUDGET_EXCEEDED = "E_FIT_BUDGET_EXCEEDED"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_FIT_OVERFLOW = "E_FIT_OVERFLOW"
E_PIVOT_UNSTABLE = "E_PIVOT_UNSTABLE"
E_REFERENCE_INVALID = "E_REFERENCE_INVALID"
E_INTERNAL = "E_INTERNAL"

ALL_CODES = {
    E_SERIES_INVALID,
    E_CONFIG_INVALID,
    E_COMMAND_INVALID,
    E_FIT_BUDGET_EXCEEDED,
    E_POINT_BUDGET_PENDING,
    E_FIT_OVERFLOW,
    E_PIVOT_UNSTABLE,
    E_REFERENCE_INVALID,
    E_INTERNAL,
}


@dataclass(frozen=True)
class Issue:
    """A validation or fit issue that is recorded rather than raised."""

    code: str
    message: str
    source: str = ""
    line: int | None = None
    series_id: str | None = None
    stage: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, "")}


class FitError(RuntimeError):
    """A deterministic fit failure carrying a business code."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
