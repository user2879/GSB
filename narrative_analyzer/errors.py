"""Business errors and stable error-code definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


E_SPEC_INVALID = "E_SPEC_INVALID"
E_GRAPH_CYCLE = "E_GRAPH_CYCLE"
E_SELECTION_UNKNOWN = "E_SELECTION_UNKNOWN"
E_PRECISION_BOUND = "E_PRECISION_BOUND"
E_PATH_DIVERGE = "E_PATH_DIVERGE"
E_INJECT_CONFIG = "E_INJECT_CONFIG"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"

ERROR_CODES = (
    E_SPEC_INVALID,
    E_GRAPH_CYCLE,
    E_SELECTION_UNKNOWN,
    E_PRECISION_BOUND,
    E_PATH_DIVERGE,
    E_INJECT_CONFIG,
    E_SNAPSHOT_STALE,
)

EXIT_CODES = {
    E_SPEC_INVALID: 2,
    E_GRAPH_CYCLE: 3,
    E_SELECTION_UNKNOWN: 4,
    E_PRECISION_BOUND: 5,
    E_PATH_DIVERGE: 6,
    E_INJECT_CONFIG: 7,
    E_SNAPSHOT_STALE: 8,
}


@dataclass
class BusinessError(Exception):
    """An expected, serializable domain failure.

    ``details`` must contain only deterministic JSON-compatible values so the
    same failure can be replayed into byte-identical reports.
    """

    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unknown business error code: {self.code}")
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "business_code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }

