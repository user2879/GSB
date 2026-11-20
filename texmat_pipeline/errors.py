"""Business error codes and the structured failure list type."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

E_TEX_DECODE = "E_TEX_DECODE"
E_LAYER_INVALID = "E_LAYER_INVALID"
E_WEIGHT_INVALID = "E_WEIGHT_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"

ALL_CODES = (
    E_TEX_DECODE,
    E_LAYER_INVALID,
    E_WEIGHT_INVALID,
    E_CONFIG_INVALID,
    E_POINT_BUDGET_PENDING,
    E_SNAPSHOT_STALE,
)


@dataclass
class Failure:
    """One failed unit of work. Processing of other units continues."""

    code: str
    target: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
