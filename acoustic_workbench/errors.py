"""Business error codes shared by the engine, CLI, and diagnostic service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_DEPENDENCY_STALE = "E_DEPENDENCY_STALE"
E_SPH_ORDER = "E_SPH_ORDER"
E_GRID_BOUNDS = "E_GRID_BOUNDS"
E_TILE_HASH = "E_TILE_HASH"
E_PARAM_RANGE = "E_PARAM_RANGE"
E_SNAPSHOT_CONFLICT = "E_SNAPSHOT_CONFLICT"

ERROR_CODES = (
    E_DEPENDENCY_STALE,
    E_SPH_ORDER,
    E_GRID_BOUNDS,
    E_TILE_HASH,
    E_PARAM_RANGE,
    E_SNAPSHOT_CONFLICT,
)


@dataclass
class WorkbenchError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


def fail(code: str, message: str, **details: Any) -> None:
    raise WorkbenchError(code=code, message=message, details=details)
