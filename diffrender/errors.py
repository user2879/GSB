"""Business errors for the incremental differentiable renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_SCENE_INVALID = "E_SCENE_INVALID"
E_GRAD_MISMATCH = "E_GRAD_MISMATCH"
E_DEPENDENCY_STALE = "E_DEPENDENCY_STALE"
E_CLOSURE_MISMATCH = "E_CLOSURE_MISMATCH"
E_SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"
E_TILE_HASH = "E_TILE_HASH"

ERROR_CODES = (
    E_SCENE_INVALID,
    E_GRAD_MISMATCH,
    E_DEPENDENCY_STALE,
    E_CLOSURE_MISMATCH,
    E_SNAPSHOT_VERSION,
    E_TILE_HASH,
)


@dataclass
class DiffRenderError(RuntimeError):
    """A stable, machine-readable failure in the rendering pipeline."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}

