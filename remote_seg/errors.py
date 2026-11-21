"""Business error codes and compact exception types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_INDEX_INVALID = "E_INDEX_INVALID"
E_TILE_SHAPE_MISMATCH = "E_TILE_SHAPE_MISMATCH"
E_BAND_VERSION_UNKNOWN = "E_BAND_VERSION_UNKNOWN"
E_TILE_DECODE = "E_TILE_DECODE"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_FUSION_VERSION_UNKNOWN = "E_FUSION_VERSION_UNKNOWN"
E_PATCH_BOUNDS = "E_PATCH_BOUNDS"
E_RECOMPUTE_BUDGET_PENDING = "E_RECOMPUTE_BUDGET_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_FUSION_DEPENDENCY_FAILED = "E_FUSION_DEPENDENCY_FAILED"
E_INTERNAL = "E_INTERNAL"


@dataclass(slots=True)
class Failure:
    code: str
    stage: str
    message: str
    row: int | None = None
    col: int | None = None
    tile_id: str | None = None
    injection_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
        }
        if self.row is not None:
            value["row"] = self.row
        if self.col is not None:
            value["col"] = self.col
        if self.tile_id is not None:
            value["tile_id"] = self.tile_id
        if self.injection_id is not None:
            value["injection_id"] = self.injection_id
        if self.details:
            value["details"] = self.details
        return value


class SegmentationError(Exception):
    """Exception carrying a stable business code and structured context."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str = "validation",
        row: int | None = None,
        col: int | None = None,
        tile_id: str | None = None,
        injection_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.row = row
        self.col = col
        self.tile_id = tile_id
        self.injection_id = injection_id
        self.details = details or {}

    def failure(self) -> Failure:
        return Failure(
            code=self.code,
            stage=self.stage,
            message=self.message,
            row=self.row,
            col=self.col,
            tile_id=self.tile_id,
            injection_id=self.injection_id,
            details=self.details,
        )
