"""Stable business error codes used by the diagnostics tool."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


E_VIEW_UNKNOWN = "E_VIEW_UNKNOWN"
E_PARAM_INVALID = "E_PARAM_INVALID"
E_VOXEL_ALIGN = "E_VOXEL_ALIGN"
E_RECON_BUDGET_EXCEEDED = "E_RECON_BUDGET_EXCEEDED"
E_PNG_INVALID = "E_PNG_INVALID"
E_GUI_UNAVAILABLE = "E_GUI_UNAVAILABLE"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    stage: str = "general"
    view: str | None = None
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnostic(
    code: str,
    message: str,
    *,
    stage: str = "general",
    view: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return Diagnostic(code, message, stage, view, detail).to_dict()
