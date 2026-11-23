"""Structured diagnostics used by every workbench layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_LAYER_INVALID = "E_LAYER_INVALID"
E_SHAPE_MISMATCH = "E_SHAPE_MISMATCH"
E_WEIGHT_NOT_FINITE = "E_WEIGHT_NOT_FINITE"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_VERSION_NOT_MONOTONIC = "E_VERSION_NOT_MONOTONIC"
E_RESULT_STALE = "E_RESULT_STALE"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_SNAPSHOT_VERSION_MISMATCH = "E_SNAPSHOT_VERSION_MISMATCH"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_IO = "E_IO"


@dataclass(slots=True)
class Issue:
    code: str
    message: str
    command_index: int | None = None
    line_number: int | None = None
    layer: str | None = None
    sample_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "command_index": self.command_index,
            "line_number": self.line_number,
            "layer": self.layer,
            "sample_id": self.sample_id,
            "details": self.details,
        }


class WorkbenchError(Exception):
    """An expected, user-facing workbench failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        command_index: int | None = None,
        line_number: int | None = None,
        layer: str | None = None,
        sample_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.issue = Issue(
            code=code,
            message=message,
            command_index=command_index,
            line_number=line_number,
            layer=layer,
            sample_id=sample_id,
            details=details or {},
        )

    @property
    def code(self) -> str:
        return self.issue.code

    def to_dict(self) -> dict[str, Any]:
        return self.issue.to_dict()


def issue_sort_key(issue: Issue | dict[str, Any]) -> tuple[Any, ...]:
    if isinstance(issue, Issue):
        data = issue.to_dict()
    else:
        data = issue
    return (
        data.get("command_index") if data.get("command_index") is not None else -1,
        data.get("layer") or "",
        data.get("line_number") if data.get("line_number") is not None else -1,
        data.get("sample_id") or "",
        data.get("code") or "",
    )
