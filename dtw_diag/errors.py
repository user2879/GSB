"""Business error codes and deterministic failure-list helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fingerprints import canonical_json


E_CONFIG = "E_CONFIG"
E_STEP_MODE = "E_STEP_MODE"
E_BANDWIDTH = "E_BANDWIDTH"
E_QUANT_LEVEL = "E_QUANT_LEVEL"
E_MAX_PATH_LENGTH = "E_MAX_PATH_LENGTH"
E_PATH_POLICY = "E_PATH_POLICY"
E_WORKERS = "E_WORKERS"
E_CELL_BUDGET = "E_CELL_BUDGET"
E_SNAPSHOT_INTERVAL = "E_SNAPSHOT_INTERVAL"
E_SEQUENCE_ID = "E_SEQUENCE_ID"
E_DUPLICATE_ID = "E_DUPLICATE_ID"
E_EMPTY_SEQUENCE = "E_EMPTY_SEQUENCE"
E_OBS_VALUE = "E_OBS_VALUE"
E_TIMESTAMP_INTERVAL = "E_TIMESTAMP_INTERVAL"
E_SEGMENT_LENGTH = "E_SEGMENT_LENGTH"
E_SEGMENT_ID = "E_SEGMENT_ID"
E_EDIT_TYPE = "E_EDIT_TYPE"
E_SEQUENCE_NOT_FOUND = "E_SEQUENCE_NOT_FOUND"
E_INDEX_RANGE = "E_INDEX_RANGE"
E_NO_PATH = "E_NO_PATH"
E_PATH_LENGTH_REJECTED = "E_PATH_LENGTH_REJECTED"
E_CELL_BUDGET_PENDING = "E_CELL_BUDGET_PENDING"
E_LOAD = "E_LOAD"

W_PATH_TRUNCATED = "W_PATH_TRUNCATED"


@dataclass(frozen=True)
class Failure:
    code: str
    stage: str
    message: str
    sequence_id: str | None = None
    batch: int | None = None
    edit_index: int | None = None
    record_index: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
        }
        if self.sequence_id is not None:
            out["sequence_id"] = self.sequence_id
        if self.batch is not None:
            out["batch"] = self.batch
        if self.edit_index is not None:
            out["edit_index"] = self.edit_index
        if self.record_index is not None:
            out["record_index"] = self.record_index
        if self.detail:
            out["detail"] = self.detail
        return out


def failure_sort_key(item: Failure) -> tuple[Any, ...]:
    """Canonical order for active failures: sequence id, code, location."""

    return (
        item.sequence_id or "",
        item.code,
        item.stage,
        -1 if item.record_index is None else item.record_index,
        -1 if item.batch is None else item.batch,
        -1 if item.edit_index is None else item.edit_index,
        item.message,
        canonical_json(item.detail),
    )


def sorted_failures(items: list[Failure]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in sorted(items, key=failure_sort_key)]
