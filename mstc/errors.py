"""Business error codes and a structured failure record.

Every validation / runtime failure is represented by one stable code so the
failure list and the diagnostic report can deterministically map a fault to the
business requirement that produced it.  Codes are never raised through the
pipeline for *per-tile* faults: they are collected into a failure list so one
bad tile never interrupts the others.  Only truly fatal, pipeline-level faults
raise :class:`PipelineFatal`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


class PipelineFatal(RuntimeError):
    """Fatal pipeline-level problem (cannot continue at all)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Stable business error codes (kept as plain strings, grouped by stage)
# ---------------------------------------------------------------------------
class Code:
    # loading / manifest / config validation
    E_MANIFEST_PARSE = "E_MANIFEST_PARSE"
    E_CONFIG_PARSE = "E_CONFIG_PARSE"
    E_TILE_ID_DUPLICATE = "E_TILE_ID_DUPLICATE"
    E_BAND_COUNT_MISMATCH = "E_BAND_COUNT_MISMATCH"
    E_BAND_MISSING = "E_BAND_MISSING"
    E_BAND_TRUNCATED = "E_BAND_TRUNCATED"
    E_RESOLUTION_MISMATCH = "E_RESOLUTION_MISMATCH"
    E_ADJ_NOT_SYMMETRIC = "E_ADJ_NOT_SYMMETRIC"
    E_ADJ_DANGLING = "E_ADJ_DANGLING"
    E_BANDSET_UNKNOWN = "E_BANDSET_UNKNOWN"
    E_WINDOW_NOT_DIVISIBLE = "E_WINDOW_NOT_DIVISIBLE"
    E_WINDOW_LARGER_TILE = "E_WINDOW_LARGER_TILE"
    E_WEIGHT_INVALID = "E_WEIGHT_INVALID"
    E_FIXED_ENUM = "E_FIXED_ENUM"
    E_BUDGET_INVALID = "E_BUDGET_INVALID"
    E_CONFIG_KEY_MISSING = "E_CONFIG_KEY_MISSING"
    E_FILE_NOT_FOUND = "E_FILE_NOT_FOUND"

    # runtime / scheduling
    E_TILE_BUDGET_EXHAUSTED = "E_TILE_BUDGET_EXHAUSTED"
    E_PIXEL_BUDGET_EXHAUSTED = "E_PIXEL_BUDGET_EXHAUSTED"
    E_FUSION_BUDGET_PENDING = "E_FUSION_BUDGET_PENDING"
    E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
    E_BOUNDARY_FLIP = "E_BOUNDARY_FLIP"
    E_TILE_COMPUTE = "E_TILE_COMPUTE"


@dataclass
class Failure:
    """A single structured failure record (order-stable)."""

    code: str
    target: str
    message: str
    injection_id: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FailureList:
    """Collects failures while preserving deterministic insertion order."""

    def __init__(self) -> None:
        self._items: list[Failure] = []
        self._seen: set[tuple[str, str, str]] = set()

    def add(
        self,
        code: str,
        target: str,
        message: str,
        injection_id: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> Failure:
        key = (code, target, message)
        if key in self._seen:
            return next(f for f in self._items if (f.code, f.target, f.message) == key)
        record = Failure(code, target, message, injection_id, dict(detail or {}))
        self._items.append(record)
        self._seen.add(key)
        return record

    def codes(self) -> list[str]:
        return [f.code for f in self._items]

    def targets(self) -> list[str]:
        return [f.target for f in self._items]

    def to_dicts(self) -> list[Dict[str, Any]]:
        return [f.to_dict() for f in self._items]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)
