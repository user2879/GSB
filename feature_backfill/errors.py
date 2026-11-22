"""Business error codes and deterministic failure records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

E_ROW_INVALID = "E_ROW_INVALID"
E_FEATURE_DAG_INVALID = "E_FEATURE_DAG_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_PLAN_INVALID = "E_PLAN_INVALID"
E_LEASE_CONFLICT = "E_LEASE_CONFLICT"
E_LEASE_EXPIRED = "E_LEASE_EXPIRED"
E_LEASE_LOST = "E_LEASE_LOST"
E_LEASE_WAIT_TIMEOUT = "E_LEASE_WAIT_TIMEOUT"
E_BATCH_QUOTA_EXHAUSTED = "E_BATCH_QUOTA_EXHAUSTED"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_SNAPSHOT_MISSING = "E_SNAPSHOT_MISSING"
E_CACHE_CAPACITY_INVALID = "E_CONFIG_INVALID"
E_INTERNAL = "E_INTERNAL"

STAGE_ORDER = {
    "load_validation": 0,
    "feature_dag": 1,
    "config": 2,
    "plan": 3,
    "lease": 4,
    "cache": 5,
    "window_features": 6,
    "entity_summary": 7,
    "quota": 8,
    "snapshot": 9,
    "runtime": 10,
}


@dataclass
class AppError(Exception):
    code: str
    message: str
    stage: str = "runtime"
    entity_id: str | None = None
    row_id: str | None = None
    feature: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        where = "/".join(
            part for part in (self.entity_id, self.row_id, self.feature) if part
        )
        return f"{self.code}: {self.message}" + (f" [{where}]" if where else "")

    def to_failure(self) -> dict[str, Any]:
        return make_failure(
            self.code,
            self.message,
            stage=self.stage,
            entity_id=self.entity_id,
            row_id=self.row_id,
            feature=self.feature,
            details=self.details,
        )


def make_failure(
    code: str,
    message: str,
    *,
    stage: str,
    entity_id: str | None = None,
    row_id: str | None = None,
    feature: str | None = None,
    injection_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "code": code,
        "stage": stage,
        "message": message,
    }
    if entity_id is not None:
        record["entity_id"] = entity_id
    if row_id is not None:
        record["row_id"] = row_id
    if feature is not None:
        record["feature"] = feature
    if injection_id is not None:
        record["injection_id"] = injection_id
    if details:
        record["details"] = dict(details)
    return record


def failure_sort_key(failure: Mapping[str, Any]) -> tuple[Any, ...]:
    details = failure.get("details") or {}
    return (
        STAGE_ORDER.get(str(failure.get("stage", "runtime")), 99),
        str(failure.get("entity_id", "")),
        str(failure.get("row_id", "")),
        str(failure.get("feature", "")),
        str(failure.get("code", "")),
        str(failure.get("injection_id", "")),
        str(details.get("line", "")),
        canonical_text(failure),
    )


def sorted_failures(failures: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in sorted(failures, key=failure_sort_key)]


def canonical_text(value: Any) -> str:
    import json

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
