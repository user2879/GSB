"""Business codes and the failure object used across the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_EVIDENCE_JSON = "E_EVIDENCE_JSON"
E_EVIDENCE_SCHEMA = "E_EVIDENCE_SCHEMA"
E_WORKORDER_ID = "E_WORKORDER_ID"
E_EVIDENCE_SEQ = "E_EVIDENCE_SEQ"
E_DUPLICATE_EVIDENCE_KEY = "E_DUPLICATE_EVIDENCE_KEY"
E_UNKNOWN_EVIDENCE_TYPE = "E_UNKNOWN_EVIDENCE_TYPE"
E_UNKNOWN_SOURCE_ROLE = "E_UNKNOWN_SOURCE_ROLE"
E_EVENT_TIME_INVALID = "E_EVENT_TIME_INVALID"
E_CONFIDENCE_RANGE = "E_CONFIDENCE_RANGE"
E_TERMS_EMPTY = "E_TERMS_EMPTY"

E_RULE_JSON = "E_RULE_JSON"
E_RULE_SCHEMA = "E_RULE_SCHEMA"
E_WEIGHT_RANGE = "E_WEIGHT_RANGE"
E_ROLE_RANGE = "E_ROLE_RANGE"
E_RELATION_SCHEMA = "E_RELATION_SCHEMA"
E_RELATION_DUPLICATE_ID = "E_RELATION_DUPLICATE_ID"
E_RELATION_UNKNOWN_TYPE = "E_RELATION_UNKNOWN_TYPE"
E_HALF_LIFE_ZERO = "E_HALF_LIFE_ZERO"
E_HALF_LIFE_INVALID = "E_HALF_LIFE_INVALID"
E_THRESHOLD_INVALID = "E_THRESHOLD_INVALID"

E_CONFIG_JSON = "E_CONFIG_JSON"
E_CONFIG_SCHEMA = "E_CONFIG_SCHEMA"
E_CONFIG_CONCURRENCY = "E_CONFIG_CONCURRENCY"
E_CONFIG_BATCH_BUDGET = "E_CONFIG_BATCH_BUDGET"
E_CONFIG_WORKORDER_QUOTA = "E_CONFIG_WORKORDER_QUOTA"
E_CONFIG_FIXED_WIDTH = "E_CONFIG_FIXED_WIDTH"
E_CONFIG_QUANTIZATION = "E_CONFIG_QUANTIZATION"
E_CONFIG_SNAPSHOT_INTERVAL = "E_CONFIG_SNAPSHOT_INTERVAL"
E_CONFIG_NUMERIC = "E_CONFIG_NUMERIC"
E_INPUT_IO = "E_INPUT_IO"
E_INJECTION_SCHEMA = "E_INJECTION_SCHEMA"

E_WORKORDER_BUDGET_EXHAUSTED = "E_WORKORDER_BUDGET_EXHAUSTED"
E_EVIDENCE_BUDGET_PENDING = "E_EVIDENCE_BUDGET_PENDING"
E_SNAPSHOT_CONFIG_MISMATCH = "E_SNAPSHOT_CONFIG_MISMATCH"
E_SNAPSHOT_VERSION_MISMATCH = "E_SNAPSHOT_VERSION_MISMATCH"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_SNAPSHOT_REBUILD_REQUIRED = "E_SNAPSHOT_REBUILD_REQUIRED"
E_SNAPSHOT_IO = "E_SNAPSHOT_IO"
E_EDIT_COMMAND = "E_EDIT_COMMAND"
E_INTERNAL = "E_INTERNAL"


STAGE_ORDER = {
    "global": 0,
    "config": 1,
    "rules": 2,
    "load": 3,
    "decay": 4,
    "relations": 5,
    "aggregation": 6,
    "scheduler": 7,
    "snapshot": 8,
}


@dataclass
class BusinessFault(Exception):
    """A deterministic business failure that can safely enter the report."""

    code: str
    message: str
    stage: str
    workorder_id: str | None = None
    evidence_seq: int | None = None
    relation_id: str | None = None
    injection_id: str | None = None
    exception_type: str = "BusinessFault"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def as_failure(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "workorder_id": self.workorder_id or "<GLOBAL>",
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "exception_type": self.exception_type,
        }
        optional = {
            "evidence_seq": self.evidence_seq,
            "relation_id": self.relation_id,
            "injection_id": self.injection_id,
        }
        for key, value in optional.items():
            if value is not None:
                result[key] = value
        if self.details:
            result["details"] = self.details
        return result


def failure_sort_key(failure: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(failure.get("workorder_id", "")),
        STAGE_ORDER.get(str(failure.get("stage", "")), 99),
        str(failure.get("code", "")),
        failure.get("evidence_seq") if failure.get("evidence_seq") is not None else -1,
        str(failure.get("relation_id", "")),
        str(failure.get("injection_id", "")),
    )


def sorted_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(failures, key=failure_sort_key)
