"""Stable business codes used by the workbench and its audit reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BUSINESS_CODES: dict[str, dict[str, str]] = {
    "CFG_UNKNOWN_COLUMN": {
        "kind": "configuration",
        "severity": "error",
        "description": "A configured feature column does not exist in the data description.",
    },
    "CFG_TYPE_MISMATCH": {
        "kind": "configuration",
        "severity": "error",
        "description": "A configured expected column type differs from the data description.",
    },
    "CFG_TARGET_STRUCTURE_MISSING": {
        "kind": "configuration",
        "severity": "error",
        "description": "The target structure or its target columns are missing.",
    },
    "CFG_CLUSTER_ZERO": {
        "kind": "configuration",
        "severity": "error",
        "description": "A clustering candidate requests zero clusters/components.",
    },
    "CFG_CONCURRENCY_NONPOSITIVE": {
        "kind": "configuration",
        "severity": "error",
        "description": "The configured thread-pool concurrency is not positive.",
    },
    "CFG_GLOBAL_BUDGET_NONPOSITIVE": {
        "kind": "configuration",
        "severity": "error",
        "description": "The configured global iteration budget is not positive.",
    },
    "CFG_PARAMETER_OUT_OF_RANGE": {
        "kind": "configuration",
        "severity": "error",
        "description": "A candidate hyperparameter is outside its supported range.",
    },
    "DATA_NON_FINITE_VALUE": {
        "kind": "data",
        "severity": "warning",
        "description": "A non-finite generated feature was isolated and imputed.",
    },
    "PREPROCESS_NO_FEATURES": {
        "kind": "preprocess",
        "severity": "error",
        "description": "No usable feature columns remained after validation.",
    },
    "SCHED_GLOBAL_BUDGET_EXHAUSTED": {
        "kind": "scheduler",
        "severity": "error",
        "description": "No global iteration budget remained for a candidate.",
    },
    "SCHED_CANDIDATE_QUOTA_HIT": {
        "kind": "scheduler",
        "severity": "warning",
        "description": "A candidate reached its admitted iteration quota before convergence.",
    },
    "SCHED_GLOBAL_BUDGET_TRUNCATED": {
        "kind": "scheduler",
        "severity": "warning",
        "description": "A candidate was truncated by the final global budget grant.",
    },
    "CANDIDATE_RUNTIME_ERROR": {
        "kind": "runtime",
        "severity": "error",
        "description": "A candidate failed in isolation; other candidates continued.",
    },
    "METRIC_UNDEFINED": {
        "kind": "evaluation",
        "severity": "warning",
        "description": "A metric is undefined for the candidate label structure.",
    },
    "EVAL_EMPTY_CLUSTER": {
        "kind": "evaluation",
        "severity": "warning",
        "description": "The candidate contains at least one empty cluster/component.",
    },
    "EVAL_ALL_NOISE": {
        "kind": "evaluation",
        "severity": "warning",
        "description": "DBSCAN classified every sample as noise.",
    },
    "SNAPSHOT_VERSION_INVALID": {
        "kind": "snapshot",
        "severity": "warning",
        "description": "A snapshot version, config hash, or injection hash changed; old state was not resumed.",
    },
    "SNAPSHOT_STAGE_MISSING": {
        "kind": "snapshot",
        "severity": "warning",
        "description": "A requested snapshot stage is absent or unreadable.",
    },
    "INJ_NON_FINITE_FEATURE": {
        "kind": "injection",
        "severity": "warning",
        "description": "Injected non-finite sample feature was detected and imputed in isolation.",
    },
    "INJ_HYPERPARAMETER_OUT_OF_RANGE": {
        "kind": "injection",
        "severity": "error",
        "description": "An injected candidate hyperparameter was rejected without stopping other candidates.",
    },
    "INJ_THRESHOLD_BOUNDARY_EQUAL": {
        "kind": "injection",
        "severity": "warning",
        "description": "An evaluation threshold was exercised exactly at the candidate score boundary.",
    },
    "INJ_DATA_SIZE_OVER_LIMIT": {
        "kind": "injection",
        "severity": "warning",
        "description": "Injected row count exceeded the configured limit and was capped in isolation.",
    },
}


@dataclass(order=False)
class Failure:
    code: str
    stage: str
    message: str
    item: str = ""
    severity: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in BUSINESS_CODES:
            raise ValueError(f"Unknown business code: {self.code}")
        if self.severity is None:
            self.severity = BUSINESS_CODES[self.code]["severity"]

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        return (self.stage, self.item, self.code, self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "kind": BUSINESS_CODES[self.code]["kind"],
            "severity": self.severity,
            "stage": self.stage,
            "item": self.item,
            "message": self.message,
            "details": self.details,
        }


def sort_failures(failures: list[Failure]) -> list[Failure]:
    """Stable failure ordering independent of thread completion order."""
    return sorted(failures, key=lambda failure: failure.sort_key)


def code_catalog() -> list[dict[str, str]]:
    return [
        {"code": code, **meta}
        for code, meta in sorted(BUSINESS_CODES.items(), key=lambda item: item[0])
    ]
