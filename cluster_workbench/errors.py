from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .util import canonical_json


# Stable business codes.  Additional detail belongs in context/reason instead
# of creating unbounded code variants.
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_COLUMN_UNKNOWN = "E_COLUMN_UNKNOWN"
E_TYPE_MISMATCH = "E_TYPE_MISMATCH"
E_CLUSTER_COUNT_ZERO = "E_CLUSTER_COUNT_ZERO"
E_CLUSTER_COUNT_RANGE = "E_CLUSTER_COUNT_RANGE"
E_EPS_NON_POSITIVE = "E_EPS_NON_POSITIVE"
E_CONCURRENCY_NON_POSITIVE = "E_CONCURRENCY_NON_POSITIVE"
E_GLOBAL_BUDGET_NON_POSITIVE = "E_GLOBAL_BUDGET_NON_POSITIVE"
E_LATENCY_BUDGET_NON_POSITIVE = "E_LATENCY_BUDGET_NON_POSITIVE"
E_NONFINITE_FEATURE = "E_NONFINITE_FEATURE"
E_COLUMN_DESCRIPTION_MISSING = "E_COLUMN_DESCRIPTION_MISSING"
E_CAND_BUDGET_EXHAUSTED = "E_CAND_BUDGET_EXHAUSTED"
E_GLOBAL_BUDGET_PENDING = "E_GLOBAL_BUDGET_PENDING"
E_RESULT_STALE = "E_RESULT_STALE"
E_EMPTY_CLUSTER = "E_EMPTY_CLUSTER"
E_ALL_NOISE = "E_ALL_NOISE"
E_ALL_MISSING_TRAIN = "E_ALL_MISSING_TRAIN"
E_UNKNOWN_CATEGORY = "E_UNKNOWN_CATEGORY"
E_MAX_ITER_REACHED = "E_MAX_ITER_REACHED"
E_COVARIANCE_SINGULAR = "E_COVARIANCE_SINGULAR"
E_METRIC_UNAVAILABLE = "E_METRIC_UNAVAILABLE"
E_UNKNOWN_COMMAND = "E_UNKNOWN_COMMAND"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_SNAPSHOT_VERSION_MISMATCH = "E_SNAPSHOT_VERSION_MISMATCH"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_CACHE_CORRUPT = "E_CACHE_CORRUPT"
E_INJECTION_INVALID = "E_INJECTION_INVALID"
E_DATA_INVALID = "E_DATA_INVALID"
E_PIPELINE_ERROR = "E_PIPELINE_ERROR"


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    severity: str = "error"
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "context": _jsonable_context(self.context),
        }

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        context = _jsonable_context(self.context)
        candidate = str(context.get("candidate", ""))
        return (candidate, self.code, self.severity, canonical_json(context))


def _jsonable_context(context: dict[str, Any]) -> dict[str, Any]:
    # Importing here avoids a dependency cycle at module import time.
    from .util import to_jsonable

    value = to_jsonable(context)
    return value if isinstance(value, dict) else {"value": value}


def issue_dicts(issues: Iterable[Issue]) -> list[dict[str, Any]]:
    return [issue.to_dict() for issue in sorted(issues, key=lambda item: item.sort_key)]


def has_error(issues: Iterable[Issue], code: str | None = None) -> bool:
    return any(
        issue.severity == "error" and (code is None or issue.code == code)
        for issue in issues
    )
