"""Business error codes and deterministic issue collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


class Codes:
    JSON_INVALID = "E_JSON_INVALID"
    SCHEMA_INVALID = "E_SCHEMA_INVALID"
    QUERY_EMPTY = "E_QUERY_EMPTY"
    QUERY_ID_DUPLICATE = "E_QUERY_ID_DUPLICATE"
    QUERY_ID_INVALID = "E_QUERY_ID_INVALID"
    QUERY_TERM_UNKNOWN = "E_QUERY_TERM_UNKNOWN"
    QUERY_TERMS_INVALID = "E_QUERY_TERMS_INVALID"
    SLOT_RANGE = "E_SLOT_RANGE"
    SLOT_TYPE = "E_SLOT_TYPE"
    SLOT_NAME_INVALID = "E_SLOT_NAME_INVALID"
    CANDIDATES_INVALID = "E_CANDIDATES_INVALID"
    CONTEXT_INVALID = "E_CONTEXT_INVALID"
    PARENT_MISSING = "E_PARENT_MISSING"
    PARENT_INVALID = "E_PARENT_INVALID"
    REVISION_CYCLE = "E_REVISION_CYCLE"
    REVISION_DUPLICATE = "E_REVISION_DUPLICATE"
    REVISION_ROOT_INVALID = "E_REVISION_ROOT_INVALID"
    REVISION_OP_INVALID = "E_REVISION_OP_INVALID"
    TRIGGER_ID_INVALID = "E_TRIGGER_ID_INVALID"
    TRIGGER_TERM_UNKNOWN = "E_TRIGGER_TERM_UNKNOWN"
    TRIGGER_WEIGHT_INVALID = "E_TRIGGER_WEIGHT_INVALID"
    TRIGGER_WEIGHT_ENUM = "E_TRIGGER_WEIGHT_ENUM"
    TRIGGER_SLOT_PATTERN_INVALID = "E_TRIGGER_SLOT_PATTERN_INVALID"
    FRACTION_BITS_ENUM = "E_FRACTION_BITS_ENUM"
    QUANTUM_ENUM = "E_QUANTUM_ENUM"
    THRESHOLD_INVALID = "E_THRESHOLD_INVALID"
    CONFIG_INVALID = "E_CONFIG_INVALID"
    EDIT_OP_UNKNOWN = "E_EDIT_OP_UNKNOWN"
    EDIT_TARGET_MISSING = "E_EDIT_TARGET_MISSING"
    EDIT_PAYLOAD_INVALID = "E_EDIT_PAYLOAD_INVALID"
    ACTIVE_REVISION_MISSING = "E_ACTIVE_REVISION_MISSING"
    ACTIVE_REVISION_INVALID = "E_ACTIVE_REVISION_INVALID"
    PENDING_BUDGET = "E_PENDING_MATCH_BUDGET"
    PENDING_SOURCE_QUOTA = "E_PENDING_SOURCE_QUOTA"
    PENDING_BATCH_LIMIT = "E_PENDING_BATCH_LIMIT"
    SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"
    SNAPSHOT_CONCURRENT_COMMIT = "E_SNAPSHOT_CONCURRENT_COMMIT"
    SNAPSHOT_INTEGRITY = "E_SNAPSHOT_INTEGRITY"
    TASK_FAILED = "E_TASK_FAILED"


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    scope: str = ""
    entity_id: str = ""
    retained: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "scope": self.scope,
            "entity_id": self.entity_id,
            "retained": self.retained,
            "details": self.details,
        }

    def sort_key(self) -> tuple[str, str, str, str]:
        return (self.code, self.scope, self.entity_id, self.message)


class IssueCollector:
    def __init__(self, issues: Iterable[Issue] | None = None) -> None:
        self._issues = list(issues or [])

    def add(
        self,
        code: str,
        message: str,
        *,
        scope: str = "",
        entity_id: str = "",
        retained: bool = True,
        details: dict[str, Any] | None = None,
    ) -> Issue:
        issue = Issue(
            code=code,
            message=message,
            scope=scope,
            entity_id=entity_id,
            retained=retained,
            details=details or {},
        )
        self._issues.append(issue)
        return issue

    def extend(self, issues: Iterable[Issue]) -> None:
        self._issues.extend(issues)

    def __iter__(self):
        return iter(self._issues)

    def __len__(self) -> int:
        return len(self._issues)

    def sorted(self) -> list[Issue]:
        return sorted(self._issues, key=lambda issue: issue.sort_key())

    def to_dicts(self) -> list[dict[str, Any]]:
        return [issue.to_dict() for issue in self.sorted()]

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for issue in self._issues:
            result[issue.code] = result.get(issue.code, 0) + 1
        return dict(sorted(result.items()))

