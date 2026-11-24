"""Business error codes and deterministic issue collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_REQUEST_INVALID = "E_REQUEST_INVALID"
E_RECEIPT_MALFORMED = "E_RECEIPT_MALFORMED"
E_RECEIPT_MISSING = "E_RECEIPT_MISSING"
E_UNKNOWN_PARAM_VERSION = "E_UNKNOWN_PARAM_VERSION"
E_BATCH_VERSION_CONFLICT = "E_BATCH_VERSION_CONFLICT"
E_BATCH_WORKER_CONFLICT = "E_BATCH_WORKER_CONFLICT"
E_DUPLICATE_RANK = "E_DUPLICATE_RANK"
E_COVERAGE_REF_INVALID = "E_COVERAGE_REF_INVALID"
E_COVERAGE_CYCLE = "E_COVERAGE_CYCLE"
E_LEDGER_DIVERGED = "E_LEDGER_DIVERGED"
E_RESPONSE_BUDGET_EXCEEDED = "E_RESPONSE_BUDGET_EXCEEDED"
E_FAULT_MISSED = "E_FAULT_MISSED"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_CONVERSION_LOSS = "E_CONVERSION_LOSS"


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    path: str = ""
    line: int = 0
    item: str = ""
    detail_code: str = ""
    injection_id: str = ""
    context: dict[str, Any] = field(default_factory=dict, compare=False)

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.path,
            self.line,
            self.code,
            self.detail_code,
            self.item,
            self.message,
            self.injection_id,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "line": self.line,
            "item": self.item,
        }
        if self.detail_code:
            result["detail_code"] = self.detail_code
        if self.injection_id:
            result["injection_id"] = self.injection_id
        if self.context:
            result["context"] = _jsonable(self.context)
        return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class IssueCollector:
    """Collects issues; serialization always uses a stable total order."""

    def __init__(self) -> None:
        self._issues: list[Issue] = []

    def add(self, issue: Issue) -> None:
        self._issues.append(issue)

    def extend(self, issues: Iterable[Issue]) -> None:
        self._issues.extend(issues)

    def __len__(self) -> int:
        return len(self._issues)

    def __iter__(self):
        return iter(self._issues)

    def sorted(self) -> list[Issue]:
        return sorted(self._issues, key=Issue.sort_key)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [issue.to_dict() for issue in self.sorted()]

    def by_code(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self._issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        return dict(sorted(counts.items()))


def issue_from_dict(data: dict[str, Any]) -> Issue:
    return Issue(
        code=str(data.get("code", E_CONFIG_INVALID)),
        message=str(data.get("message", "")),
        path=str(data.get("path", "")),
        line=int(data.get("line", 0) or 0),
        item=str(data.get("item", "")),
        detail_code=str(data.get("detail_code", "")),
        injection_id=str(data.get("injection_id", "")),
        context=data.get("context", {}) if isinstance(data.get("context", {}), dict) else {},
    )


class ValidationError(ValueError):
    """Raised only when an operation cannot produce a safe result at all."""

    def __init__(self, code: str, message: str, *, path: str = "", item: str = ""):
        super().__init__(message)
        self.code = code
        self.path = path
        self.item = item

    def as_issue(self) -> Issue:
        return Issue(code=self.code, message=str(self), path=self.path, item=self.item)
