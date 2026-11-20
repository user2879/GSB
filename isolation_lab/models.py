from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable

from .errors import BusinessError


DEFAULT_ISOLATION_LEVEL = "read_committed"
DEFAULT_LOCK_WAIT_BUDGET_MS = 1000
DEFAULT_CONCURRENCY = 1
DEFAULT_OPERATION_TIMEOUT_MS = 1000
CURRENT_FORMAT_VERSION = 3
ISOLATION_LEVELS = ("read_committed", "repeatable_read")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Operation:
    op_no: int
    kind: str
    row_key: str
    value: Any = None
    has_value: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "op_no": self.op_no,
            "type": self.kind,
            "row_key": self.row_key,
        }
        if self.has_value:
            result["value"] = self.value
        return result


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    operations: tuple[Operation, ...]
    isolation_level: str
    lock_wait_budget_ms: int
    concurrency: int
    operation_timeout_ms: int
    lock_order: tuple[str, ...]

    @property
    def row_keys(self) -> tuple[str, ...]:
        return tuple(sorted({op.row_key for op in self.operations}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "isolation_level": self.isolation_level,
            "lock_wait_budget_ms": self.lock_wait_budget_ms,
            "concurrency": self.concurrency,
            "operation_timeout_ms": self.operation_timeout_ms,
            "lock_order": list(self.lock_order),
            "operations": [op.to_dict() for op in self.operations],
        }


@dataclass
class EditScript:
    script_id: str
    source_version: int
    normalized_version: int
    initial_rows: dict[str, Any]
    sessions: tuple[SessionSpec, ...]
    description: str | None = None
    upgrade_changes: tuple[str, ...] = ()

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "script_id": self.script_id,
            "initial_rows": dict(sorted(self.initial_rows.items())),
            "sessions": [session.to_dict() for session in self.sessions],
        }

    @property
    def semantic_digest(self) -> str:
        return digest_json(self.semantic_dict())

    @property
    def source_digest(self) -> str:
        return digest_json(self.to_dict(include_source=True))

    def to_dict(self, *, include_source: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format_version": self.normalized_version,
            "script_id": self.script_id,
            "initial_rows": dict(sorted(self.initial_rows.items())),
            "sessions": [session.to_dict() for session in self.sessions],
        }
        if self.description is not None:
            result["description"] = self.description
        if include_source:
            result["source_format_version"] = self.source_version
            if self.upgrade_changes:
                result["upgrade_changes"] = list(self.upgrade_changes)
        return result

    def normalized_for_snapshot(self) -> dict[str, Any]:
        return self.to_dict(include_source=False)

    def replace_sessions(self, sessions: Iterable[SessionSpec]) -> "EditScript":
        return replace(self, sessions=tuple(sessions))


@dataclass(frozen=True)
class SchedulerParams:
    global_concurrency_limit: int = 2
    clock_quantum_ms: int = 1
    parameter_version: int = CURRENT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if type(self.global_concurrency_limit) is not int or self.global_concurrency_limit < 1:
            raise BusinessError(
                "E_SCRIPT",
                "global_concurrency_limit must be an integer >= 1",
                path="$.scheduler.global_concurrency_limit",
            )
        if type(self.clock_quantum_ms) is not int or self.clock_quantum_ms < 1:
            raise BusinessError(
                "E_SCRIPT",
                "clock_quantum_ms must be an integer >= 1",
                path="$.scheduler.clock_quantum_ms",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_concurrency_limit": self.global_concurrency_limit,
            "clock_quantum_ms": self.clock_quantum_ms,
            "parameter_version": self.parameter_version,
        }

    @property
    def digest(self) -> str:
        return digest_json(self.to_dict())


@dataclass
class WaitEvidence:
    request_seq: int
    row_key: str
    mode: str
    started_virtual_ms: int
    granted_virtual_ms: int | None = None
    wait_ms: int = 0
    blocked_by: tuple[str, ...] = ()
    queue_position: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperationResult:
    op_no: int
    kind: str
    row_key: str
    status: str
    value: Any = None
    data_version: int | None = None
    grant_seq: int | None = None
    wait_ms: int = 0
    monotonic_start_ms: int = 0
    monotonic_end_ms: int = 0
    business_code: str | None = None
    message: str | None = None
    wait: WaitEvidence | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["wait"] = self.wait.to_dict() if self.wait else None
        return result


@dataclass
class SessionOutcome:
    session_id: str
    start_seq: int
    isolation_level: str
    concurrency_limit: int
    started_virtual_ms: int
    finished_virtual_ms: int
    started_monotonic_ms: int
    finished_monotonic_ms: int
    outcome: str
    business_code: str | None = None
    message: str | None = None
    wait_used_ms: int = 0
    committed_data_version: int | None = None
    operations: list[OperationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "start_seq": self.start_seq,
            "isolation_level": self.isolation_level,
            "concurrency_limit": self.concurrency_limit,
            "started_virtual_ms": self.started_virtual_ms,
            "finished_virtual_ms": self.finished_virtual_ms,
            "started_monotonic_ms": self.started_monotonic_ms,
            "finished_monotonic_ms": self.finished_monotonic_ms,
            "outcome": self.outcome,
            "business_code": self.business_code,
            "message": self.message,
            "wait_used_ms": self.wait_used_ms,
            "committed_data_version": self.committed_data_version,
            "operations": [operation.to_dict() for operation in self.operations],
        }


@dataclass
class TraceEvent:
    event_seq: int
    virtual_ms: int
    monotonic_ms: int
    session_id: str | None
    op_no: int | None
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "virtual_ms": self.virtual_ms,
            "monotonic_ms": self.monotonic_ms,
            "session_id": self.session_id,
            "op_no": self.op_no,
            "event_type": self.event_type,
            "payload": self.payload,
        }

    def deterministic_dict(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "virtual_ms": self.virtual_ms,
            "session_id": self.session_id,
            "op_no": self.op_no,
            "event_type": self.event_type,
            "payload": self.payload,
        }


@dataclass
class CommitRecord:
    commit_seq: int
    session_id: str
    data_version: int
    virtual_ms: int
    writes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    script_id: str
    source_version: int
    normalized_version: int
    script_digest: str
    params: SchedulerParams
    run_id: str
    initial_rows: dict[str, Any]
    final_rows: dict[str, dict[str, Any]]
    global_data_version: int
    outcomes: list[SessionOutcome]
    trace: list[TraceEvent]
    commits: list[CommitRecord]
    lock_state: dict[str, Any]
    started_monotonic_ms: int
    finished_monotonic_ms: int

    def outcome_by_session(self) -> dict[str, SessionOutcome]:
        return {outcome.session_id: outcome for outcome in self.outcomes}

    def summary_dict(self) -> dict[str, Any]:
        return {
            "script_id": self.script_id,
            "source_version": self.source_version,
            "normalized_version": self.normalized_version,
            "script_digest": self.script_digest,
            "params": self.params.to_dict(),
            "params_version": self.params.parameter_version,
            "params_digest": self.params.digest,
            "run_id": self.run_id,
            "global_data_version": self.global_data_version,
            "final_rows": self.final_rows,
            "commits": [commit.to_dict() for commit in self.commits],
            "sessions": [outcome.to_dict() for outcome in self.outcomes],
            "lock_state": self.lock_state,
        }


@dataclass
class RestoreReport:
    snapshot_id: str
    mode: str
    script_digest_match: bool
    param_digest_match: bool
    parameter_version_match: bool
    changed_sessions: list[str]
    changed_parameter_items: list[str]
    cleared_lock_entries: int
    trace_entries: int
    data_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayReport:
    snapshot_id: str
    reproducible: bool
    session_id: str | None = None
    op_no: int | None = None
    field: str | None = None
    expected: Any = None
    actual: Any = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
