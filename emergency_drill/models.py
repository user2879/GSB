"""Domain models and canonical serialization helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable


STATUSES = ("pending", "running", "succeeded", "failed", "skipped")
RECOVERY_STATUS = "unknown"
ALL_VIEW_STATUSES = STATUSES + (RECOVERY_STATUS,)
EVENT_TYPES = ("node_lost", "switchover_failed", "validation_timeout")
FAILURE_POLICIES = ("skip_downstream", "continue_downstream")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StageDefinition:
    seq: int
    name: str
    estimated_seconds: int
    depends_on: tuple[int, ...]
    actions: tuple[str, ...]
    success_criteria: tuple[str, ...]
    failure_policy: str

    @property
    def estimated_ms(self) -> int:
        return self.estimated_seconds * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "name": self.name,
            "estimated_seconds": self.estimated_seconds,
            "depends_on": list(self.depends_on),
            "actions": list(self.actions),
            "success_criteria": list(self.success_criteria),
            "failure_policy": self.failure_policy,
        }


@dataclass(frozen=True)
class EventDefinition:
    event_id: str
    type: str
    offset_ms: int
    stage_seq: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "offset_ms": self.offset_ms,
            "stage_seq": self.stage_seq,
        }


@dataclass(frozen=True)
class Plan:
    plan_id: str
    name: str
    version: int
    failure_policy: str
    stages: tuple[StageDefinition, ...]
    events: tuple[EventDefinition, ...] = ()

    def stage(self, seq: int) -> StageDefinition:
        for item in self.stages:
            if item.seq == seq:
                return item
        raise KeyError(seq)

    @property
    def stage_seqs(self) -> tuple[int, ...]:
        return tuple(item.seq for item in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "version": self.version,
            "failure_policy": self.failure_policy,
            "stages": [item.to_dict() for item in self.stages],
            "events": [item.to_dict() for item in self.events],
        }

    @property
    def digest(self) -> str:
        return digest_json(self.to_dict())


@dataclass
class StageRuntime:
    seq: int
    status: str
    skip_reason: str | None = None
    business_code: str | None = None
    skip_from_stage_seq: int | None = None
    unknown_reason: str | None = None
    updated_ms: int = 0
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "business_code": self.business_code,
            "skip_from_stage_seq": self.skip_from_stage_seq,
            "unknown_reason": self.unknown_reason,
            "updated_ms": self.updated_ms,
        }


@dataclass
class EventRuntime:
    injection_seq: int
    event_id: str
    type: str
    offset_ms: int
    stage_seq: int
    status: str = "pending"
    source: str = "plan"
    stale: bool = False

    def spec_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "offset_ms": self.offset_ms,
            "stage_seq": self.stage_seq,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "injection_seq": self.injection_seq,
            **self.spec_dict(),
            "status": self.status,
            "source": self.source,
        }


@dataclass
class SessionRecord:
    session_id: str
    plan_id: str
    plan_version: int
    plan_digest: str
    clock_ms: int
    status: str
    generation: int


@dataclass
class SessionView:
    session: SessionRecord
    plan: Plan
    stages: list[StageRuntime]
    events: list[EventRuntime]

    def stage_map(self) -> dict[int, StageRuntime]:
        return {item.seq: item for item in self.stages}


@dataclass
class LedgerEntry:
    seq: int
    kind: str
    stage_seq: int | None
    from_status: str | None
    to_status: str | None
    reason: str | None
    code: str | None
    logical_ms: int
    plan_version: int
    details: dict[str, Any] = field(default_factory=dict)
    is_complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "stage_seq": self.stage_seq,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "reason": self.reason,
            "code": self.code,
            "logical_ms": self.logical_ms,
            "plan_version": self.plan_version,
            "details": self.details,
            "is_complete": self.is_complete,
        }


@dataclass
class RecoveryReport:
    session_id: str
    recovered_stage_count: int
    recovered_stage_seqs: list[int]
    unknown_stage_seqs: list[int]
    pending_stage_seqs: list[int]
    incomplete_tail_seqs: list[int]
    tail_action: str
    exit_report_available: bool = False
    notice: str = "强制终止进程不生成退出报告；未知阶段必须重跑。"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "recovered_stage_count": self.recovered_stage_count,
            "recovered_stage_seqs": self.recovered_stage_seqs,
            "unknown_stage_seqs": self.unknown_stage_seqs,
            "pending_stage_seqs": self.pending_stage_seqs,
            "incomplete_tail_seqs": self.incomplete_tail_seqs,
            "tail_action": self.tail_action,
            "exit_report_available": self.exit_report_available,
            "notice": self.notice,
        }


@dataclass
class ReplayReport:
    session_id: str
    snapshot_id: str
    reproducible: bool
    compared_stage_count: int
    comparisons: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "snapshot_id": self.snapshot_id,
            "reproducible": self.reproducible,
            "compared_stage_count": self.compared_stage_count,
            "comparisons": self.comparisons,
        }


def sorted_stage_tuples(items: Iterable[StageDefinition]) -> tuple[StageDefinition, ...]:
    return tuple(sorted(items, key=lambda item: item.seq))
