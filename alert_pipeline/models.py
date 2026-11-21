from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .util import canonical_json, digest_json, sha256_text


KNOWN_LEVELS = ("critical", "warning", "info")


@dataclass(frozen=True)
class ServiceGraph:
    version: int
    services: tuple[str, ...]
    dependencies: tuple[tuple[str, str], ...]
    source_digest: str
    topology: tuple[str, ...] = field(default_factory=tuple)
    topology_rank: dict[str, int] = field(default_factory=dict)
    topology_level: dict[str, int] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "services": list(self.services),
            "dependencies": [list(edge) for edge in self.dependencies],
            "topology": list(self.topology),
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class RootRule:
    rule_id: str
    services: tuple[str, ...]
    directions: tuple[str, ...]
    max_depth: int
    include_self: bool


@dataclass(frozen=True)
class RootRules:
    version: int
    rules: tuple[RootRule, ...]
    source_digest: str
    defaults: dict[str, Any]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "services": list(rule.services),
                    "directions": list(rule.directions),
                    "max_depth": rule.max_depth,
                    "include_self": rule.include_self,
                }
                for rule in self.rules
            ],
            "defaults": dict(self.defaults),
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class Provenance:
    source: str
    rule: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "rule": self.rule}


@dataclass(frozen=True)
class NormalizedAlert:
    line_number: int
    timestamp_ms: int
    service_id: str
    level: str
    fingerprint: str
    tags: dict[str, Any]
    silence_window_ms: int
    root_candidate_refs: tuple[str, ...]
    strategy_id: str
    format_version: int
    declared_version: int
    provenance: dict[str, Provenance]
    raw_digest: str

    @property
    def format_label(self) -> str:
        return f"v{self.format_version}"

    def core_view(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "service_id": self.service_id,
            "level": self.level,
            "fingerprint": self.fingerprint,
            "tags": dict(sorted(self.tags.items())),
        }

    def to_ledger_source(self) -> dict[str, Any]:
        return {
            "line": self.line_number,
            "format_version": self.format_version,
            "format_label": self.format_label,
            "declared_version": self.declared_version,
            "timestamp_ms": self.timestamp_ms,
            "level": self.level,
            "fingerprint": self.fingerprint,
            "silence_window_ms": self.silence_window_ms,
            "strategy_id": self.strategy_id,
            "root_candidate_refs": list(self.root_candidate_refs),
            "provenance": {
                key: value.to_dict()
                for key, value in sorted(self.provenance.items())
            },
        }


@dataclass(frozen=True)
class AggregateAlert:
    alert_id: str
    service_id: str
    level: str
    fingerprint: str
    tags: dict[str, Any]
    first_timestamp_ms: int
    last_timestamp_ms: int
    suppressed_count: int
    total_count: int
    participant_lines: tuple[int, ...]
    silence_window_ms: int
    window_version: str
    strategy_ids: tuple[str, ...]
    source_format_versions: tuple[int, ...]
    source_records: tuple[dict[str, Any], ...]
    declared_format_version: int
    declared_version: int

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "service_id": self.service_id,
            "level": self.level,
            "fingerprint": self.fingerprint,
            "tags": dict(sorted(self.tags.items())),
            "first_timestamp_ms": self.first_timestamp_ms,
            "last_timestamp_ms": self.last_timestamp_ms,
            "suppressed_count": self.suppressed_count,
            "total_count": self.total_count,
            "participant_lines": list(self.participant_lines),
            "silence_window_ms": self.silence_window_ms,
            "window_version": self.window_version,
            "strategy_ids": list(self.strategy_ids),
            "source_format_versions": list(self.source_format_versions),
            "source_records": [dict(item) for item in self.source_records],
            "declared_format_version": self.declared_format_version,
            "declared_version": self.declared_version,
        }

    def content_digest(self) -> str:
        return digest_json(self.to_public_dict())


@dataclass(frozen=True)
class RootCandidate:
    service_id: str
    direction: str
    depth: int
    path: tuple[str, ...]
    evidence: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "direction": self.direction,
            "depth": self.depth,
            "path": list(self.path),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class LedgerEntry:
    alert_id: str
    aggregate: AggregateAlert
    candidates: tuple[RootCandidate, ...]
    rule_id: str
    rule_version: int
    graph_version: int
    declared_version: int
    format_version: int
    declared_root_refs: tuple[str, ...]
    content_digest: str

    def consistency_view(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "aggregate": self.aggregate.to_public_dict(),
            "candidates": [item.to_public_dict() for item in self.candidates],
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "graph_version": self.graph_version,
            "declared_version": self.declared_version,
            "format_version": self.format_version,
            "declared_root_refs": list(self.declared_root_refs),
            "content_digest": self.content_digest,
        }

    def to_public_dict(self) -> dict[str, Any]:
        result = self.consistency_view()
        result["candidate_services"] = [
            candidate.service_id for candidate in self.candidates
        ]
        result["paths"] = [
            {
                "candidate": candidate.service_id,
                "path": list(candidate.path),
                "direction": candidate.direction,
                "depth": candidate.depth,
            }
            for candidate in self.candidates
        ]
        return result


@dataclass(frozen=True)
class TimelineEntry:
    ordinal: int
    task_id: str
    stage: str
    service_id: str
    queued_at_ms: int
    started_at_ms: int
    finished_at_ms: int
    wait_ms: int
    run_ms: int
    outcome: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "task_id": self.task_id,
            "stage": self.stage,
            "service_id": self.service_id,
            "queued_at_ms": self.queued_at_ms,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "wait_ms": self.wait_ms,
            "run_ms": self.run_ms,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class ScheduleReport:
    mode: str
    global_limit: int
    per_service_limit: int
    timeline: tuple[TimelineEntry, ...]
    peak_global: int
    peak_by_service: dict[str, int]
    business_code: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "global_limit": self.global_limit,
            "per_service_limit": self.per_service_limit,
            "timeline": [item.to_public_dict() for item in self.timeline],
            "peak_global": self.peak_global,
            "peak_by_service": dict(sorted(self.peak_by_service.items())),
            "business_code": self.business_code,
        }


@dataclass(frozen=True)
class BatchOutcome:
    batch_index: int
    start_cursor: int
    end_cursor: int
    status: str
    business_code: str
    schedule: ScheduleReport
    summary_digest: str
    conclusion_digest: str = ""
    aggregate_count: int = 0
    ledger_count: int = 0
    normalized_count: int = 0

    def summary_dict(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "start_cursor": self.start_cursor,
            "end_cursor": self.end_cursor,
            "status": self.status,
            "business_code": self.business_code,
            "summary_digest": self.summary_digest,
            "conclusion_digest": self.conclusion_digest,
            "aggregate_count": self.aggregate_count,
            "ledger_count": self.ledger_count,
            "normalized_count": self.normalized_count,
            "peak_global": self.schedule.peak_global,
            "peak_by_service": dict(sorted(self.schedule.peak_by_service.items())),
            "global_limit": self.schedule.global_limit,
            "per_service_limit": self.schedule.per_service_limit,
            "timeline": [item.to_public_dict() for item in self.schedule.timeline],
        }


def build_alert_id(service_id: str, fingerprint: str, first_timestamp_ms: int) -> str:
    material = canonical_json(
        {
            "service_id": service_id,
            "fingerprint": fingerprint,
            "first_timestamp_ms": first_timestamp_ms,
        }
    )
    return "A-" + sha256_text(material)[:20]


def aggregate_from_public_dict(value: dict[str, Any]) -> AggregateAlert:
    return AggregateAlert(
        alert_id=value["alert_id"],
        service_id=value["service_id"],
        level=value["level"],
        fingerprint=value["fingerprint"],
        tags=dict(value.get("tags", {})),
        first_timestamp_ms=int(value["first_timestamp_ms"]),
        last_timestamp_ms=int(value["last_timestamp_ms"]),
        suppressed_count=int(value["suppressed_count"]),
        total_count=int(value["total_count"]),
        participant_lines=tuple(int(item) for item in value["participant_lines"]),
        silence_window_ms=int(value["silence_window_ms"]),
        window_version=value["window_version"],
        strategy_ids=tuple(value.get("strategy_ids", [])),
        source_format_versions=tuple(int(item) for item in value.get("source_format_versions", [])),
        source_records=tuple(dict(item) for item in value.get("source_records", [])),
        declared_format_version=int(value["declared_format_version"]),
        declared_version=int(value["declared_version"]),
    )
