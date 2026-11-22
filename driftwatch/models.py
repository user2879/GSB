from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

METRICS: tuple[str, ...] = ("cpu_quota", "memory_pages", "processes")
METRIC_INDEX = {metric: index for index, metric in enumerate(METRICS)}
AGGREGATION_KINDS = ("median", "mean_floor", "trimmed_mean", "minimum", "maximum")
BOUNDARY_RULES = ("tolerance_floor", "conservative_reject")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_json(value: Any) -> str:
    payload = canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class HostSpec:
    host_id: str
    expected: Mapping[str, int]
    tolerance_bps: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "host_id": self.host_id,
            "expected": {metric: self.expected[metric] for metric in METRICS},
            "tolerance_bps": {metric: self.tolerance_bps[metric] for metric in METRICS},
        }


@dataclass(frozen=True)
class Snapshot:
    seq: int
    parent_seq: int | None
    hosts: Mapping[str, HostSpec]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "parent_seq": self.parent_seq,
            "hosts": {host: self.hosts[host].to_dict() for host in sorted(self.hosts)},
        }


@dataclass(frozen=True)
class SnapshotChain:
    snapshots: Mapping[int, Snapshot]

    def ordered(self) -> tuple[Snapshot, ...]:
        return tuple(self.snapshots[seq] for seq in sorted(self.snapshots))

    def host_ids(self) -> tuple[str, ...]:
        hosts: set[str] = set()
        for snapshot in self.snapshots.values():
            hosts.update(snapshot.hosts)
        return tuple(sorted(hosts))

    def to_dict(self) -> dict[str, Any]:
        return {"snapshots": [self.snapshots[seq].to_dict() for seq in sorted(self.snapshots)]}

    def version(self) -> str:
        return digest_json(self.to_dict())


@dataclass(frozen=True)
class AggregationSpec:
    kind: str
    window: int
    version: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "window": self.window, "version": self.version}


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    host: str
    port: int
    seed: int
    aggregation: AggregationSpec
    hosts: tuple[str, ...]
    metrics: tuple[str, ...]
    per_agent_concurrency: int
    profiles: Mapping[str, Mapping[str, Mapping[str, int]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "host": self.host,
            "port": self.port,
            "seed": self.seed,
            "aggregation": self.aggregation.to_dict(),
            "hosts": list(self.hosts),
            "metrics": list(self.metrics),
            "per_agent_concurrency": self.per_agent_concurrency,
            "profiles": {
                host: {
                    metric: {
                        "base": int(self.profiles[host][metric]["base"]),
                        "spread": int(self.profiles[host][metric]["spread"]),
                    }
                    for metric in sorted(self.profiles[host])
                }
                for host in sorted(self.profiles)
            },
        }

    def covers(self, host_id: str, metric_id: str) -> bool:
        return host_id in self.hosts and metric_id in self.metrics


@dataclass(frozen=True)
class Criteria:
    global_concurrency: int
    per_agent_concurrency: int
    min_sources: int
    scale_exp: int
    socket_timeout_ms: int
    handshake_timeout_ms: int
    max_retries: int
    retry_backoff_ms: int
    boundary_rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_concurrency": self.global_concurrency,
            "per_agent_concurrency": self.per_agent_concurrency,
            "min_sources": self.min_sources,
            "scale_exp": self.scale_exp,
            "socket_timeout_ms": self.socket_timeout_ms,
            "handshake_timeout_ms": self.handshake_timeout_ms,
            "max_retries": self.max_retries,
            "retry_backoff_ms": self.retry_backoff_ms,
            "boundary_rule": self.boundary_rule,
        }

    def scale_version(self) -> str:
        return digest_json(
            {
                "scale_exp": self.scale_exp,
                "rounding": "floor_for_nonnegative_truncate_toward_zero_for_signed",
                "boundary_rule": self.boundary_rule,
            }
        )


@dataclass(frozen=True)
class Observation:
    agent_id: str
    observation_seq: int
    host_id: str
    metric_id: str
    value: int
    aggregation_version: str
    monotonic_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "observation_seq": self.observation_seq,
            "host_id": self.host_id,
            "metric_id": self.metric_id,
            "value": self.value,
            "aggregation_version": self.aggregation_version,
            "monotonic_ms": self.monotonic_ms,
        }


@dataclass(frozen=True)
class CollectionTask:
    agent: AgentSpec
    snapshot_seq: int
    host_id: str
    metric_id: str
    observation_seq: int

    @property
    def task_key(self) -> str:
        return f"s{self.snapshot_seq}:{self.host_id}:{self.metric_id}:{self.agent.agent_id}"

    @property
    def request_id(self) -> str:
        return f"snapshot-{self.snapshot_seq}/{self.host_id}/{self.metric_id}"


@dataclass
class AttemptRecord:
    batch_id: str
    task_key: str
    agent_id: str
    snapshot_seq: int
    host_id: str
    metric_id: str
    observation_seq: int
    attempt_no: int
    is_retry: bool
    wait_start_ms: int
    wait_end_ms: int
    run_start_ms: int
    run_end_ms: int
    status: str
    error_code: str | None = None
    message: str = ""
    value: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "task_key": self.task_key,
            "agent_id": self.agent_id,
            "snapshot_seq": self.snapshot_seq,
            "host_id": self.host_id,
            "metric_id": self.metric_id,
            "observation_seq": self.observation_seq,
            "attempt_no": self.attempt_no,
            "is_retry": self.is_retry,
            "wait_start_ms": self.wait_start_ms,
            "wait_end_ms": self.wait_end_ms,
            "run_start_ms": self.run_start_ms,
            "run_end_ms": self.run_end_ms,
            "status": self.status,
            "error_code": self.error_code,
            "message": self.message,
            "value": self.value,
        }


@dataclass
class MergedResult:
    snapshot_seq: int
    host_id: str
    metric_id: str
    status: str
    authority_agent: str | None
    authority_observation_seq: int | None
    authority_value: int | None
    aggregation_version: str | None
    source_count: int
    sparse: bool
    sources: list[dict[str, Any]]
    max_abs_deviation: int
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_seq": self.snapshot_seq,
            "host_id": self.host_id,
            "metric_id": self.metric_id,
            "status": self.status,
            "authority_agent": self.authority_agent,
            "authority_observation_seq": self.authority_observation_seq,
            "authority_value": self.authority_value,
            "aggregation_version": self.aggregation_version,
            "source_count": self.source_count,
            "sparse": self.sparse,
            "sources": self.sources,
            "max_abs_deviation": self.max_abs_deviation,
            "error_code": self.error_code,
        }


@dataclass
class ErrorBound:
    observation_quantization: int
    tolerance_comparison: int
    merge_deviation: int
    composite: int
    unit: str = "fixed"
    saturated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_quantization": self.observation_quantization,
            "tolerance_comparison": self.tolerance_comparison,
            "merge_deviation": self.merge_deviation,
            "composite": self.composite,
            "unit": self.unit,
            "saturated": self.saturated,
        }


@dataclass
class DriftResult:
    snapshot_seq: int
    host_id: str
    metric_id: str
    status: str
    decision: str
    expected: int | None
    observed: int | None
    tolerance_bps: int | None
    drift_signed_fixed: int | None
    drift_excess_fixed: int | None
    drift_ratio_bps: int | None
    allowed_fixed: int | None
    error_bound: ErrorBound | None
    error_affected: bool
    boundary_rule: str
    saturated: bool
    error_code: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_seq": self.snapshot_seq,
            "host_id": self.host_id,
            "metric_id": self.metric_id,
            "status": self.status,
            "decision": self.decision,
            "expected": self.expected,
            "observed": self.observed,
            "tolerance_bps": self.tolerance_bps,
            "drift_signed_fixed": self.drift_signed_fixed,
            "drift_excess_fixed": self.drift_excess_fixed,
            "drift_ratio_bps": self.drift_ratio_bps,
            "allowed_fixed": self.allowed_fixed,
            "error_bound": self.error_bound.to_dict() if self.error_bound else None,
            "error_affected": self.error_affected,
            "boundary_rule": self.boundary_rule,
            "saturated": self.saturated,
            "error_code": self.error_code,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AffectedItem:
    snapshot_seq: int
    host_id: str
    metric_id: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_seq": self.snapshot_seq,
            "host_id": self.host_id,
            "metric_id": self.metric_id,
            "reasons": list(self.reasons),
        }


def agents_version(agents: Iterable[AgentSpec]) -> str:
    return digest_json([agent.to_dict() for agent in sorted(agents, key=lambda item: item.agent_id)])


def declaration_digest(chain: SnapshotChain, agents: Iterable[AgentSpec], criteria: Criteria) -> str:
    return digest_json(
        {
            "snapshots": chain.to_dict(),
            "agents": [agent.to_dict() for agent in sorted(agents, key=lambda item: item.agent_id)],
            "criteria": criteria.to_dict(),
        }
    )
