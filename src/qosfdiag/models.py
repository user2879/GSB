"""Typed domain models used by the policy, engine, ledger and UI layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ShaperPolicy:
    shaper_id: str
    committed_rate_kbps: int
    burst_bytes: int
    queue_limit_packets: int
    match_dst_port_prefix: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "shaper_id": self.shaper_id,
            "committed_rate_kbps": self.committed_rate_kbps,
            "burst_bytes": self.burst_bytes,
            "queue_limit_packets": self.queue_limit_packets,
            "match_dst_port_prefix": self.match_dst_port_prefix,
        }


@dataclass(frozen=True)
class CascadeEdge:
    upstream: str
    downstream: str

    def as_dict(self) -> dict[str, str]:
        return {"upstream": self.upstream, "downstream": self.downstream}


@dataclass(frozen=True)
class PolicySet:
    policy_version: str
    max_committed_rate_kbps: int
    shapers: tuple[ShaperPolicy, ...]
    edges: tuple[CascadeEdge, ...]
    source: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def by_id(self) -> dict[str, ShaperPolicy]:
        return {shaper.shaper_id: shaper for shaper in self.shapers}

    @property
    def roots(self) -> tuple[str, ...]:
        children = {edge.downstream for edge in self.edges}
        return tuple(sorted(s.shaper_id for s in self.shapers if s.shaper_id not in children))

    @property
    def children_by_upstream(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {s.shaper_id: [] for s in self.shapers}
        for edge in self.edges:
            result[edge.upstream].append(edge.downstream)
        return {key: tuple(sorted(values)) for key, values in result.items()}

    @property
    def parent_by_downstream(self) -> dict[str, str]:
        return {edge.downstream: edge.upstream for edge in self.edges}

    def topological_order(self) -> tuple[str, ...]:
        """Return a stable topological order (lexicographic among ready nodes)."""
        indegree = {s.shaper_id: 0 for s in self.shapers}
        children = self.children_by_upstream
        for edge in self.edges:
            indegree[edge.downstream] += 1
        ready = sorted(node for node, degree in indegree.items() if degree == 0)
        result: list[str] = []
        while ready:
            node = ready.pop(0)
            result.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        return tuple(result)

    def chain_order(self) -> tuple[str, ...]:
        return self.topological_order()


@dataclass(frozen=True)
class Packet:
    packet_id: str
    seq: int
    timestamp_ns: int
    size_bytes: int
    dst_port: int
    payload: bytes = field(repr=False, compare=False)

    def manifest(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "seq": self.seq,
            "timestamp_ns": self.timestamp_ns,
            "size_bytes": self.size_bytes,
            "dst_port": self.dst_port,
        }


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int
    count: int
    packet_interval_ns: int
    start_time_ns: int
    min_packet_bytes: int
    max_packet_bytes: int
    dst_port_start: int
    dst_port_span: int
    batch_size: int = 4

    def as_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "count": self.count,
            "packet_interval_ns": self.packet_interval_ns,
            "start_time_ns": self.start_time_ns,
            "min_packet_bytes": self.min_packet_bytes,
            "max_packet_bytes": self.max_packet_bytes,
            "dst_port_start": self.dst_port_start,
            "dst_port_span": self.dst_port_span,
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True)
class DecisionEvent:
    event_seq: int
    batch_no: int
    packet_id: str
    packet_seq: int
    stage_id: str
    kind: str
    decision: str
    time_tick: int
    time_ns: int
    tokens_before: int
    tokens_after: int
    queue_len_before: int
    queue_len_after: int
    delay_ticks: int = 0
    release_tick: int | None = None
    forwarded_to: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "batch_no": self.batch_no,
            "packet_id": self.packet_id,
            "packet_seq": self.packet_seq,
            "stage_id": self.stage_id,
            "kind": self.kind,
            "decision": self.decision,
            "time_tick": self.time_tick,
            "time_ns": self.time_ns,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "queue_len_before": self.queue_len_before,
            "queue_len_after": self.queue_len_after,
            "delay_ticks": self.delay_ticks,
            "release_tick": self.release_tick,
            "forwarded_to": list(self.forwarded_to),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecisionEvent":
        return cls(
            event_seq=int(value["event_seq"]),
            batch_no=int(value["batch_no"]),
            packet_id=str(value["packet_id"]),
            packet_seq=int(value["packet_seq"]),
            stage_id=str(value["stage_id"]),
            kind=str(value["kind"]),
            decision=str(value["decision"]),
            time_tick=int(value["time_tick"]),
            time_ns=int(value["time_ns"]),
            tokens_before=int(value["tokens_before"]),
            tokens_after=int(value["tokens_after"]),
            queue_len_before=int(value["queue_len_before"]),
            queue_len_after=int(value["queue_len_after"]),
            delay_ticks=int(value.get("delay_ticks", 0)),
            release_tick=(None if value.get("release_tick") is None else int(value["release_tick"])),
            forwarded_to=tuple(value.get("forwarded_to", [])),
            reason=str(value.get("reason", "")),
        )


@dataclass(frozen=True)
class StageStats:
    stage_id: str
    passed_packets: int
    queued_packets: int
    dropped_packets: int
    released_packets: int
    forwarded_bytes: int
    max_queue_len: int
    total_delay_ticks: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "passed_packets": self.passed_packets,
            "queued_packets": self.queued_packets,
            "dropped_packets": self.dropped_packets,
            "released_packets": self.released_packets,
            "forwarded_bytes": self.forwarded_bytes,
            "max_queue_len": self.max_queue_len,
            "total_delay_ticks": self.total_delay_ticks,
        }


@dataclass
class RunSummary:
    stage_stats: tuple[StageStats, ...]
    final_released_bytes: int
    final_released_packets: int
    average_delay_ns: int
    total_generated_packets: int
    unmatched_packets: int
    transport_datagrams_sent: int
    transport_datagrams_received: int
    unknown_packet_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_stats": [stats.as_dict() for stats in self.stage_stats],
            "final_released_bytes": self.final_released_bytes,
            "final_released_packets": self.final_released_packets,
            "average_delay_ns": self.average_delay_ns,
            "total_generated_packets": self.total_generated_packets,
            "unmatched_packets": self.unmatched_packets,
            "transport_datagrams_sent": self.transport_datagrams_sent,
            "transport_datagrams_received": self.transport_datagrams_received,
            "unknown_packet_ids": list(self.unknown_packet_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunSummary":
        return cls(
            stage_stats=tuple(
                StageStats(
                    stage_id=str(item["stage_id"]),
                    passed_packets=int(item["passed_packets"]),
                    queued_packets=int(item["queued_packets"]),
                    dropped_packets=int(item["dropped_packets"]),
                    released_packets=int(item["released_packets"]),
                    forwarded_bytes=int(item["forwarded_bytes"]),
                    max_queue_len=int(item["max_queue_len"]),
                    total_delay_ticks=int(item["total_delay_ticks"]),
                )
                for item in value.get("stage_stats", [])
            ),
            final_released_bytes=int(value["final_released_bytes"]),
            final_released_packets=int(value["final_released_packets"]),
            average_delay_ns=int(value["average_delay_ns"]),
            total_generated_packets=int(value["total_generated_packets"]),
            unmatched_packets=int(value["unmatched_packets"]),
            transport_datagrams_sent=int(value["transport_datagrams_sent"]),
            transport_datagrams_received=int(value["transport_datagrams_received"]),
            unknown_packet_ids=tuple(value.get("unknown_packet_ids", [])),
        )


@dataclass(frozen=True)
class ErrorTerm:
    stage_id: str
    rate_quantization_error_bytes_per_s: int
    refill_cumulative_error_units: int
    refill_cumulative_error_bytes_ceiling: int
    delay_quantization_error_ns: int
    bytes_contribution: int
    delay_contribution_ns: int
    bytes_share: float
    delay_share: float
    derivation: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "rate_quantization_error_bytes_per_s": self.rate_quantization_error_bytes_per_s,
            "refill_cumulative_error_units": self.refill_cumulative_error_units,
            "refill_cumulative_error_bytes_ceiling": self.refill_cumulative_error_bytes_ceiling,
            "delay_quantization_error_ns": self.delay_quantization_error_ns,
            "bytes_contribution": self.bytes_contribution,
            "delay_contribution_ns": self.delay_contribution_ns,
            "bytes_share": round(self.bytes_share, 12),
            "delay_share": round(self.delay_share, 12),
            "derivation": dict(sorted(self.derivation.items())),
        }


@dataclass(frozen=True)
class ErrorPropagationResult:
    scale: int
    tick_ns: int
    total_released_bytes: int
    released_packet_count: int
    final_released_bytes: int
    bytes_error_bound: int
    average_delay_ns: int
    average_delay_error_bound_ns: int
    terms: tuple[ErrorTerm, ...]
    dominant_stage: str
    dominance_threshold: float
    dominance_rule: str
    intermediate_terms: dict[str, Any]
    config_fingerprint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "tick_ns": self.tick_ns,
            "total_released_bytes": self.total_released_bytes,
            "released_packet_count": self.released_packet_count,
            "final_released_bytes": self.final_released_bytes,
            "bytes_error_bound": self.bytes_error_bound,
            "average_delay_ns": self.average_delay_ns,
            "average_delay_error_bound_ns": self.average_delay_error_bound_ns,
            "terms": [term.as_dict() for term in self.terms],
            "dominant_stage": self.dominant_stage,
            "dominance_threshold": self.dominance_threshold,
            "dominance_rule": self.dominance_rule,
            "intermediate_terms": self.intermediate_terms,
            "config_fingerprint": self.config_fingerprint,
        }


@dataclass(frozen=True)
class IndistinguishabilityResult:
    indistinguishable: bool
    error_affected: bool
    released_bytes_a: int
    released_bytes_b: int
    released_bytes_difference: int
    combined_bytes_error_bound: int
    average_delay_ns_a: int
    average_delay_ns_b: int
    average_delay_difference_ns: int
    combined_delay_error_bound_ns: int
    winner: str
    stable_order: tuple[str, str]
    decision_basis: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "indistinguishable": self.indistinguishable,
            "error_affected": self.error_affected,
            "released_bytes_a": self.released_bytes_a,
            "released_bytes_b": self.released_bytes_b,
            "released_bytes_difference": self.released_bytes_difference,
            "combined_bytes_error_bound": self.combined_bytes_error_bound,
            "average_delay_ns_a": self.average_delay_ns_a,
            "average_delay_ns_b": self.average_delay_ns_b,
            "average_delay_difference_ns": self.average_delay_difference_ns,
            "combined_delay_error_bound_ns": self.combined_delay_error_bound_ns,
            "winner": self.winner,
            "stable_order": list(self.stable_order),
            "decision_basis": self.decision_basis,
        }
