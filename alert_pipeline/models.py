"""Internal data model (normalized alerts, aggregates, candidates, batches)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .util import canonical_bytes, canonical_digest, stable_id

SEVERITIES = ("critical", "major", "minor", "warning", "info")
SCHEMA_VERSIONS = ("v1", "v2", "v3")


@dataclass(frozen=True)
class NormalizedAlert:
    """Version-independent internal representation.

    ``source_version`` records the originating schema version and ``filled`` is
    the ordered, de-duplicated set of fields synthesized during normalization,
    each with an explicit, traceable fill rule.
    """

    alert_id: str
    source_version: str
    timestamp_ms: int
    service_id: str
    severity: str
    fingerprint: str
    silence_window_ms: int
    labels: tuple[tuple[str, str], ...]
    root_cause_refs: tuple[str, ...]
    strategy_id: str
    filled: tuple[str, ...]
    line: int

    def canonical(self) -> bytes:
        return canonical_bytes(self.to_record())

    def canonical_digest(self) -> str:
        return canonical_digest(self.to_record())

    def to_record(self) -> dict[str, Any]:
        # ``line`` is provenance, not normalization content: two alerts with
        # identical logical content normalize to identical bytes regardless of
        # the row order they appeared in.
        return {
            "alert_id": self.alert_id,
            "source_version": self.source_version,
            "timestamp_ms": self.timestamp_ms,
            "service_id": self.service_id,
            "severity": self.severity,
            "fingerprint": self.fingerprint,
            "silence_window_ms": self.silence_window_ms,
            "labels": [list(kv) for kv in self.labels],
            "root_cause_refs": list(self.root_cause_refs),
            "strategy_id": self.strategy_id,
            "filled": list(self.filled),
        }

    def to_wire(self) -> dict[str, Any]:
        rec = self.to_record()
        rec["line"] = self.line
        return rec


@dataclass(frozen=True)
class AggregateAlert:
    aggregate_id: str
    service_id: str
    fingerprint: str
    first_ts_ms: int
    last_ts_ms: int
    suppressed_count: int
    member_lines: tuple[int, ...]
    silence_window_ms: int
    window_version: int
    strategy_id: str
    schema_versions: tuple[str, ...]
    member_ids: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("member_lines", "root_cause_refs", "labels",
                    "root_cause_refs", "schema_versions", "member_ids"):
            if key in d:
                d[key] = list(d[key])
        return d

    @staticmethod
    def build_id(service_id: str, fingerprint: str, first_ts_ms: int) -> str:
        return stable_id("agg", [service_id, fingerprint, first_ts_ms])


@dataclass(frozen=True)
class RootCandidate:
    service_id: str
    direction: str          # "self" | "upstream" | "downstream"
    layer: int              # dependency distance from the alerting service
    path: tuple[str, ...]   # services traversed from alerting service

    def to_record(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "direction": self.direction,
            "layer": self.layer,
            "path": list(self.path),
        }


@dataclass(frozen=True)
class Correlation:
    aggregate_id: str
    rule_version: int
    candidates: tuple[RootCandidate, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "aggregate_id": self.aggregate_id,
            "rule_version": self.rule_version,
            "candidates": [c.to_record() for c in self.candidates],
        }
