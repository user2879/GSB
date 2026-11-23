"""Fingerprint + silence-window noise reduction (order independent).

Window boundary rule (explicit): within one (service_id, fingerprint) group the
first alert opens a window anchored at ``first_ts_ms``. Every later alert whose
timestamp satisfies ``ts <= anchor + silence_window_ms`` (closed interval) is
suppressed; the first alert with ``ts > anchor + silence_window_ms`` opens a new
window. Counts are integer-accumulated and groups are processed from sorted
input, so the result never depends on dict/group traversal order.

Only *closed* windows are emitted during streaming; open windows remain in
snapshot-able state so a recovered process can continue suppressing correctly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import AggregateAlert, NormalizedAlert
from .rules import RuleSpec


@dataclass
class _OpenAggregate:
    service_id: str
    fingerprint: str
    anchor_ts: int
    last_ts: int
    silence_window_ms: int
    window_version: int
    strategy_id: str
    member_lines: list[int] = field(default_factory=list)
    member_ids: list[str] = field(default_factory=list)
    schema_versions: set[str] = field(default_factory=set)

    def suppressed(self) -> int:
        return len(self.member_lines) - 1

    def seal(self) -> AggregateAlert:
        return AggregateAlert(
            aggregate_id=AggregateAlert.build_id(
                self.service_id, self.fingerprint, self.anchor_ts),
            service_id=self.service_id,
            fingerprint=self.fingerprint,
            first_ts_ms=self.anchor_ts,
            last_ts_ms=self.last_ts,
            suppressed_count=self.suppressed(),
            member_lines=tuple(sorted(self.member_lines)),
            silence_window_ms=self.silence_window_ms,
            window_version=self.window_version,
            strategy_id=self.strategy_id,
            schema_versions=tuple(sorted(self.schema_versions)),
            member_ids=tuple(self.member_ids))

    def to_state(self) -> dict:
        return {
            "service_id": self.service_id,
            "fingerprint": self.fingerprint,
            "anchor_ts": self.anchor_ts,
            "last_ts": self.last_ts,
            "silence_window_ms": self.silence_window_ms,
            "window_version": self.window_version,
            "strategy_id": self.strategy_id,
            "member_lines": list(self.member_lines),
            "member_ids": list(self.member_ids),
            "schema_versions": sorted(self.schema_versions),
        }

    @staticmethod
    def from_state(rec: dict) -> "_OpenAggregate":
        return _OpenAggregate(
            service_id=rec["service_id"],
            fingerprint=rec["fingerprint"],
            anchor_ts=rec["anchor_ts"],
            last_ts=rec["last_ts"],
            silence_window_ms=rec["silence_window_ms"],
            window_version=rec["window_version"],
            strategy_id=rec["strategy_id"],
            member_lines=list(rec["member_lines"]),
            member_ids=list(rec["member_ids"]),
            schema_versions=set(rec["schema_versions"]))


class Deduplicator:
    """Incremental, replay-safe deduplicator. Feed alerts in stream order."""

    def __init__(self, rules: RuleSpec):
        self.rules = rules
        self._open: dict[tuple[str, str], _OpenAggregate] = {}
        self._closed: list[AggregateAlert] = []

    def feed(self, alert: NormalizedAlert) -> list[AggregateAlert]:
        key = (alert.service_id, alert.fingerprint)
        window_ms = self.rules.effective_silence_ms(
            alert.service_id, alert.fingerprint, alert.silence_window_ms)
        current = self._open.get(key)
        if current is None or alert.timestamp_ms > \
                current.anchor_ts + current.silence_window_ms:
            sealed: list[AggregateAlert] = []
            if current is not None:
                sealed_agg = current.seal()
                sealed.append(sealed_agg)
                # Required for restored instances so finalize() never seals a
                # window that was already sealed when it expired.
                self._closed.append(sealed_agg)
            self._open[key] = _OpenAggregate(
                service_id=alert.service_id,
                fingerprint=alert.fingerprint,
                anchor_ts=alert.timestamp_ms,
                last_ts=alert.timestamp_ms,
                silence_window_ms=window_ms,
                window_version=self.rules.window_version,
                strategy_id=alert.strategy_id,
                member_lines=[alert.line],
                member_ids=[alert.alert_id],
                schema_versions={alert.source_version})
            return sealed
        current.last_ts = alert.timestamp_ms
        current.member_lines.append(alert.line)
        current.member_ids.append(alert.alert_id)
        current.schema_versions.add(alert.source_version)
        return []

    def open_state(self) -> list[dict]:
        return [self._open[key].to_state()
                for key in sorted(self._open)]

    def restore_state(self, state: list[dict]) -> None:
        self._open = {}
        for rec in state:
            obj = _OpenAggregate.from_state(rec)
            self._open[(obj.service_id, obj.fingerprint)] = obj

    def finalize(self) -> list[AggregateAlert]:
        """Seal still-open windows and return all sealed aggregates once."""
        for key in sorted(self._open):
            self._closed.append(self._open[key].seal())
        self._open.clear()
        # De-duplicate defensively by aggregate identity; an aggregate that
        # expired during feeding must never be sealed twice.
        unique: dict[str, AggregateAlert] = {}
        for agg in self._closed:
            unique.setdefault(agg.aggregate_id, agg)
        result = sorted(unique.values(),
                        key=lambda a: (a.first_ts_ms, a.service_id,
                                       a.fingerprint, a.aggregate_id))
        self._closed = list(result)
        return list(result)


def aggregate_alerts(alerts: list[NormalizedAlert], rules: RuleSpec
                     ) -> list[AggregateAlert]:
    """Batch entry point; same code path as incremental feeding."""
    dedup = Deduplicator(rules)
    for alert in sorted(alerts, key=lambda a: (a.timestamp_ms, a.line,
                                               a.alert_id)):
        dedup.feed(alert)
    return dedup.finalize()
