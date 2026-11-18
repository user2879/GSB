"""Event-bus routing, watermark-driven window closure and deterministic output."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from .generator import Event, rejection_to_dict, validate_event
from .scheduler import Shard, WaitRecord
from .spec import RuleSpec, Spec
from .util import digest_of, monotonic_ms


@dataclass
class WindowResult:
    rule_id: str
    rule_version: int
    window_start_ms: int
    window_end_ms: int
    groups: list[dict[str, Any]]
    event_count: int
    content_digest: str

    @property
    def window_id(self) -> str:
        return f"{self.rule_id}:v{self.rule_version}:{self.window_start_ms}"

    def result_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "event_count": self.event_count,
            "groups": self.groups,
        }

    def ledger_dict(self) -> dict[str, Any]:
        return {"window_id": self.window_id, **self.result_dict()}

    def output_dict(self) -> dict[str, Any]:
        return {"window_id": self.window_id, "content_digest": self.content_digest, **self.result_dict()}


@dataclass
class LateRecord:
    source_id: str
    source_seq: int
    rule_id: str
    timestamp_ms: int
    window_start_ms: int
    closed_window_end_ms: int
    group_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_seq": self.source_seq,
            "rule_id": self.rule_id,
            "timestamp_ms": self.timestamp_ms,
            "window_start_ms": self.window_start_ms,
            "closed_window_end_ms": self.closed_window_end_ms,
            "group_key": self.group_key,
        }


@dataclass
class RunReport:
    accepted_events: int = 0
    rejected_events: list[dict[str, Any]] = field(default_factory=list)
    late: list[LateRecord] = field(default_factory=list)
    results: list[WindowResult] = field(default_factory=list)
    watermark_ms: int | str = 0

    @property
    def late_rule_events(self) -> int:
        return len(self.late)

    def event_report(self) -> dict[str, Any]:
        return {
            "accepted_events": self.accepted_events,
            "rejected_events": len(self.rejected_events),
            "rejected": self.rejected_events,
            "late_rule_events": self.late_rule_events,
            "late": [record.to_dict() for record in self.late],
        }


class Engine:
    _EOF_WATERMARK = 10**18

    def __init__(
        self,
        spec: Spec,
        global_limit: int,
        *,
        budget_wait_threshold_ms: float | None = None,
        schedule_jitter: bool = False,
        max_workers: int | None = None,
        compute_hook=None,
    ):
        from .scheduler import Scheduler

        self.spec = spec
        self.global_limit = global_limit
        self.schedule_jitter = schedule_jitter
        self.scheduler = Scheduler(
            spec.rules,
            global_limit,
            budget_wait_threshold_ms=budget_wait_threshold_ms,
            max_workers=max_workers,
            compute_hook=compute_hook,
        )
        self.shards: dict[tuple[str, int, str], Shard] = {}
        self.closed_windows: set[tuple[str, int]] = set()
        self.windows_by_rule: dict[str, set[int]] = {rule.rule_id: set() for rule in spec.rules}
        self.source_watermark: dict[str, int] = {source.source_id: -1 for source in spec.sources}
        self.global_watermark = -1
        self._inflight: set[Future] = set()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._parked: list[tuple[Event, RuleSpec, int]] = []
        self._park_wait_start: dict[tuple[str, int, str], float] = {}
        self._buffered: dict[tuple[str, int, str], list[Event]] = {}

    def _rule_accepts_source(self, rule: RuleSpec, source_id: str) -> bool:
        return not rule.source_ids or source_id in rule.source_ids

    def _get_or_create_shard(self, rule: RuleSpec, window_start_ms: int, group_key: str) -> Shard:
        key = (rule.rule_id, window_start_ms, group_key)
        shard = self.shards.get(key)
        if shard is None:
            shard = Shard(rule_id=rule.rule_id, window_start_ms=window_start_ms, group_key=group_key)
            self.shards[key] = shard
            self.windows_by_rule[rule.rule_id].add(window_start_ms)
        return shard

    def _dispatch(self, shard: Shard, events: list[Event]) -> None:
        future: Future = Future()

        def on_complete(done_shard: Shard = shard) -> None:
            with self._wake:
                self._inflight.discard(future)
                self._wake.notify_all()

        self._inflight.add(future)
        self.scheduler.dispatch(shard, events, on_complete)
        future.set_running_or_notify_cancel()

    def _route_event(self, event: Event, rule: RuleSpec, window_start_ms: int, report: RunReport) -> None:
        window_end_ms = window_start_ms + rule.window_seconds * 1000
        if (rule.rule_id, window_start_ms) in self.closed_windows or window_end_ms <= self.global_watermark:
            report.late.append(
                LateRecord(
                    source_id=event.source_id,
                    source_seq=event.source_seq,
                    rule_id=rule.rule_id,
                    timestamp_ms=event.timestamp_ms,
                    window_start_ms=window_start_ms,
                    closed_window_end_ms=window_end_ms,
                    group_key=event.group_key,
                )
            )
            return
        group_key = event.group_key
        shard = self._get_or_create_shard(rule, window_start_ms, group_key)
        if not shard.admitted and not self.scheduler.admit(shard):
            # Back-pressure, never an eviction: park until the window frees items.
            park_key = (rule.rule_id, window_start_ms, group_key)
            self._park_wait_start.setdefault(park_key, monotonic_ms())
            self._parked.append((event, rule, window_start_ms))
            return
        shard_key = (rule.rule_id, window_start_ms, group_key)
        self._buffered.setdefault(shard_key, []).append(event)

    def _drain_parked(self) -> None:
        if not self._parked:
            return
        remaining: list[tuple[Event, RuleSpec, int]] = []
        admitted_now: list[tuple[Event, RuleSpec, int]] = []
        for event, rule, window_start_ms in self._parked:
            shard = self._get_or_create_shard(rule, window_start_ms, event.group_key)
            if shard.admitted or self.scheduler.admit(shard):
                admitted_now.append((event, rule, window_start_ms))
                park_key = (rule.rule_id, window_start_ms, event.group_key)
                start = self._park_wait_start.pop(park_key, None)
                if start is not None:
                    self.scheduler.record_memory_wait(
                        WaitRecord(
                            "memory_budget",
                            rule.rule_id,
                            window_start_ms,
                            event.group_key,
                            start,
                            monotonic_ms(),
                        )
                    )
            else:
                remaining.append((event, rule, window_start_ms))
        self._parked = remaining
        for event, rule, window_start_ms in admitted_now:
            shard_key = (rule.rule_id, window_start_ms, event.group_key)
            self._buffered.setdefault(shard_key, []).append(event)

    def _force_admit_parked_at_eof(self) -> None:
        for event, rule, window_start_ms in self._parked:
            shard = self._get_or_create_shard(rule, window_start_ms, event.group_key)
            if not shard.admitted:
                # Bypass the live budget only at EOF; this never drops a group
                # and only runs once the input stream is fully known.
                key = (rule.rule_id, window_start_ms)
                items = self.scheduler.memory_items.setdefault(key, set())
                items.add(shard.group_key)
                shard.admitted = True
                self.scheduler.memory_peak[key] = max(
                    self.scheduler.memory_peak.get(key, 0), len(items)
                )
            shard_key = (rule.rule_id, window_start_ms, event.group_key)
            self._buffered.setdefault(shard_key, []).append(event)
        self._parked = []
        self._park_wait_start.clear()

    def _flush_buffers(self) -> None:
        """Dispatch one short task per shard for all buffered events."""
        if not self._buffered:
            return
        buffered = self._buffered
        self._buffered = {}
        for (rule_id, window_start_ms, group_key), events in buffered.items():
            shard = self.shards[(rule_id, window_start_ms, group_key)]
            self._dispatch(shard, events)

    def _check_budget_timeout(self) -> None:
        threshold = self.scheduler.budget_wait_threshold_ms
        if threshold is None or not self._parked:
            return
        now = monotonic_ms()
        for event, rule, window_start_ms in self._parked:
            park_key = (rule.rule_id, window_start_ms, event.group_key)
            start = self._park_wait_start.get(park_key)
            if start is not None and now - start >= threshold:
                items = self.scheduler.memory_items.get((rule.rule_id, window_start_ms), set())
                from .errors import BusinessError

                raise BusinessError(
                    "E_BUDGET",
                    "per-window memory budget remained exhausted beyond the configured threshold",
                    f"rules[{rule.rule_id}].memory_budget",
                    {
                        "rule_id": rule.rule_id,
                        "window_id": f"{rule.rule_id}:v{rule.version}:{window_start_ms}",
                        "window_start_ms": window_start_ms,
                        "window_end_ms": window_start_ms + rule.window_seconds * 1000,
                        "group_key": event.group_key,
                        "memory_budget": rule.memory_budget,
                        "memory_items": len(items),
                        "affected_rules": [rule.rule_id],
                        "wait_start_monotonic_ms": round(start, 3),
                        "wait_end_monotonic_ms": round(now, 3),
                        "waited_ms": round(now - start, 3),
                        "blocked_event_count": 1,
                        "last_closed_window_start_ms": None,
                        "last_closed_window_end_ms": None,
                    },
                )

    def _closure_due(self) -> bool:
        parked_keys = {
            (parked_rule.rule_id, parked_start)
            for _event, parked_rule, parked_start in self._parked
        }
        for rule in self.spec.rules:
            for start in self.windows_by_rule[rule.rule_id]:
                if (
                    (rule.rule_id, start) not in self.closed_windows
                    and (rule.rule_id, start) not in parked_keys
                    and start + rule.window_seconds * 1000 <= self.global_watermark
                ):
                    return True
        return False

    def _close_due_windows(self) -> list[WindowResult]:
        results: list[WindowResult] = []
        for rule in sorted(self.spec.rules, key=lambda item: item.rule_id):
            parked_keys = {
                (parked_rule.rule_id, parked_start)
                for _event, parked_rule, parked_start in self._parked
            }
            due = sorted(
                start
                for start in self.windows_by_rule[rule.rule_id]
                if (rule.rule_id, start) not in self.closed_windows
                and (rule.rule_id, start) not in parked_keys
                and start + rule.window_seconds * 1000 <= self.global_watermark
            )
            for window_start_ms in due:
                window_shards = [
                    shard
                    for (rule_id, start, _group), shard in self.shards.items()
                    if rule_id == rule.rule_id and start == window_start_ms
                ]
                groups = self._build_groups(rule, window_shards)
                event_count = sum(shard.event_count for shard in window_shards)
                payload = {
                    "rule_id": rule.rule_id,
                    "rule_version": rule.version,
                    "window_start_ms": window_start_ms,
                    "window_end_ms": window_start_ms + rule.window_seconds * 1000,
                    "event_count": event_count,
                    "groups": groups,
                }
                results.append(
                    WindowResult(
                        rule_id=rule.rule_id,
                        rule_version=rule.version,
                        window_start_ms=window_start_ms,
                        window_end_ms=window_start_ms + rule.window_seconds * 1000,
                        groups=groups,
                        event_count=event_count,
                        content_digest=digest_of(payload),
                    )
                )
                self.closed_windows.add((rule.rule_id, window_start_ms))
                self.scheduler.release_window(rule.rule_id, window_start_ms)
        return results

    def _build_groups(self, rule: RuleSpec, window_shards: list[Shard]) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for shard in sorted(window_shards, key=lambda item: item.group_key):
            entry: dict[str, Any] = {"group_key": shard.group_key}
            for aggregation in rule.aggregations:
                if aggregation == "max":
                    entry[aggregation] = shard.state.get("max")
                else:
                    entry[aggregation] = shard.state.get(aggregation, 0)
            groups.append(entry)
        groups.sort(key=lambda item: item["group_key"])
        return groups

    def _wait_inflight(self) -> None:
        with self._wake:
            while self._inflight:
                self.scheduler.raise_if_failed()
                self._wake.wait(timeout=0.05)
        self.scheduler.raise_if_failed()

    def ingest(self, plan: list[Event], streams: dict[str, list[Event]] | None = None) -> RunReport:
        report = RunReport()
        try:
            for event in plan:
                rejection = validate_event(event, self.spec, self.source_watermark[event.source_id])
                if rejection is not None:
                    report.rejected_events.append(rejection_to_dict(rejection))
                    continue
                self.source_watermark[event.source_id] = event.timestamp_ms
                self.global_watermark = min(self.source_watermark.values())
                for rule in self.spec.rules:
                    if not self._rule_accepts_source(rule, event.source_id):
                        continue
                    window_size_ms = rule.window_seconds * 1000
                    window_start_ms = (event.timestamp_ms // window_size_ms) * window_size_ms
                    self._route_event(event, rule, window_start_ms, report)
                report.accepted_events += 1
                self._drain_parked()
                self._flush_buffers()
                self._check_budget_timeout()
                if self._closure_due():
                    self._wait_inflight()
                    report.results.extend(self._close_due_windows())
                    self._drain_parked()
                    self._flush_buffers()

            # EOF punctuation: no further events can arrive, so the watermark
            # advances to an unbounded sentinel that closes every open window.
            self.global_watermark = self._EOF_WATERMARK
            # EOF is the final punctuation: no further groups can appear, so any
            # event still parked on the memory budget is force-admitted now. The
            # window then closes and releases its items in the normal path.
            self._force_admit_parked_at_eof()
            self._flush_buffers()
            self._wait_inflight()
            report.results.extend(self._close_due_windows())
            report.watermark_ms = "EOF"
            self.scheduler.shutdown()
            report.results = sorted(report.results, key=lambda item: (item.window_start_ms, item.rule_id))
            return report
        except BaseException:
            self.scheduler.shutdown()
            raise

    def mark_external_watermark(self, watermark_ms: int) -> RunReport:
        """Advance the watermark without new events and close due windows."""
        self.global_watermark = watermark_ms
        self._drain_parked()
        self._flush_buffers()
        self._wait_inflight()
        report = RunReport()
        report.results.extend(self._close_due_windows())
        return report
