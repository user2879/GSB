"""Deterministic per-partition watermark/finalization/aggregation engine.

Determinism model
-----------------
Events inside a partition are always applied in canonical ``seq`` order no
matter how ingest calls interleave or batch them. Finalization depends only
on the event set and the integer-ms watermark, so the finalized set, the late
set and the aggregates are identical for any arrival order, batch size or
watermark step size. Cross-partition interleaving cannot affect a partition
because each partition owns an independent engine.

Sums accumulate in Python arbitrary-precision integers and are saturated to
int64 exactly once when materialized, avoiding the non-associativity that
per-add saturation would introduce.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Event, SessionSpec, saturate_int64
from .errors import WatermarkError


@dataclass(frozen=True)
class FinalizedEvent:
    seq: int
    ts_ms: int
    value: int


@dataclass(frozen=True)
class LateEvent:
    seq: int
    ts_ms: int
    watermark_ms: int
    source_line: int


@dataclass(frozen=True)
class WatermarkStep:
    partition: str
    old_watermark_ms: int
    new_watermark_ms: int
    trigger_seq: int
    reason: str


class PartitionEngine:
    """State machine for one partition key."""

    def __init__(self, key: str, spec: SessionSpec) -> None:
        self.key = key
        self.spec = spec
        self.max_ts_seen = 0
        self.watermark_ms = 0
        self._processed_seq = -1
        self._pending: dict[int, tuple[int, int]] = {}
        self.raw_sum = 0
        self.finalized_count = 0
        self.late_count = 0
        self.finalized: list[FinalizedEvent] = []
        self.late_events: list[LateEvent] = []
        self.steps: list[WatermarkStep] = []

    def apply_event(self, event: Event) -> list[WatermarkStep]:
        """Apply one event; callers guarantee contiguous seq ordering."""
        if event.seq <= self._processed_seq:
            return []
        if event.seq != self._processed_seq + 1:
            raise WatermarkError(
                "non-contiguous event application; reload required",
                partition=self.key, seq=event.seq,
                expected=self._processed_seq + 1)
        self._processed_seq = event.seq

        old_watermark = self.watermark_ms
        if event.ts_ms > self.max_ts_seen:
            self.max_ts_seen = event.ts_ms
        candidate = max(0, self.max_ts_seen - self.spec.out_of_order_ms)
        new_watermark = max(self.watermark_ms, candidate)
        steps: list[WatermarkStep] = []
        if new_watermark > old_watermark:
            self.watermark_ms = new_watermark
            step = WatermarkStep(self.key, old_watermark, new_watermark,
                                 event.seq, "event_advance")
            self.steps.append(step)
            steps.append(step)

        if event.ts_ms < self.watermark_ms:
            self.late_count += 1
            self.late_events.append(LateEvent(
                seq=event.seq, ts_ms=event.ts_ms,
                watermark_ms=self.watermark_ms,
                source_line=event.source_line))
        else:
            self._pending[event.seq] = (event.ts_ms, event.value)
        self._finalize_pending()
        return steps

    def request_watermark(self, requested_ms: int,
                          trigger_seq: int = -1) -> list[WatermarkStep]:
        """Explicit absolute watermark request; rollbacks are rejected."""
        if requested_ms < self.watermark_ms:
            raise WatermarkError(
                "watermark rollback rejected; watermark is monotonic",
                partition=self.key,
                current_watermark_ms=self.watermark_ms,
                requested_watermark_ms=requested_ms)
        if requested_ms == self.watermark_ms:
            return []
        old = self.watermark_ms
        self.watermark_ms = requested_ms
        if requested_ms > self.max_ts_seen:
            self.max_ts_seen = requested_ms
        step = WatermarkStep(self.key, old, requested_ms, trigger_seq,
                             "explicit_advance")
        self.steps.append(step)
        self._finalize_pending()
        return [step]

    def drain(self) -> list[WatermarkStep]:
        """Close the stream: advance to max observed ts, finalize the rest."""
        if not self._pending and self.watermark_ms >= self.max_ts_seen:
            return []
        return self.request_watermark(self.max_ts_seen, trigger_seq=-1)

    def _finalize_pending(self) -> None:
        if not self._pending:
            return
        for seq in sorted(self._pending):
            ts_ms, value = self._pending[seq]
            if ts_ms <= self.watermark_ms:
                self.raw_sum += value
                self.finalized_count += 1
                self.finalized.append(
                    FinalizedEvent(seq=seq, ts_ms=ts_ms, value=value))
                del self._pending[seq]
            else:
                break

    @property
    def cursor(self) -> int:
        """Last contiguously applied seq (-1 when nothing applied)."""
        return self._processed_seq

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def fixed_sum(self) -> int:
        return saturate_int64(self.raw_sum * (10 ** self.spec.scale))

    def view(self) -> dict[str, Any]:
        """Canonical, byte-stable per-partition view payload."""
        fixed = self.fixed_sum()
        return {
            "partition": self.key,
            "watermark_ms": self.watermark_ms,
            "max_ts_seen_ms": self.max_ts_seen,
            "finalized_count": self.finalized_count,
            "late_count": self.late_count,
            "pending_count": len(self._pending),
            "sum_raw": self.raw_sum,
            "sum_fixed": fixed,
            "sum_saturated": fixed != self.raw_sum * (10 ** self.spec.scale),
            "scale": self.spec.scale,
            "cursor_seq": self._processed_seq,
        }


class SessionEngine:
    """Owns one PartitionEngine per declared partition key."""

    def __init__(self, spec: SessionSpec, param_version: int = 1) -> None:
        self.spec = spec
        self.param_version = param_version
        self.parts: dict[str, PartitionEngine] = {
            rule.key: PartitionEngine(rule.key, spec) for rule in spec.rules
        }

    def apply_events(self, events: list[Event]) -> list[WatermarkStep]:
        """Apply a batch in canonical per-partition seq order.

        The batch is bucketed and sorted internally, so callers may pass
        events in any arrival order without affecting the result.
        """
        buckets: dict[str, list[Event]] = {key: [] for key in self.parts}
        for event in events:
            if event.partition not in buckets:
                raise WatermarkError("unknown partition",
                                     partition=event.partition)
            buckets[event.partition].append(event)
        steps: list[WatermarkStep] = []
        for key in sorted(buckets):
            for event in sorted(buckets[key], key=lambda e: e.seq):
                steps.extend(self.parts[key].apply_event(event))
        return steps

    def request_watermarks(self, requested_ms: int) -> list[WatermarkStep]:
        steps: list[WatermarkStep] = []
        for key in sorted(self.parts):
            steps.extend(self.parts[key].request_watermark(requested_ms))
        return steps


def _drain(self: "SessionEngine") -> list[WatermarkStep]:
    steps: list[WatermarkStep] = []
    for key in sorted(self.parts):
        steps.extend(self.parts[key].drain())
    return steps


SessionEngine.drain = _drain


def _session_views(self: "SessionEngine") -> list[dict[str, Any]]:
    return [self.parts[key].view() for key in sorted(self.parts)]


SessionEngine.views = _session_views
