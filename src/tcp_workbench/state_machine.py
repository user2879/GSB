"""Teaching TCP connection state machine and congestion-state advancement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .congestion import (
    RttEstimator,
    enter_fast_recovery,
    grow_cwnd_on_ack,
    threshold_after_loss,
    timeout_reset_cwnd,
)
from .errors import E_BUDGET, E_STATE, BusinessError
from .models import (
    ConnectionState,
    Event,
    EventType,
    PeerScript,
    SessionConfig,
    canonical_json,
)


@dataclass
class Segment:
    seq: int
    end_seq: int
    payload_bytes: int
    sent_mono_ms: int
    sent_event_seq: int
    retransmitted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "end_seq": self.end_seq,
            "payload_bytes": self.payload_bytes,
            "sent_mono_ms": self.sent_mono_ms,
            "sent_event_seq": self.sent_event_seq,
            "retransmitted": self.retransmitted,
        }


@dataclass
class ApplyResult:
    changed_items: tuple[str, ...]
    transitioned: bool = False


@dataclass
class SimulationState:
    config: SessionConfig
    script: PeerScript
    connection_state: ConnectionState = ConnectionState.CLOSED
    cwnd: int = 0
    ssthresh: int = 0
    dup_ack_count: int = 0
    last_ack_seq: int = 0
    highest_ack_seq: int = 0
    next_send_seq: int = 0
    outstanding: list[Segment] = field(default_factory=list)
    rtt: RttEstimator = field(init=False)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    events_by_type: dict[str, int] = field(default_factory=dict)
    bytes_sent: int = 0
    bytes_acked: int = 0
    segments_sent: int = 0
    segments_acked: int = 0
    bytes_retransmitted: int = 0
    retransmissions: int = 0
    timeouts: int = 0
    duplicate_acks: int = 0
    stale_acks: int = 0
    fast_retransmits: int = 0
    last_event_mono_ms: int | None = None
    first_event_mono_ms: int | None = None
    events_processed: int = 0
    fast_recovery: bool = False

    def __post_init__(self) -> None:
        self.cwnd = self.config.initial_cwnd_bytes
        self.ssthresh = self.config.congestion.initial_ssthresh_bytes
        self.last_ack_seq = self.config.initial_seq
        self.highest_ack_seq = self.config.initial_seq
        self.next_send_seq = self.config.first_data_seq
        self.rtt = RttEstimator(self.config.congestion)

    @property
    def bytes_in_flight(self) -> int:
        return sum(segment.payload_bytes for segment in self.outstanding)

    @property
    def segments_in_flight(self) -> int:
        return len(self.outstanding)

    def _fail(self, message: str, fields: list[str], code: str = E_STATE) -> None:
        raise BusinessError(code, message, fields)

    def _require_seq(self, event: Event, *, allow_ack_initial: bool = True) -> int:
        if not isinstance(event.seq, int) or isinstance(event.seq, bool):
            self._fail("event seq is required", ["seq"])
        lower = (
            self.config.initial_seq if allow_ack_initial else self.config.first_data_seq
        )
        if event.seq < lower or event.seq > self.config.max_data_seq + 1:
            self._fail("event seq is outside the configured sequence space", ["seq"])
        return event.seq

    def _require_ack(self, event: Event) -> int:
        if not isinstance(event.ack_seq, int) or isinstance(event.ack_seq, bool):
            self._fail("ack_seq is required", ["ack_seq"])
        if (
            event.ack_seq < self.config.first_data_seq
            or event.ack_seq > self.config.max_data_seq
        ):
            self._fail(
                "ack_seq is outside the configured data sequence space", ["ack_seq"]
            )
        return event.ack_seq

    def _transition(self, event: Event, target: ConnectionState) -> None:
        source = self.connection_state
        if source == target:
            return
        self.connection_state = target
        self.transitions.append(
            {
                "event_seq": event.event_seq,
                "from": str(source),
                "to": str(target),
                "trigger_packet_type": event.packet_type,
                "trigger_seq": event.seq,
                "mono_ms": event.recorded_mono_ms,
                "config_version": self.config.config_version,
                "script_version": self.script.script_version,
            }
        )

    def _ensure_transition(
        self,
        event: Event,
        allowed_sources: set[ConnectionState],
        target: ConnectionState,
    ) -> None:
        if self.connection_state not in allowed_sources:
            self._fail(
                f"{event.event_type} is invalid in {self.connection_state}",
                ["event_type", "connection_state"],
            )
        self._transition(event, target)

    def _check_budget(self, event: Event) -> None:
        if event.event_seq < 1 or event.event_seq > self.config.budget.max_events:
            self._fail(
                "session event budget exhausted", ["budget.max_events"], E_BUDGET
            )
        if event.recorded_mono_ms < 0:
            self._fail("recorded_mono_ms must be non-negative", ["recorded_mono_ms"])
        if (
            self.last_event_mono_ms is not None
            and event.recorded_mono_ms < self.last_event_mono_ms
        ):
            self._fail("event clock must be monotonic", ["recorded_mono_ms"])
        if self.first_event_mono_ms is None:
            self.first_event_mono_ms = event.recorded_mono_ms
        elapsed = event.recorded_mono_ms - self.first_event_mono_ms
        if elapsed > self.config.budget.max_elapsed_ms:
            self._fail(
                "session elapsed-time budget exhausted",
                ["budget.max_elapsed_ms"],
                E_BUDGET,
            )

    def _handle_send(self, event: Event) -> None:
        seq = self._require_seq(event, allow_ack_initial=False)
        if self.connection_state not in {
            ConnectionState.ESTABLISHED,
            ConnectionState.CLOSE_WAIT,
        }:
            self._fail(
                f"SEND is invalid in {self.connection_state}",
                ["connection_state", "event_type"],
            )
        if event.payload_bytes < 0 or event.payload_bytes > self.config.mss:
            self._fail("payload_bytes must be between 0 and MSS", ["payload_bytes"])

        if event.retransmitted:
            match = next(
                (segment for segment in self.outstanding if segment.seq == seq), None
            )
            if match is None:
                self._fail(
                    "retransmitted SEND must identify an outstanding segment", ["seq"]
                )
            match.retransmitted = True
            self.retransmissions += 1
            self.bytes_retransmitted += event.payload_bytes or match.payload_bytes
            return

        if seq != self.next_send_seq:
            self._fail("new SEND sequence is not contiguous", ["seq"])
        payload = event.payload_bytes or self.config.mss
        if payload <= 0:
            self._fail("payload_bytes must be positive", ["payload_bytes"])
        if self.bytes_sent + payload > self.config.budget.max_sent_bytes:
            self._fail(
                "sent-byte budget exhausted", ["budget.max_sent_bytes"], E_BUDGET
            )
        segment = Segment(
            seq=seq,
            end_seq=seq + payload,
            payload_bytes=payload,
            sent_mono_ms=event.recorded_mono_ms,
            sent_event_seq=event.event_seq,
        )
        self.outstanding.append(segment)
        self.outstanding.sort(key=lambda item: item.seq)
        self.next_send_seq = segment.end_seq
        self.bytes_sent += payload
        self.segments_sent += 1

    def _enter_duplicate_ack_state(self, event: Event) -> bool:
        self.duplicate_acks += 1
        self.dup_ack_count += 1
        if self.dup_ack_count == 3:
            self.ssthresh = threshold_after_loss(self.bytes_in_flight, self.config.mss)
            self.cwnd = enter_fast_recovery(
                self.ssthresh, self.config.mss, self.config.window_limit
            )
            self.fast_recovery = True
            self.fast_retransmits += 1
            if self.outstanding:
                target = self.outstanding[0]
                if not target.retransmitted:
                    target.retransmitted = True
                    self.retransmissions += 1
                    self.bytes_retransmitted += target.payload_bytes
            return True
        if self.fast_recovery and self.dup_ack_count > 3:
            self.cwnd = min(self.config.window_limit, self.cwnd + self.config.mss)
        return False

    def _handle_ack(self, event: Event, *, explicit_duplicate: bool) -> None:
        ack_seq = self._require_ack(event)
        if ack_seq < self.last_ack_seq:
            self.stale_acks += 1
            if explicit_duplicate:
                self._enter_duplicate_ack_state(event)
            return
        if ack_seq == self.last_ack_seq or explicit_duplicate:
            self._enter_duplicate_ack_state(event)
            return

        cleared = [
            segment for segment in self.outstanding if segment.end_seq <= ack_seq
        ]
        if not cleared:
            # A pure ACK that advances no tracked payload has no congestion effect.
            self.last_ack_seq = max(self.last_ack_seq, ack_seq)
            self.highest_ack_seq = max(self.highest_ack_seq, ack_seq)
            self.dup_ack_count = 0
            self.fast_recovery = False
            return
        if any(segment.end_seq > ack_seq for segment in cleared):
            self._fail("partial segment acknowledgement is not supported", ["ack_seq"])

        acked_bytes = sum(segment.payload_bytes for segment in cleared)
        newest = max(cleared, key=lambda segment: segment.end_seq)
        self.outstanding = [
            segment for segment in self.outstanding if segment.end_seq > ack_seq
        ]
        self.bytes_acked += acked_bytes
        self.segments_acked += len(cleared)
        self.last_ack_seq = ack_seq
        self.highest_ack_seq = max(self.highest_ack_seq, ack_seq)
        self.dup_ack_count = 0
        self.fast_recovery = False
        self.cwnd = grow_cwnd_on_ack(
            self.cwnd,
            self.ssthresh,
            acked_bytes,
            self.config.mss,
            self.config.window_limit,
        )

        if any(segment.retransmitted for segment in cleared):
            self.rtt.exclude_sample()
        else:
            sample = event.rtt_sample_ms
            if sample is None:
                sample = event.recorded_mono_ms - newest.sent_mono_ms
            if sample is not None:
                self.rtt.add_sample(sample)

    def _handle_timeout(self, event: Event) -> None:
        self.timeouts += 1
        target = None
        if event.seq is not None:
            target = next(
                (segment for segment in self.outstanding if segment.seq == event.seq),
                None,
            )
        if target is None and self.outstanding:
            target = self.outstanding[0]
        if target is not None:
            target.retransmitted = True
        self.ssthresh = threshold_after_loss(self.bytes_in_flight, self.config.mss)
        self.cwnd = timeout_reset_cwnd(self.config.mss)
        self.dup_ack_count = 0
        self.fast_recovery = False
        self.rtt.on_timeout()

    def _metric_snapshot(self) -> dict[str, Any]:
        return {
            "connection_state": str(self.connection_state),
            "cwnd": self.cwnd,
            "ssthresh": self.ssthresh,
            "dup_ack_count": self.dup_ack_count,
            "sent_unacked": {
                "bytes": self.bytes_in_flight,
                "segments": self.segments_in_flight,
            },
            "rtt": self.rtt.as_dict(),
            "stats": self.stats_dict(),
            "transitions": list(self.transitions),
            "budget": self.budget_dict(),
        }

    def stats_dict(self) -> dict[str, Any]:
        return {
            "events_total": self.events_processed,
            "events_by_type": dict(sorted(self.events_by_type.items())),
            "segments_sent": self.segments_sent,
            "segments_acked": self.segments_acked,
            "bytes_sent": self.bytes_sent,
            "bytes_acked": self.bytes_acked,
            "bytes_retransmitted": self.bytes_retransmitted,
            "retransmissions": self.retransmissions,
            "fast_retransmits": self.fast_retransmits,
            "duplicate_acks": self.duplicate_acks,
            "stale_acks": self.stale_acks,
            "timeouts": self.timeouts,
            "karn_excluded": self.rtt.karn_excluded,
            "transitions": len(self.transitions),
            "last_ack_seq": self.last_ack_seq,
            "highest_ack_seq": self.highest_ack_seq,
            "next_send_seq": self.next_send_seq,
        }

    def budget_dict(self) -> dict[str, Any]:
        elapsed = 0
        if self.first_event_mono_ms is not None and self.last_event_mono_ms is not None:
            elapsed = self.last_event_mono_ms - self.first_event_mono_ms
        return {
            "events_used": self.events_processed,
            "events_limit": self.config.budget.max_events,
            "bytes_sent": self.bytes_sent,
            "bytes_limit": self.config.budget.max_sent_bytes,
            "elapsed_ms": elapsed,
            "elapsed_limit_ms": self.config.budget.max_elapsed_ms,
            "exhausted": self.events_processed >= self.config.budget.max_events,
        }

    def apply(self, event: Event) -> ApplyResult:
        before = self._metric_snapshot()
        self._check_budget(event)
        if event.event_type not in {item.value for item in EventType}:
            self._fail("unknown event_type", ["event_type"])

        self.events_processed += 1
        self.events_by_type[event.event_type] = (
            self.events_by_type.get(event.event_type, 0) + 1
        )
        self.last_event_mono_ms = event.recorded_mono_ms

        if event.event_type == EventType.SYN.value:
            self._require_seq(event)
            self._ensure_transition(
                event, {ConnectionState.CLOSED}, ConnectionState.SYN_SENT
            )
        elif event.event_type == EventType.SYN_ACK.value:
            self._require_seq(event)
            self._ensure_transition(
                event, {ConnectionState.SYN_SENT}, ConnectionState.ESTABLISHED
            )
        elif event.event_type == EventType.SEND.value:
            self._handle_send(event)
        elif event.event_type == EventType.ACK.value:
            self._handle_ack(event, explicit_duplicate=False)
        elif event.event_type == EventType.DUP_ACK.value:
            self._require_ack(event)
            self._handle_ack(event, explicit_duplicate=True)
        elif event.event_type == EventType.TIMEOUT.value:
            if event.seq is not None:
                self._require_seq(event, allow_ack_initial=False)
            self._handle_timeout(event)
        elif event.event_type == EventType.FIN.value:
            self._require_seq(event)
            self._ensure_transition(
                event, {ConnectionState.ESTABLISHED}, ConnectionState.FIN_WAIT
            )
        elif event.event_type == EventType.FIN_ACK.value:
            self._require_seq(event)
            self._ensure_transition(
                event, {ConnectionState.FIN_WAIT}, ConnectionState.CLOSED
            )
        elif event.event_type == EventType.PEER_FIN.value:
            self._require_seq(event)
            self._ensure_transition(
                event,
                {ConnectionState.ESTABLISHED, ConnectionState.FIN_WAIT},
                ConnectionState.CLOSE_WAIT,
            )
        elif event.event_type == EventType.CLOSE.value:
            self._require_seq(event)
            self._ensure_transition(
                event, {ConnectionState.CLOSE_WAIT}, ConnectionState.CLOSED
            )

        after = self._metric_snapshot()
        changed = tuple(
            key
            for key in after
            if canonical_json(before[key]) != canonical_json(after[key])
        )
        return ApplyResult(
            changed_items=changed,
            transitioned=bool(before["transitions"] != after["transitions"]),
        )

    def diagnosis(self) -> dict[str, Any]:
        evidence = tuple(
            sorted(
                {
                    int(transition["event_seq"])
                    for transition in self.transitions
                    if transition.get("event_seq") is not None
                }
            )
        )
        if self.timeouts:
            return {
                "level": "critical",
                "code": "timeout_recovery",
                "message": "超时触发重传并将拥塞窗口回退到慢启动",
                "evidence_event_seqs": list(evidence),
            }
        if self.fast_retransmits or self.dup_ack_count >= 3:
            return {
                "level": "warning",
                "code": "fast_retransmit_observed",
                "message": "收到三次重复确认并进入快速重传",
                "evidence_event_seqs": list(evidence),
            }
        if self.connection_state == ConnectionState.CLOSE_WAIT:
            return {
                "level": "warning",
                "code": "close_wait",
                "message": "对端已关闭，本地仍处于 CLOSE_WAIT",
                "evidence_event_seqs": list(evidence),
            }
        if (
            self.connection_state == ConnectionState.CLOSED
            and self.events_processed == 0
        ):
            return {
                "level": "info",
                "code": "not_started",
                "message": "会话尚未开始",
                "evidence_event_seqs": [],
            }
        if self.cwnd > 0 and self.bytes_in_flight >= self.cwnd:
            return {
                "level": "warning",
                "code": "window_limited",
                "message": "已发送未确认数据达到拥塞窗口上限",
                "evidence_event_seqs": list(evidence),
            }
        if self.connection_state == ConnectionState.CLOSED:
            return {
                "level": "info",
                "code": "closed",
                "message": "连接已正常关闭",
                "evidence_event_seqs": list(evidence),
            }
        return {
            "level": "ok",
            "code": "healthy",
            "message": "连接与拥塞状态在预算内",
            "evidence_event_seqs": list(evidence),
        }
