"""Deterministic cascaded shaping engine with loopback datagram forwarding."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .errors import ETHICAL_INGRESS_MARKER, LedgerError
from .fixed import FixedPointSpec
from .generator import DeterministicUdpGenerator, decode_packet, encode_packet
from .ledger import DecisionLedger, LedgerRecovery
from .models import (
    DecisionEvent,
    GeneratorConfig,
    Packet,
    PolicySet,
    RunSummary,
    ShaperPolicy,
    StageStats,
)
from .token_bucket import TokenBucket
from .transport import InMemoryTransport, LoopbackTransport, Transport


@dataclass
class QueueEntry:
    packet_id: str
    packet_seq: int
    enqueued_tick: int
    release_tick: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "packet_seq": self.packet_seq,
            "enqueued_tick": self.enqueued_tick,
            "release_tick": self.release_tick,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QueueEntry":
        release = value.get("release_tick")
        return cls(
            packet_id=str(value["packet_id"]),
            packet_seq=int(value["packet_seq"]),
            enqueued_tick=int(value["enqueued_tick"]),
            release_tick=None if release is None else int(release),
        )


@dataclass
class ShaperRuntime:
    policy: ShaperPolicy
    bucket: TokenBucket
    queue: deque[QueueEntry] = field(default_factory=deque)
    passed_packets: int = 0
    queued_packets: int = 0
    dropped_packets: int = 0
    released_packets: int = 0
    forwarded_bytes: int = 0
    max_queue_len: int = 0
    total_delay_ticks: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.as_dict(),
            "bucket": self.bucket.to_state(),
            "queue": [entry.as_dict() for entry in self.queue],
            "passed_packets": self.passed_packets,
            "queued_packets": self.queued_packets,
            "dropped_packets": self.dropped_packets,
            "released_packets": self.released_packets,
            "forwarded_bytes": self.forwarded_bytes,
            "max_queue_len": self.max_queue_len,
            "total_delay_ticks": self.total_delay_ticks,
        }


@dataclass(slots=True)
class _ScheduledEvent:
    kind: str
    packet: Packet | None
    stage_id: str


@dataclass(frozen=True)
class EngineRunResult:
    summary: RunSummary
    events: tuple[DecisionEvent, ...]
    recovery: LedgerRecovery | None
    state: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.as_dict(),
            "events": [event.as_dict() for event in self.events],
            "recovery": None if self.recovery is None else self.recovery.as_dict(),
            "state": self.state,
        }


class CascadeEngine:
    """Drive packets through a policy's cascade using deterministic virtual ticks."""

    RELEASE_PRIORITY = 0
    ARRIVAL_PRIORITY = 1
    INGRESS_PRIORITY = 2

    def __init__(
        self,
        policy: PolicySet,
        generator: DeterministicUdpGenerator | GeneratorConfig | Mapping[str, Any],
        *,
        scale: int,
        transport: Transport | None = None,
        use_loopback: bool = True,
        base_port: int = 49152,
        crash_hook: Callable[[int, str], None] | None = None,
        drain_final: bool = True,
    ) -> None:
        self.policy = policy
        self.generator = (
            generator
            if isinstance(generator, DeterministicUdpGenerator)
            else DeterministicUdpGenerator(generator)
        )
        self.spec = FixedPointSpec(scale)
        self.scale = scale
        self.crash_hook = crash_hook
        self.drain_final = drain_final
        self._owns_transport = transport is None
        if transport is None:
            stage_ids = [shaper.shaper_id for shaper in policy.shapers]
            if use_loopback:
                transport = LoopbackTransport(stage_ids, base_port=base_port)
            else:
                transport = InMemoryTransport(stage_ids, base_port=base_port)
        self.transport = transport
        self._packets_by_id = {packet.packet_id: packet for packet in self.generator.packets}
        self._runtimes: dict[str, ShaperRuntime] = {}
        self._reset_runtimes()
        self._heap: list[tuple[int, int, str, int, str, int, _ScheduledEvent]] = []
        self._order = 0
        self._next_event_seq = 1
        self._current_batch_no = 0
        self._batch_events: list[DecisionEvent] = []
        self._all_events: list[DecisionEvent] = []
        self._final_released_bytes = 0
        self._final_released_packets = 0
        self._unmatched_packets = 0
        self._udp_sent = 0
        self._udp_received = 0
        self._closed = False

    def _reset_runtimes(self) -> None:
        self._runtimes = {}
        for shaper in self.policy.shapers:
            self._runtimes[shaper.shaper_id] = ShaperRuntime(
                policy=shaper,
                bucket=TokenBucket.create(
                    scale=self.scale,
                    committed_rate_kbps=shaper.committed_rate_kbps,
                    burst_bytes=shaper.burst_bytes,
                ),
            )

    def __enter__(self) -> "CascadeEngine":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed and self._owns_transport:
            self.transport.close()
        self._closed = True

    @property
    def ports(self) -> dict[str, int]:
        return self.transport.ports

    def _schedule(
        self,
        *,
        tick: int,
        priority: int,
        kind: str,
        packet: Packet | None,
        stage_id: str,
    ) -> None:
        self._order += 1
        packet_seq = 0 if packet is None else packet.seq
        heapq.heappush(
            self._heap,
            (
                tick,
                priority,
                stage_id,
                packet_seq,
                kind,
                self._order,
                _ScheduledEvent(kind, packet, stage_id),
            ),
        )

    def _emit(
        self,
        *,
        packet: Packet,
        stage_id: str,
        kind: str,
        decision: str,
        tick: int,
        tokens_before: int,
        tokens_after: int,
        queue_len_before: int,
        queue_len_after: int,
        delay_ticks: int = 0,
        release_tick: int | None = None,
        forwarded_to: Iterable[str] = (),
        reason: str = "",
    ) -> DecisionEvent:
        event = DecisionEvent(
            event_seq=self._next_event_seq,
            batch_no=self._current_batch_no,
            packet_id=packet.packet_id,
            packet_seq=packet.seq,
            stage_id=stage_id,
            kind=kind,
            decision=decision,
            time_tick=tick,
            time_ns=self.spec.time_tick_to_ns(tick),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            queue_len_before=queue_len_before,
            queue_len_after=queue_len_after,
            delay_ticks=delay_ticks,
            release_tick=release_tick,
            forwarded_to=tuple(sorted(forwarded_to)),
            reason=reason,
        )
        self._next_event_seq += 1
        self._batch_events.append(event)
        self._all_events.append(event)
        return event

    @staticmethod
    def _matches_prefix(dst_port: int, prefix: int) -> bool:
        return str(dst_port).startswith(str(prefix))

    def _roots_for(self, packet: Packet) -> tuple[str, ...]:
        return tuple(
            root
            for root in self.policy.roots
            if self._matches_prefix(packet.dst_port, self.policy.by_id[root].match_dst_port_prefix)
        )

    def _send_generated_packet(self, packet: Packet) -> None:
        roots = self._roots_for(packet)
        tick = self.spec.quantize_time_ns(packet.timestamp_ns)
        if not roots:
            self._unmatched_packets += 1
            self._schedule(
                tick=tick,
                priority=self.INGRESS_PRIORITY,
                kind="ingress_drop",
                packet=packet,
                stage_id=ETHICAL_INGRESS_MARKER,
            )
            return
        datagram = encode_packet(packet)
        for root in roots:
            self.transport.send_stage(root, datagram)
            self._udp_sent += 1
            received = self.transport.receive_stage(root)
            self._udp_received += 1
            if received != datagram:
                raise LedgerError(f"loopback corruption at stage {root}")
            decoded = decode_packet(received)
            if decoded.packet_id != packet.packet_id:
                raise LedgerError("loopback packet identity changed")
            self._schedule(
                tick=tick,
                priority=self.ARRIVAL_PRIORITY,
                kind="arrival",
                packet=decoded,
                stage_id=root,
            )

    def _forward(self, packet: Packet, stage_id: str, tick: int) -> tuple[str, ...]:
        children = self.policy.children_by_upstream.get(stage_id, ())
        if not children:
            datagram = encode_packet(packet)
            self.transport.deliver_sink(datagram)
            self._udp_sent += 1
            self._udp_received += 1
            self._final_released_packets += 1
            self._final_released_bytes += packet.size_bytes
            return ("<sink>",)
        datagram = encode_packet(packet)
        for child in children:
            received = self.transport.deliver_stage(child, datagram)
            self._udp_sent += 1
            self._udp_received += 1
            if received != datagram:
                raise LedgerError(f"loopback corruption forwarding to {child}")
            decoded = decode_packet(received)
            self._schedule(
                tick=tick,
                priority=self.ARRIVAL_PRIORITY,
                kind="arrival",
                packet=decoded,
                stage_id=child,
            )
        return tuple(children)

    def _schedule_head_release(self, runtime: ShaperRuntime, now_tick: int) -> None:
        if not runtime.queue:
            return
        entry = runtime.queue[0]
        packet = self._packets_by_id[entry.packet_id]
        cost = self.spec.bytes_to_units(packet.size_bytes)
        delay_ticks = runtime.bucket.release_delay_ticks(cost)
        release_tick = max(now_tick, now_tick + delay_ticks)
        entry.release_tick = release_tick
        self._schedule(
            tick=release_tick,
            priority=self.RELEASE_PRIORITY,
            kind="release",
            packet=packet,
            stage_id=runtime.policy.shaper_id,
        )

    def _process_arrival(self, packet: Packet, stage_id: str, tick: int) -> None:
        runtime = self._runtimes[stage_id]
        cost = self.spec.bytes_to_units(packet.size_bytes)
        queue_before = len(runtime.queue)
        tokens = runtime.bucket.refill(tick)
        tokens_before = tokens
        if cost > runtime.bucket.capacity_units:
            runtime.dropped_packets += 1
            self._emit(
                packet=packet,
                stage_id=stage_id,
                kind="arrival",
                decision="DROP",
                tick=tick,
                tokens_before=tokens_before,
                tokens_after=runtime.bucket.tokens,
                queue_len_before=queue_before,
                queue_len_after=len(runtime.queue),
                reason="packet_exceeds_burst",
            )
            return
        # Strict FIFO: once a packet is waiting, newly arriving packets may not
        # bypass it even if their smaller size would fit in the current tokens.
        if not runtime.queue and runtime.bucket.tokens >= cost:
            runtime.bucket.tokens -= cost
            runtime.passed_packets += 1
            runtime.forwarded_bytes += packet.size_bytes
            forwarded_to = self._forward(packet, stage_id, tick)
            self._emit(
                packet=packet,
                stage_id=stage_id,
                kind="arrival",
                decision="PASS",
                tick=tick,
                tokens_before=tokens_before,
                tokens_after=runtime.bucket.tokens,
                queue_len_before=queue_before,
                queue_len_after=len(runtime.queue),
                forwarded_to=forwarded_to,
            )
            return
        if runtime.policy.queue_limit_packets > 0 and len(runtime.queue) < (
            runtime.policy.queue_limit_packets
        ):
            was_empty = not runtime.queue
            runtime.queue.append(
                QueueEntry(
                    packet_id=packet.packet_id,
                    packet_seq=packet.seq,
                    enqueued_tick=tick,
                )
            )
            runtime.queued_packets += 1
            runtime.max_queue_len = max(runtime.max_queue_len, len(runtime.queue))
            if was_empty:
                self._schedule_head_release(runtime, tick)
            entry = runtime.queue[-1]
            release_tick = entry.release_tick
            delay_ticks = 0 if release_tick is None else release_tick - tick
            self._emit(
                packet=packet,
                stage_id=stage_id,
                kind="arrival",
                decision="QUEUE",
                tick=tick,
                tokens_before=tokens_before,
                tokens_after=runtime.bucket.tokens,
                queue_len_before=queue_before,
                queue_len_after=len(runtime.queue),
                delay_ticks=delay_ticks,
                release_tick=release_tick,
                reason="insufficient_tokens",
            )
            return
        runtime.dropped_packets += 1
        self._emit(
            packet=packet,
            stage_id=stage_id,
            kind="arrival",
            decision="DROP",
            tick=tick,
            tokens_before=tokens_before,
            tokens_after=runtime.bucket.tokens,
            queue_len_before=queue_before,
            queue_len_after=len(runtime.queue),
            reason="queue_full",
        )

    def _process_release(self, stage_id: str, tick: int) -> None:
        runtime = self._runtimes[stage_id]
        if not runtime.queue:
            return
        entry = runtime.queue[0]
        packet = self._packets_by_id[entry.packet_id]
        cost = self.spec.bytes_to_units(packet.size_bytes)
        tokens = runtime.bucket.refill(tick)
        tokens_before = tokens
        if runtime.bucket.tokens < cost:
            self._schedule_head_release(runtime, tick)
            return
        queue_before = len(runtime.queue)
        queued = runtime.queue.popleft()
        runtime.bucket.tokens -= cost
        delay_ticks = tick - queued.enqueued_tick
        runtime.released_packets += 1
        runtime.forwarded_bytes += packet.size_bytes
        runtime.total_delay_ticks += delay_ticks
        forwarded_to = self._forward(packet, stage_id, tick)
        self._emit(
            packet=packet,
            stage_id=stage_id,
            kind="release",
            decision="RELEASE",
            tick=tick,
            tokens_before=tokens_before,
            tokens_after=runtime.bucket.tokens,
            queue_len_before=queue_before,
            queue_len_after=len(runtime.queue),
            delay_ticks=delay_ticks,
            release_tick=tick,
            forwarded_to=forwarded_to,
        )
        if runtime.queue:
            self._schedule_head_release(runtime, tick)

    def _process_ingress_drop(self, packet: Packet, tick: int) -> None:
        self._emit(
            packet=packet,
            stage_id=ETHICAL_INGRESS_MARKER,
            kind="ingress",
            decision="DROP",
            tick=tick,
            tokens_before=0,
            tokens_after=0,
            queue_len_before=0,
            queue_len_after=0,
            reason="no_matching_root",
        )

    def _process_event(self, tick: int, event: _ScheduledEvent) -> None:
        if event.kind == "arrival" and event.packet is not None:
            self._process_arrival(event.packet, event.stage_id, tick)
        elif event.kind == "release":
            self._process_release(event.stage_id, tick)
        elif event.kind == "ingress_drop" and event.packet is not None:
            self._process_ingress_drop(event.packet, tick)
        else:
            raise LedgerError(f"unknown scheduled event kind: {event.kind}")

    def _serialize_state(self, cursor_seq: int) -> dict[str, Any]:
        return {
            "format_version": 1,
            "scale": self.scale,
            "cursor_seq": cursor_seq,
            "next_event_seq": self._next_event_seq,
            "event_order": self._order,
            "stages": {
                stage_id: self._runtimes[stage_id].as_dict()
                for stage_id in self.policy.topological_order()
            },
            "final_released_bytes": self._final_released_bytes,
            "final_released_packets": self._final_released_packets,
            "unmatched_packets": self._unmatched_packets,
            "udp_sent": self._udp_sent,
            "udp_received": self._udp_received,
        }

    def _restore_state(self, state: Mapping[str, Any]) -> None:
        if int(state.get("format_version", -1)) != 1:
            raise LedgerError("unsupported engine state format")
        if int(state.get("scale", -1)) != self.scale:
            raise LedgerError("engine state scale mismatch")
        stages = state.get("stages", {})
        if set(stages) != set(self._runtimes):
            raise LedgerError("engine state shaper set mismatch")
        for stage_id, runtime in self._runtimes.items():
            saved = stages[stage_id]
            if int(saved["policy"]["committed_rate_kbps"]) != runtime.policy.committed_rate_kbps:
                raise LedgerError(f"engine state policy mismatch for {stage_id}")
            runtime.bucket = TokenBucket.from_state(self.scale, saved["bucket"])
            runtime.queue = deque(QueueEntry.from_dict(item) for item in saved["queue"])
            runtime.passed_packets = int(saved["passed_packets"])
            runtime.queued_packets = int(saved["queued_packets"])
            runtime.dropped_packets = int(saved["dropped_packets"])
            runtime.released_packets = int(saved["released_packets"])
            runtime.forwarded_bytes = int(saved["forwarded_bytes"])
            runtime.max_queue_len = int(saved["max_queue_len"])
            runtime.total_delay_ticks = int(saved["total_delay_ticks"])
        self._next_event_seq = int(state["next_event_seq"])
        self._order = int(state["event_order"])
        self._final_released_bytes = int(state["final_released_bytes"])
        self._final_released_packets = int(state["final_released_packets"])
        self._unmatched_packets = int(state["unmatched_packets"])
        self._udp_sent = int(state.get("udp_sent", 0))
        self._udp_received = int(state.get("udp_received", 0))
        for stage_id in self.policy.topological_order():
            runtime = self._runtimes[stage_id]
            if runtime.queue:
                self._schedule_head_release(runtime, runtime.bucket.last_tick)

    def _summary(self, unknown_packet_ids: Iterable[str] = ()) -> RunSummary:
        stage_stats: list[StageStats] = []
        total_delay_ticks = 0
        total_releases = 0
        for stage_id in self.policy.topological_order():
            runtime = self._runtimes[stage_id]
            total_delay_ticks += runtime.total_delay_ticks
            total_releases += runtime.released_packets
            stage_stats.append(
                StageStats(
                    stage_id=stage_id,
                    passed_packets=runtime.passed_packets,
                    queued_packets=runtime.queued_packets,
                    dropped_packets=runtime.dropped_packets,
                    released_packets=runtime.released_packets,
                    forwarded_bytes=runtime.forwarded_bytes,
                    max_queue_len=runtime.max_queue_len,
                    total_delay_ticks=runtime.total_delay_ticks,
                )
            )
        average_delay_ns = (
            0
            if total_releases == 0
            else (total_delay_ticks * self.spec.tick_ns) // total_releases
        )
        return RunSummary(
            stage_stats=tuple(stage_stats),
            final_released_bytes=self._final_released_bytes,
            final_released_packets=self._final_released_packets,
            average_delay_ns=average_delay_ns,
            total_generated_packets=len(self.generator.packets),
            unmatched_packets=self._unmatched_packets,
            transport_datagrams_sent=self._udp_sent,
            transport_datagrams_received=self._udp_received,
            unknown_packet_ids=tuple(sorted(unknown_packet_ids)),
        )

    def _assert_queues_empty(self) -> None:
        nonempty = [
            stage_id
            for stage_id, runtime in self._runtimes.items()
            if runtime.queue
        ]
        if nonempty:
            raise LedgerError(
                "final drain left non-empty queues: " + ",".join(sorted(nonempty))
            )

    def run(self, *, ledger: DecisionLedger | None = None) -> EngineRunResult:
        recovery: LedgerRecovery | None = None
        packet_index = 0
        loaded_summary: RunSummary | None = None
        if ledger is not None:
            ledger.initialize(policy=self.policy, scale=self.scale, generator=self.generator)
            recovery = ledger.recover()
            progress = ledger.progress()
            packet_index = progress.cursor_seq
            if packet_index > 0:
                if progress.state is None:
                    raise LedgerError("ledger cursor is nonzero but has no committed state")
                self._restore_state(progress.state)
            if packet_index >= len(self.generator.packets):
                if progress.state is None:
                    raise LedgerError("completed ledger has no state")
                if progress.state["cursor_seq"] != len(self.generator.packets):
                    raise LedgerError("ledger cursor/state mismatch")
                self._assert_queues_empty()
                if progress.summary is None:
                    raise LedgerError("completed ledger has no summary")
                loaded_summary = RunSummary.from_dict(progress.summary)
                events = ledger.decisions()
                return EngineRunResult(
                    summary=loaded_summary,
                    events=events,
                    recovery=recovery,
                    state=progress.state,
                )
        elif self._next_event_seq != 1 or self._final_released_packets:
            raise LedgerError("an engine instance can only be run once without a ledger")

        batch_size = self.generator.config.batch_size
        total = len(self.generator.packets)
        while packet_index < total:
            batch_no = packet_index // batch_size
            batch_end = min(total, packet_index + batch_size)
            batch_packets = self.generator.packets[packet_index:batch_end]
            self._current_batch_no = batch_no
            self._batch_events = []
            if ledger is not None:
                ledger.begin_batch(batch_no, [packet.packet_id for packet in batch_packets])
            if self.crash_hook is not None:
                self.crash_hook(batch_no, "after_marker")

            for packet in batch_packets:
                self._send_generated_packet(packet)

            is_final_batch = batch_end == total
            if is_final_batch and self.drain_final:
                while self._heap:
                    tick, _, _, _, _, _, event = heapq.heappop(self._heap)
                    self._process_event(tick, event)
                self._assert_queues_empty()
            else:
                cutoff = self.spec.quantize_time_ns(batch_packets[-1].timestamp_ns)
                while self._heap and self._heap[0][0] <= cutoff:
                    tick, _, _, _, _, _, event = heapq.heappop(self._heap)
                    self._process_event(tick, event)

            next_index = batch_end
            state = self._serialize_state(next_index)
            final_summary = (
                self._summary() if is_final_batch and self.drain_final else None
            )
            if self.crash_hook is not None:
                self.crash_hook(batch_no, "before_commit")
            if ledger is not None:
                ledger.commit_batch(
                    batch_no=batch_no,
                    events=self._batch_events,
                    state=state,
                    cursor_seq=next_index,
                    next_event_seq=self._next_event_seq,
                    summary=None if final_summary is None else final_summary.as_dict(),
                    packet_ids=[packet.packet_id for packet in batch_packets],
                )
            packet_index = next_index

        unresolved_unknown = ()
        if ledger is not None:
            unresolved_unknown = tuple(
                item["packet_id"] for item in ledger.unknown_packets(unresolved_only=True)
            )
            events = ledger.decisions()
            progress = ledger.progress()
            if progress.summary is not None:
                loaded_summary = RunSummary.from_dict(progress.summary)
        else:
            events = tuple(self._all_events)
        summary = loaded_summary or self._summary(unresolved_unknown)
        state = self._serialize_state(packet_index)
        return EngineRunResult(
            summary=summary,
            events=events,
            recovery=recovery,
            state=state,
        )
