"""Deterministic integer-millisecond event simulation core."""

from __future__ import annotations

import bisect
import copy
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .canon import (
    ceil_fraction,
    download_duration_ms,
    fraction_from_str,
    fraction_str,
    output_number,
)
from .errors import BusinessError
from .model import Representation, Segment, Session
from .snapshot import RecoveryInfo, SnapshotManager


THROUGHPUT_WINDOW = 3
EXIT_INTERRUPTED = 75


class SimulationInterrupted(Exception):
    def __init__(self, marker: dict, snapshot: dict | None = None) -> None:
        super().__init__("simulation interrupted at an event boundary")
        self.marker = marker
        self.snapshot = snapshot


@dataclass
class DecisionJob:
    job_seq: int
    segment_index: int
    rep_index: int
    bitrate_bps: Fraction
    size_bytes: Fraction
    duration_ms: int
    decided_at_ms: int
    trace_cursor: int
    buffer_ms: int
    playing: bool
    safety_ok: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_seq": self.job_seq,
            "segment_index": self.segment_index,
            "rep_index": self.rep_index,
            "bitrate_bps": fraction_str(self.bitrate_bps),
            "size_bytes": fraction_str(self.size_bytes),
            "duration_ms": self.duration_ms,
            "decided_at_ms": self.decided_at_ms,
            "trace_cursor": self.trace_cursor,
            "buffer_ms": self.buffer_ms,
            "playing": self.playing,
            "safety_ok": self.safety_ok,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionJob":
        return cls(
            job_seq=int(data["job_seq"]),
            segment_index=int(data["segment_index"]),
            rep_index=int(data["rep_index"]),
            bitrate_bps=fraction_from_str(data["bitrate_bps"], "snapshot.job.bitrate_bps"),
            size_bytes=fraction_from_str(data["size_bytes"], "snapshot.job.size_bytes"),
            duration_ms=int(data["duration_ms"]),
            decided_at_ms=int(data["decided_at_ms"]),
            trace_cursor=int(data["trace_cursor"]),
            buffer_ms=int(data["buffer_ms"]),
            playing=bool(data["playing"]),
            safety_ok=bool(data["safety_ok"]),
            reasons=list(data.get("reasons", [])),
        )


@dataclass
class DownloadTask:
    segment_index: int
    rep_index: int
    job_seq: int
    bitrate_bps: Fraction
    size_bytes: Fraction
    duration_ms: int
    start_ms: int
    complete_ms: int
    trace_cursor: int
    trace_bandwidth_bps: Fraction
    rtt_ms: int
    decision_buffer_ms: int
    safety_ok: bool
    reasons: list[str]
    fragment_digest: str

    def to_dict(self, now_ms: int | None = None) -> dict:
        result = {
            "segment_index": self.segment_index,
            "rep_index": self.rep_index,
            "job_seq": self.job_seq,
            "bitrate_bps": fraction_str(self.bitrate_bps),
            "size_bytes": fraction_str(self.size_bytes),
            "duration_ms": self.duration_ms,
            "start_ms": self.start_ms,
            "complete_ms": self.complete_ms,
            "trace_cursor": self.trace_cursor,
            "trace_bandwidth_bps": fraction_str(self.trace_bandwidth_bps),
            "rtt_ms": self.rtt_ms,
            "decision_buffer_ms": self.decision_buffer_ms,
            "safety_ok": self.safety_ok,
            "reasons": list(self.reasons),
            "fragment_digest": self.fragment_digest,
        }
        if now_ms is not None:
            result["progress_ms"] = max(0, min(self.duration_ms, now_ms - self.start_ms))
            result["remaining_ms"] = max(0, self.complete_ms - now_ms)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "DownloadTask":
        return cls(
            segment_index=int(data["segment_index"]),
            rep_index=int(data["rep_index"]),
            job_seq=int(data["job_seq"]),
            bitrate_bps=fraction_from_str(data["bitrate_bps"], "snapshot.in_flight.bitrate_bps"),
            size_bytes=fraction_from_str(data["size_bytes"], "snapshot.in_flight.size_bytes"),
            duration_ms=int(data["duration_ms"]),
            start_ms=int(data["start_ms"]),
            complete_ms=int(data["complete_ms"]),
            trace_cursor=int(data["trace_cursor"]),
            trace_bandwidth_bps=fraction_from_str(
                data["trace_bandwidth_bps"], "snapshot.in_flight.trace_bandwidth_bps"
            ),
            rtt_ms=int(data["rtt_ms"]),
            decision_buffer_ms=int(data["decision_buffer_ms"]),
            safety_ok=bool(data["safety_ok"]),
            reasons=list(data.get("reasons", [])),
            fragment_digest=str(data["fragment_digest"]),
        )


@dataclass
class CompletedDownload:
    segment_index: int
    rep_index: int
    job_seq: int
    bitrate_bps: Fraction
    size_bytes: Fraction
    duration_ms: int
    start_ms: int
    complete_ms: int
    trace_cursor: int
    media_start_ms: int
    media_end_ms: int
    fragment_digest: str

    def to_dict(self) -> dict:
        return {
            "segment_index": self.segment_index,
            "rep_index": self.rep_index,
            "job_seq": self.job_seq,
            "bitrate_bps": fraction_str(self.bitrate_bps),
            "size_bytes": fraction_str(self.size_bytes),
            "duration_ms": self.duration_ms,
            "start_ms": self.start_ms,
            "complete_ms": self.complete_ms,
            "trace_cursor": self.trace_cursor,
            "media_start_ms": self.media_start_ms,
            "media_end_ms": self.media_end_ms,
            "fragment_digest": self.fragment_digest,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompletedDownload":
        return cls(
            segment_index=int(data["segment_index"]),
            rep_index=int(data["rep_index"]),
            job_seq=int(data["job_seq"]),
            bitrate_bps=fraction_from_str(data["bitrate_bps"], "snapshot.completed.bitrate_bps"),
            size_bytes=fraction_from_str(data["size_bytes"], "snapshot.completed.size_bytes"),
            duration_ms=int(data["duration_ms"]),
            start_ms=int(data["start_ms"]),
            complete_ms=int(data["complete_ms"]),
            trace_cursor=int(data["trace_cursor"]),
            media_start_ms=int(data["media_start_ms"]),
            media_end_ms=int(data["media_end_ms"]),
            fragment_digest=str(data["fragment_digest"]),
        )


@dataclass
class EngineState:
    now_ms: int = 0
    playhead_ms: int = 0
    buffer_end_ms: int = 0
    playing: bool = False
    pause_reason: str | None = None
    download_index: int = 0
    trace_cursor: int = 0
    last_bitrate: Fraction | None = None
    completed: list[CompletedDownload] = field(default_factory=list)
    planned_jobs: list[DecisionJob] = field(default_factory=list)
    in_flight: DownloadTask | None = None
    next_event_index: int = 0
    event_seq: int = 0
    switch_count: int = 0
    drain_count: int = 0
    snapshot_seq: int = 0
    finished: bool = False
    quiescent: bool = False
    journal: list[dict] = field(default_factory=list)

    @property
    def buffer_ms(self) -> int:
        return max(0, self.buffer_end_ms - self.playhead_ms)


class StreamingSimulator:
    def __init__(
        self,
        session: Session,
        snapshot_manager: SnapshotManager,
        *,
        recovered_state: dict | None = None,
        recovery_info: RecoveryInfo | None = None,
        stop_event: threading.Event | None = None,
        event_delay_ms: int = 0,
    ) -> None:
        self.session = session
        self.client = session.client
        self.manager = snapshot_manager
        self.recovery_info = recovery_info or RecoveryInfo(mode="fresh")
        self.stop_event = stop_event or threading.Event()
        self.event_delay_ms = max(0, event_delay_ms)
        if recovered_state is None:
            self.state = EngineState()
            self._append_journal("simulation_start")
        else:
            self.state = self._state_from_dict(recovered_state)

    def run(self) -> EngineState:
        if self.state.finished or self.state.quiescent:
            return self.state
        while True:
            if self.stop_event.is_set():
                self._interrupt("SIGTERM")
            core = self._core_event_at_now()
            if core is not None:
                self._process_core_event(core)
                continue
            self._settle()
            if self.state.finished:
                break
            target = self._next_target()
            if target is None:
                self._quiesce()
                break
            self._wall_delay()
            if self.stop_event.is_set():
                self._interrupt("SIGTERM")
            self._advance_to(target[0])
        return self.state

    def snapshot_state_dict(self) -> dict:
        task_data = None
        if self.state.in_flight is not None:
            task_data = self.state.in_flight.to_dict(now_ms=self.state.now_ms)
        return {
            "now_ms": self.state.now_ms,
            "playhead_ms": self.state.playhead_ms,
            "buffer_start_ms": self.state.playhead_ms,
            "buffer_end_ms": self.state.buffer_end_ms,
            "buffer_ms": self.state.buffer_ms,
            "playing": self.state.playing,
            "pause_reason": self.state.pause_reason,
            "download_index": self.state.download_index,
            "current_fragment_index": self.state.download_index,
            "trace_cursor": self.state.trace_cursor,
            "last_bitrate_bps": (
                fraction_str(self.state.last_bitrate)
                if self.state.last_bitrate is not None
                else None
            ),
            "completed": [record.to_dict() for record in self.state.completed],
            "planned_jobs": [job.to_dict() for job in self.state.planned_jobs],
            "in_flight": task_data,
            "download_progress": task_data,
            "next_event_index": self.state.next_event_index,
            "event_seq": self.state.event_seq,
            "switch_count": self.state.switch_count,
            "drain_count": self.state.drain_count,
            "snapshot_seq": self.state.snapshot_seq,
            "finished": self.state.finished,
            "quiescent": self.state.quiescent,
            "event_script_digest": self.session.events.digest,
            "journal": copy.deepcopy(self.state.journal),
        }

    def build_report(self, *, prefetch: dict | None = None, log_path: str | None = None) -> dict:
        decisions = [event for event in self.state.journal if event.get("kind") == "decision"]
        invalid_snapshots = list(self.recovery_info.invalid_snapshots)
        integrity = {
            "status": "invalid_recovered" if invalid_snapshots else "ok",
            "invalid_snapshots": invalid_snapshots,
        }
        if invalid_snapshots:
            integrity["code"] = "E_DIGEST"
        return {
            "ok": True,
            "version_id": self.session.version_id,
            "recovery_start": self.recovery_info.as_dict(),
            "summary": {
                "event_count": self.state.event_seq,
                "decision_count": len(decisions),
                "switch_count": self.state.switch_count,
                "drain_count": self.state.drain_count,
                "fragment_downloads_completed": len(self.state.completed),
                "final_buffer_ms": self.state.buffer_ms,
                "final_playhead_ms": self.state.playhead_ms,
                "finished": self.state.finished,
                "quiescent": self.state.quiescent,
                "current_fragment_index": self.state.download_index,
                "trace_cursor": self.state.trace_cursor,
            },
            "decisions": decisions,
            "events": copy.deepcopy(self.state.journal),
            "integrity": integrity,
            "prefetch": prefetch,
            "log_path": log_path,
        }

    def _state_from_dict(self, data: dict) -> EngineState:
        try:
            state = EngineState(
                now_ms=int(data["now_ms"]),
                playhead_ms=int(data["playhead_ms"]),
                buffer_end_ms=int(data["buffer_end_ms"]),
                playing=bool(data["playing"]),
                pause_reason=data.get("pause_reason"),
                download_index=int(data["download_index"]),
                trace_cursor=int(data["trace_cursor"]),
                last_bitrate=(
                    fraction_from_str(data["last_bitrate_bps"], "snapshot.last_bitrate_bps")
                    if data.get("last_bitrate_bps") is not None
                    else None
                ),
                completed=[CompletedDownload.from_dict(item) for item in data.get("completed", [])],
                planned_jobs=[DecisionJob.from_dict(item) for item in data.get("planned_jobs", [])],
                in_flight=(
                    DownloadTask.from_dict(data["in_flight"])
                    if data.get("in_flight") is not None
                    else None
                ),
                next_event_index=int(data["next_event_index"]),
                event_seq=int(data["event_seq"]),
                switch_count=int(data["switch_count"]),
                drain_count=int(data["drain_count"]),
                snapshot_seq=int(data["snapshot_seq"]),
                finished=bool(data.get("finished", False)),
                quiescent=bool(data.get("quiescent", False)),
                journal=copy.deepcopy(data.get("journal", [])),
            )
        except (KeyError, TypeError, ValueError, BusinessError) as exc:
            raise BusinessError("E_SNAPSHOT", f"invalid snapshot state: {exc}", "snapshot.state") from exc
        if state.next_event_index < 0 or state.next_event_index > len(self.session.events.events):
            raise BusinessError("E_SNAPSHOT", "next_event_index is out of range", "snapshot.state")
        if state.download_index < 0 or state.download_index > len(self.session.media.fragments):
            raise BusinessError("E_SNAPSHOT", "download_index is out of range", "snapshot.state")
        if state.trace_cursor < 0 or state.trace_cursor >= len(self.session.network.segments):
            raise BusinessError("E_SNAPSHOT", "trace_cursor is out of range", "snapshot.state")
        if not (0 <= state.playhead_ms <= self.session.media.total_duration_ms):
            raise BusinessError("E_SNAPSHOT", "playhead_ms is out of range", "snapshot.state")
        if not (state.playhead_ms <= state.buffer_end_ms <= self.session.media.total_duration_ms):
            raise BusinessError("E_SNAPSHOT", "buffer range is out of order", "snapshot.state")
        return state

    def _append_journal(self, kind: str, **fields: Any) -> dict:
        self.state.event_seq += 1
        record = {
            "seq": self.state.event_seq,
            "at_ms": self.state.now_ms,
            "kind": kind,
            "buffer_ms": self.state.buffer_ms,
            "playhead_ms": self.state.playhead_ms,
            "playing": self.state.playing,
            "pause_reason": self.state.pause_reason,
            "download_index": self.state.download_index,
            "trace_cursor": self.state.trace_cursor,
        }
        record.update(fields)
        self.state.journal.append(record)
        return record

    def _core_event_at_now(self) -> tuple[str, dict | None] | None:
        if self.state.in_flight is not None:
            complete_ms = self.state.in_flight.complete_ms
            if complete_ms < self.state.now_ms:
                raise BusinessError("E_SNAPSHOT", "in-flight completion lies in the past", "snapshot.in_flight")
            if complete_ms == self.state.now_ms:
                return "completion", None
        if self.state.playing and self.state.playhead_ms >= self.session.media.total_duration_ms:
            return "finish", None
        if (
            self.state.playing
            and self.state.buffer_ms == 0
            and self.state.playhead_ms < self.session.media.total_duration_ms
        ):
            return "empty", None
        if self.state.next_event_index < len(self.session.events.events):
            event = self.session.events.events[self.state.next_event_index]
            if event.at_ms <= self.state.now_ms:
                return "script", {"event": event}
        return None

    def _process_core_event(self, core: tuple[str, dict | None]) -> None:
        kind, payload = core
        if kind == "completion":
            self._complete_download()
        elif kind == "finish":
            self._finish_playback()
        elif kind == "empty":
            self._buffer_empty()
        elif kind == "script":
            assert payload is not None
            self._process_script_event(payload["event"])
        else:
            raise BusinessError("E_INTERNAL", f"unknown core event kind: {kind}")

    def _settle(self) -> None:
        if self.state.finished or self.state.quiescent:
            return
        if (
            not self.state.playing
            and self.state.pause_reason in (None, "starved")
            and self.state.buffer_ms > 0
            and self.state.buffer_ms >= self.client.startup_buffer_ms
        ):
            self.state.playing = True
            self.state.pause_reason = None
            self._append_journal(
                "playback_start",
                buffer_target_ms=self.client.startup_buffer_ms,
            )
        if self.state.in_flight is None and self._can_start_download():
            self._start_download()

    def _can_start_download(self) -> bool:
        if self.state.in_flight is not None:
            return False
        if self.state.download_index >= len(self.session.media.fragments):
            return False
        if self.state.finished:
            return False
        segment = self.session.media.fragments[self.state.download_index]
        if segment.duration_ms > self.client.max_buffer_ms:
            return True
        if self.state.buffer_ms == 0:
            return True
        return self.state.buffer_ms + segment.duration_ms <= self.client.max_buffer_ms

    def _next_target(self) -> tuple[int, int, str, dict | None] | None:
        candidates: list[tuple[int, int, str, dict | None]] = []
        if self.state.in_flight is not None and self.state.in_flight.complete_ms > self.state.now_ms:
            candidates.append((self.state.in_flight.complete_ms, 0, "completion", None))
        if self.state.playing:
            if self.state.buffer_end_ms >= self.session.media.total_duration_ms:
                finish_at = self.state.now_ms + (
                    self.session.media.total_duration_ms - self.state.playhead_ms
                )
                if finish_at > self.state.now_ms:
                    candidates.append((finish_at, 1, "finish", None))
            elif self.state.buffer_ms > 0:
                candidates.append((self.state.now_ms + self.state.buffer_ms, 2, "empty", None))
        if self.state.next_event_index < len(self.session.events.events):
            event = self.session.events.events[self.state.next_event_index]
            event_at = max(self.state.now_ms, event.at_ms)
            if event_at > self.state.now_ms:
                candidates.append((event_at, 3, "script", {"event": event}))

        room_at = self._room_wait_target()
        if room_at is not None and room_at > self.state.now_ms:
            candidates.append((room_at, 4, "room", None))

        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], item[1]))

    def _room_wait_target(self) -> int | None:
        if self.state.in_flight is not None or self.state.finished:
            return None
        if self.state.download_index >= len(self.session.media.fragments):
            return None
        if not self.state.playing:
            return None
        segment = self.session.media.fragments[self.state.download_index]
        if segment.duration_ms > self.client.max_buffer_ms:
            return None
        required = max(0, self.client.max_buffer_ms - segment.duration_ms)
        if self.state.buffer_ms > required:
            return self.state.now_ms + (self.state.buffer_ms - required)
        return None

    def _wall_delay(self) -> None:
        if self.event_delay_ms <= 0:
            return
        deadline = time.monotonic() + self.event_delay_ms / 1000
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self.stop_event.is_set():
                return
            time.sleep(min(remaining, 0.02))

    def _advance_to(self, target_ms: int) -> None:
        if target_ms < self.state.now_ms:
            raise BusinessError("E_INTERNAL", "event clock moved backwards")
        delta = target_ms - self.state.now_ms
        if self.state.playing and delta:
            available = self.state.buffer_ms
            if delta > available:
                raise BusinessError(
                    "E_INTERNAL",
                    "event advanced beyond an unscheduled buffer-empty boundary",
                )
            self.state.playhead_ms += delta
        self.state.now_ms = target_ms

    def _buffer_empty(self) -> None:
        if not self.state.playing:
            return
        self.state.playing = False
        self.state.pause_reason = "starved"
        self.state.drain_count += 1
        self._append_journal("buffer_empty", drain_count=self.state.drain_count)
        self._snapshot("buffer_empty")

    def _finish_playback(self) -> None:
        self.state.playing = False
        self.state.pause_reason = None
        self.state.finished = True
        self._append_journal(
            "finish",
            final_playhead_ms=self.state.playhead_ms,
            total_duration_ms=self.session.media.total_duration_ms,
        )
        self._snapshot("finish")

    def _quiesce(self) -> None:
        if self.state.quiescent:
            return
        self.state.quiescent = True
        self._append_journal(
            "quiescent",
            download_complete=self.state.download_index >= len(self.session.media.fragments),
        )
        self._snapshot("quiescent")

    def _process_script_event(self, event: Any) -> None:
        self.state.next_event_index += 1
        if event.action == "terminate":
            snapshot = self._snapshot("terminate")
            marker = self._write_marker("scripted_terminate", snapshot)
            self.manager.write_log(self.state.journal)
            raise SimulationInterrupted(marker, snapshot)
        if event.action == "pause":
            self.state.playing = False
            self.state.pause_reason = "manual"
            self._append_journal("pause", event_at_ms=event.at_ms)
        elif event.action == "resume":
            was_manual = self.state.pause_reason == "manual"
            if was_manual:
                self.state.pause_reason = None
            self._append_journal(
                "resume",
                event_at_ms=event.at_ms,
                resumed_from_manual_pause=was_manual,
            )
        elif event.action == "jump":
            self._jump(event.target_ms or 0, event.at_ms)
        elif event.action == "trace_switch":
            assert event.segment_index is not None
            old_cursor = self.state.trace_cursor
            self.state.trace_cursor = event.segment_index
            self._append_journal(
                "trace_switch",
                event_at_ms=event.at_ms,
                old_trace_cursor=old_cursor,
                new_trace_cursor=event.segment_index,
            )
        else:
            raise BusinessError("E_EVENT", f"unsupported action: {event.action}")
        if event.action in {"pause", "resume", "jump", "trace_switch"}:
            self._snapshot(event.action)

    def _jump(self, target_ms: int, event_at_ms: int) -> None:
        old_playhead = self.state.playhead_ms
        old_buffer_end = self.state.buffer_end_ms
        old_in_flight = self.state.in_flight
        if target_ms <= old_playhead or target_ms > old_buffer_end:
            self.state.buffer_end_ms = target_ms
        self.state.playhead_ms = target_ms
        self.state.in_flight = None
        self.state.pause_reason = None
        self.state.playing = False
        self.state.download_index = self.session.segment_at_or_after(self.state.buffer_end_ms)
        if self.state.buffer_end_ms > self.state.playhead_ms:
            self.state.download_index = self.session.segment_at_or_after(self.state.buffer_end_ms)
        self._append_journal(
            "jump",
            event_at_ms=event_at_ms,
            target_ms=target_ms,
            old_playhead_ms=old_playhead,
            old_buffer_end_ms=old_buffer_end,
            new_buffer_end_ms=self.state.buffer_end_ms,
            cancelled_segment_index=(
                old_in_flight.segment_index if old_in_flight is not None else None
            ),
        )
        if target_ms >= self.session.media.total_duration_ms:
            self.state.buffer_end_ms = self.session.media.total_duration_ms
            self.state.playhead_ms = self.session.media.total_duration_ms
            self.state.download_index = len(self.session.media.fragments)
            self.state.finished = True
            self.state.playing = False
            self._append_journal(
                "finish",
                final_playhead_ms=self.state.playhead_ms,
                total_duration_ms=self.session.media.total_duration_ms,
                finish_reason="jump_to_end",
            )
    def _start_download(self) -> None:
        segment_index = self.state.download_index
        segment = self.session.media.fragments[segment_index]
        trace = self.session.network.segments[self.state.trace_cursor]
        recent = self.state.completed[-THROUGHPUT_WINDOW:]
        if recent:
            total_bytes = sum((record.size_bytes for record in recent), Fraction(0))
            total_ms = sum(record.duration_ms for record in recent)
            measured = total_bytes * 8000 / total_ms
            estimate = min(measured, trace.bandwidth_bps)
        else:
            measured = None
            estimate = trace.bandwidth_bps
        selected, safety_ok, reasons, projected, buffers = self._select_representation(
            segment, estimate
        )
        duration_ms = download_duration_ms(selected.size_bytes, trace.bandwidth_bps)
        complete_ms = self.state.now_ms + duration_ms
        job_seq = len(self.state.planned_jobs) + 1
        previous_bitrate = self.state.last_bitrate
        switched = previous_bitrate is not None and previous_bitrate != selected.bitrate_bps
        if switched:
            self.state.switch_count += 1
        job = DecisionJob(
            job_seq=job_seq,
            segment_index=segment_index,
            rep_index=selected.source_index,
            bitrate_bps=selected.bitrate_bps,
            size_bytes=selected.size_bytes,
            duration_ms=duration_ms,
            decided_at_ms=self.state.now_ms,
            trace_cursor=self.state.trace_cursor,
            buffer_ms=self.state.buffer_ms,
            playing=self.state.playing,
            safety_ok=safety_ok,
            reasons=reasons,
        )
        self.state.planned_jobs.append(job)
        self._append_journal(
            "decision",
            job_seq=job_seq,
            segment_index=segment_index,
            rep_index=selected.source_index,
            bitrate_bps=output_number(selected.bitrate_bps),
            size_bytes=output_number(selected.size_bytes),
            duration_ms=duration_ms,
            estimated_bandwidth_bps=output_number(estimate),
            measured_bandwidth_bps=(output_number(measured) if measured is not None else None),
            projected_buffer_after_next_ms=projected,
            buffer_at_completion_estimate_ms=buffers,
            safety_ok=safety_ok,
            reasons=reasons,
            switch_penalty_applied=switched,
            cumulative_switch_count=self.state.switch_count,
            strategy=self.client.strategy,
        )
        task = DownloadTask(
            segment_index=segment_index,
            rep_index=selected.source_index,
            job_seq=job_seq,
            bitrate_bps=selected.bitrate_bps,
            size_bytes=selected.size_bytes,
            duration_ms=duration_ms,
            start_ms=self.state.now_ms,
            complete_ms=complete_ms,
            trace_cursor=self.state.trace_cursor,
            trace_bandwidth_bps=trace.bandwidth_bps,
            rtt_ms=trace.rtt_ms,
            decision_buffer_ms=self.state.buffer_ms,
            safety_ok=safety_ok,
            reasons=reasons,
            fragment_digest=segment.fragment_digest,
        )
        self.state.in_flight = task
        self.state.last_bitrate = selected.bitrate_bps
        self.state.download_index += 1
        self._append_journal(
            "download_start",
            job_seq=job_seq,
            segment_index=segment_index,
            rep_index=selected.source_index,
            bitrate_bps=output_number(selected.bitrate_bps),
            size_bytes=output_number(selected.size_bytes),
            download_start_ms=self.state.now_ms,
            download_complete_ms=complete_ms,
            duration_ms=duration_ms,
            trace_cursor=self.state.trace_cursor,
            trace_bandwidth_bps=output_number(trace.bandwidth_bps),
            rtt_ms=trace.rtt_ms,
            fragment_digest=segment.fragment_digest,
        )

    def _select_representation(
        self, segment: Segment, estimate: Fraction
    ) -> tuple[Representation, bool, list[str], int, int]:
        ladder = self.session.media.bitrate_ladder
        reasons: list[str] = []
        buffer_before = self.state.buffer_ms
        playing = self.state.playing
        throughput_cap = estimate * self.client.throughput_safety_ratio
        cap_rate: Fraction | None = throughput_cap
        if throughput_cap < ladder[0]:
            reasons.append("throughput_below_lowest_ladder")

        if self.client.strategy in {"buffer", "hybrid"}:
            low_threshold = max(1, self.client.safety_margin_ms)
            if buffer_before < low_threshold:
                cap_rate = min(cap_rate or ladder[0], ladder[0])
                reasons.append("buffer_low")
            elif self.client.strategy == "buffer" and buffer_before < self.client.startup_buffer_ms:
                middle = ladder[min(1, len(ladder) - 1)]
                cap_rate = min(cap_rate or middle, middle)
                reasons.append("buffer_below_startup")
            elif self.client.strategy == "hybrid" and buffer_before < self.client.startup_buffer_ms:
                middle = ladder[len(ladder) // 2]
                cap_rate = min(cap_rate or middle, middle)
                reasons.append("buffer_below_startup")

        if (
            self.state.last_bitrate is not None
            and buffer_before + segment.duration_ms >= self.client.max_buffer_ms
        ):
            last_index = max(0, bisect.bisect_right(ladder, self.state.last_bitrate) - 1)
            down_index = max(0, last_index - 1)
            down_rate = ladder[down_index]
            cap_rate = min(cap_rate or down_rate, down_rate)
            reasons.append("buffer_cap_downgrade")

        base = [
            rep
            for rep in segment.representations
            if cap_rate is None or rep.bitrate_bps <= cap_rate
        ]
        if not base:
            base = [segment.representations[0]]
            reasons.append("rate_cap_forced_lowest")

        def projected_values(rep: Representation) -> tuple[int, int, int, bool]:
            current_download_ms = download_duration_ms(rep.size_bytes, estimate)
            buffer_after_current = (
                max(0, buffer_before - current_download_ms) + segment.duration_ms
                if playing
                else buffer_before + segment.duration_ms
            )
            may_play_next = playing or (
                self.state.pause_reason != "manual"
                and buffer_after_current > 0
                and buffer_after_current >= self.client.startup_buffer_ms
            )
            next_download_ms = ceil_fraction(
                rep.bitrate_bps * segment.duration_ms / estimate
            )
            projected = (
                buffer_after_current - next_download_ms
                if may_play_next
                else buffer_after_current
            )
            safe = (not may_play_next) or projected >= self.client.safety_margin_ms
            return current_download_ms, buffer_after_current, projected, safe

        safety_candidates: list[Representation] = []
        values_by_rep: dict[int, tuple[int, int, int, bool]] = {}
        for rep in base:
            values = projected_values(rep)
            values_by_rep[rep.source_index] = values
            if values[3]:
                safety_candidates.append(rep)
        if safety_candidates:
            candidates = safety_candidates
            safety_ok = True
        else:
            candidates = [base[0]]
            safety_ok = False
            reasons.append("no_representation_meets_safety")

        def score(rep: Representation) -> tuple[Fraction, Fraction, Fraction, int]:
            switched = self.state.last_bitrate is not None and rep.bitrate_bps != self.state.last_bitrate
            adjusted = rep.bitrate_bps * (1 - self.client.switch_penalty) if switched else rep.bitrate_bps
            return (adjusted, rep.bitrate_bps, -rep.size_bytes, -rep.source_index)

        selected = max(candidates, key=score)
        _, buffer_after_current, projected, _ = values_by_rep[selected.source_index]
        if self.state.last_bitrate is not None and selected.bitrate_bps != self.state.last_bitrate:
            reasons.append("switch_penalty_applied")
        reasons.append("selected_safe" if safety_ok else "selected_unsafe_fallback")
        return selected, safety_ok, reasons, projected, buffer_after_current

    def _complete_download(self) -> None:
        task = self.state.in_flight
        if task is None:
            raise BusinessError("E_INTERNAL", "completion event without an in-flight task")
        segment = self.session.media.fragments[task.segment_index]
        record = CompletedDownload(
            segment_index=task.segment_index,
            rep_index=task.rep_index,
            job_seq=task.job_seq,
            bitrate_bps=task.bitrate_bps,
            size_bytes=task.size_bytes,
            duration_ms=task.duration_ms,
            start_ms=task.start_ms,
            complete_ms=task.complete_ms,
            trace_cursor=task.trace_cursor,
            media_start_ms=segment.media_start_ms,
            media_end_ms=segment.media_end_ms,
            fragment_digest=task.fragment_digest,
        )
        self.state.completed.append(record)
        self.state.in_flight = None
        if segment.media_end_ms > self.state.buffer_end_ms:
            self.state.buffer_end_ms = segment.media_end_ms
        observed = task.size_bytes * 8000 / task.duration_ms
        self._append_journal(
            "download_complete",
            job_seq=task.job_seq,
            segment_index=task.segment_index,
            rep_index=task.rep_index,
            bitrate_bps=output_number(task.bitrate_bps),
            size_bytes=output_number(task.size_bytes),
            media_start_ms=segment.media_start_ms,
            media_end_ms=segment.media_end_ms,
            download_start_ms=task.start_ms,
            download_complete_ms=task.complete_ms,
            download_duration_ms=task.duration_ms,
            observed_throughput_bps=output_number(observed),
            buffer_after_ms=self.state.buffer_ms,
            buffer_cap_reached=self.state.buffer_ms >= self.client.max_buffer_ms,
            buffer_cap_exceeded=(
                self.state.buffer_ms > self.client.max_buffer_ms
            ),
        )
        self._snapshot("download_complete")

    def _snapshot(self, reason: str) -> dict:
        next_seq = self.state.snapshot_seq + 1
        top = self.manager.save_snapshot(self.snapshot_state_dict(), reason, next_seq)
        self.state.snapshot_seq = next_seq
        return top

    def _interrupt(self, signal_name: str) -> None:
        snapshot = self._snapshot("interrupt")
        marker = self._write_marker(signal_name, snapshot)
        self.manager.write_log(self.state.journal)
        raise SimulationInterrupted(marker, snapshot)

    def _write_marker(self, signal_name: str, snapshot: dict) -> dict:
        marker = {
            "version_id": self.session.version_id,
            "completed": False,
            "signal": signal_name,
            "event_seq": self.state.event_seq,
            "snapshot_seq": self.state.snapshot_seq,
            "snapshot_digest": snapshot.get("digest"),
            "next_event_index": self.state.next_event_index,
            "at_ms": self.state.now_ms,
        }
        self.manager.write_interruption_marker(marker)
        return marker


def simulate(
    session: Session,
    *,
    stop_event: threading.Event | None = None,
    event_delay_ms: int = 0,
) -> tuple[StreamingSimulator, dict | None, RecoveryInfo]:
    manager = SnapshotManager(
        session.state_dir,
        session.snapshot_dir,
        session.version_id,
        session.events.digest,
    )
    recovered_state, recovery = manager.prepare()
    simulator = StreamingSimulator(
        session,
        manager,
        recovered_state=recovered_state,
        recovery_info=recovery,
        stop_event=stop_event,
        event_delay_ms=event_delay_ms,
    )
    if simulator.state.finished:
        recovery.mode = "completed"
    simulator.run()
    return simulator, recovered_state, recovery
