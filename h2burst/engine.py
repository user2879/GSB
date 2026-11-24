"""Deterministic HTTP/2 flow-control scheduling and diagnostic engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .errors import WorkbenchError
from .frames import (
    FLAG_END_STREAM,
    FRAME_DATA,
    FRAME_HEADERS,
    make_data,
    make_headers,
)
from .model import EventSpec, MAX_WINDOW, SessionSpec, digest_value
from .state_machine import (
    ACTIVE_LOCAL_STATES,
    CLOSED,
    HALF_CLOSED_LOCAL,
    HALF_CLOSED_REMOTE,
    IDLE,
    OPEN,
    STATES,
    Transition,
    transition,
)


ENGINE_FORMAT_VERSION = 1
ROUND_ROBIN = "round_robin"
PROPORTIONAL = "proportional"


@dataclass
class StreamRuntime:
    stream_id: int
    priority: int
    request_body_bytes: int
    body_remaining: int
    window: int
    state: str = IDLE
    headers_sent: bool = False
    granted_bytes: int = 0
    granted_frames: int = 0
    expected_bytes: int = 0
    no_grant_rounds: int = 0
    max_grant_gap_rounds: int = 0
    starve_start_round: int | None = None
    starve_start_event_seq: int | None = None
    last_grant_frame_seq: int | None = None
    last_burst_event_seq: int | None = None
    last_burst_increment: int = 0
    opened_event_seq: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "priority": self.priority,
            "request_body_bytes": self.request_body_bytes,
            "body_remaining": self.body_remaining,
            "window": self.window,
            "state": self.state,
            "headers_sent": self.headers_sent,
            "granted_bytes": self.granted_bytes,
            "granted_frames": self.granted_frames,
            "expected_bytes": self.expected_bytes,
            "no_grant_rounds": self.no_grant_rounds,
            "max_grant_gap_rounds": self.max_grant_gap_rounds,
            "starve_start_round": self.starve_start_round,
            "starve_start_event_seq": self.starve_start_event_seq,
            "last_grant_frame_seq": self.last_grant_frame_seq,
            "last_burst_event_seq": self.last_burst_event_seq,
            "last_burst_increment": self.last_burst_increment,
            "opened_event_seq": self.opened_event_seq,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StreamRuntime":
        return cls(
            stream_id=int(raw["stream_id"]),
            priority=int(raw["priority"]),
            request_body_bytes=int(raw["request_body_bytes"]),
            body_remaining=int(raw["body_remaining"]),
            window=int(raw["window"]),
            state=str(raw["state"]),
            headers_sent=bool(raw["headers_sent"]),
            granted_bytes=int(raw.get("granted_bytes", 0)),
            granted_frames=int(raw.get("granted_frames", 0)),
            expected_bytes=int(raw.get("expected_bytes", 0)),
            no_grant_rounds=int(raw.get("no_grant_rounds", 0)),
            max_grant_gap_rounds=int(raw.get("max_grant_gap_rounds", 0)),
            starve_start_round=(
                None if raw.get("starve_start_round") is None else int(raw["starve_start_round"])
            ),
            starve_start_event_seq=(
                None
                if raw.get("starve_start_event_seq") is None
                else int(raw["starve_start_event_seq"])
            ),
            last_grant_frame_seq=(
                None if raw.get("last_grant_frame_seq") is None else int(raw["last_grant_frame_seq"])
            ),
            last_burst_event_seq=(
                None if raw.get("last_burst_event_seq") is None else int(raw["last_burst_event_seq"])
            ),
            last_burst_increment=int(raw.get("last_burst_increment", 0)),
            opened_event_seq=(
                None if raw.get("opened_event_seq") is None else int(raw["opened_event_seq"])
            ),
        )


@dataclass(frozen=True)
class FrameRecord:
    frame_seq: int
    event_seq: int | None
    round_index: int
    strategy: str
    frame_type: str
    stream_id: int
    flags: int
    length: int
    data_bytes: int
    connection_window_before: int
    connection_window_after: int
    stream_window_before: int
    stream_window_after: int
    business_code: str
    end_stream: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_seq": self.frame_seq,
            "event_seq": self.event_seq,
            "round_index": self.round_index,
            "strategy": self.strategy,
            "frame_type": self.frame_type,
            "stream_id": self.stream_id,
            "flags": self.flags,
            "length": self.length,
            "data_bytes": self.data_bytes,
            "connection_window_before": self.connection_window_before,
            "connection_window_after": self.connection_window_after,
            "stream_window_before": self.stream_window_before,
            "stream_window_after": self.stream_window_after,
            "business_code": self.business_code,
            "end_stream": self.end_stream,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FrameRecord":
        return cls(
            frame_seq=int(raw["frame_seq"]),
            event_seq=None if raw.get("event_seq") is None else int(raw["event_seq"]),
            round_index=int(raw["round_index"]),
            strategy=str(raw["strategy"]),
            frame_type=str(raw["frame_type"]),
            stream_id=int(raw["stream_id"]),
            flags=int(raw["flags"]),
            length=int(raw["length"]),
            data_bytes=int(raw["data_bytes"]),
            connection_window_before=int(raw["connection_window_before"]),
            connection_window_after=int(raw["connection_window_after"]),
            stream_window_before=int(raw["stream_window_before"]),
            stream_window_after=int(raw["stream_window_after"]),
            business_code=str(raw["business_code"]),
            end_stream=bool(raw["end_stream"]),
        )


@dataclass
class StrategySelection:
    strategy: str
    base_strategy: str
    mode: str
    tie_affected: bool
    reason_code: str
    reason: str
    params: dict[str, Any]
    predicted_round_robin_frames: int
    predicted_proportional_frames: int
    frame_difference: int
    tie_threshold_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "base_strategy": self.base_strategy,
            "mode": self.mode,
            "tie_affected": self.tie_affected,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "params": dict(self.params),
            "predicted_round_robin_frames": self.predicted_round_robin_frames,
            "predicted_proportional_frames": self.predicted_proportional_frames,
            "frame_difference": self.frame_difference,
            "tie_threshold_frames": self.tie_threshold_frames,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StrategySelection":
        return cls(
            strategy=str(raw["strategy"]),
            base_strategy=str(raw["base_strategy"]),
            mode=str(raw["mode"]),
            tie_affected=bool(raw["tie_affected"]),
            reason_code=str(raw["reason_code"]),
            reason=str(raw["reason"]),
            params=dict(raw["params"]),
            predicted_round_robin_frames=int(raw["predicted_round_robin_frames"]),
            predicted_proportional_frames=int(raw["predicted_proportional_frames"]),
            frame_difference=int(raw["frame_difference"]),
            tie_threshold_frames=int(raw["tie_threshold_frames"]),
        )


@dataclass
class StrategySwitch:
    round_index: int
    event_seq: int | None
    from_strategy: str
    to_strategy: str
    reason_code: str
    reason: str
    stream_deviations: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "event_seq": self.event_seq,
            "from_strategy": self.from_strategy,
            "to_strategy": self.to_strategy,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "stream_deviations": [dict(item) for item in self.stream_deviations],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StrategySwitch":
        return cls(
            round_index=int(raw["round_index"]),
            event_seq=None if raw.get("event_seq") is None else int(raw["event_seq"]),
            from_strategy=str(raw["from_strategy"]),
            to_strategy=str(raw["to_strategy"]),
            reason_code=str(raw["reason_code"]),
            reason=str(raw["reason"]),
            stream_deviations=[dict(item) for item in raw["stream_deviations"]],
        )


@dataclass
class EngineOptions:
    starvation_rounds: int = 5
    deviation_ratio_threshold: float = 0.5
    tie_threshold_small: int = 1
    tie_threshold_large: int = 2
    preemption_increment_threshold: int | None = None
    deadline_rounds: int = 100_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "starvation_rounds": self.starvation_rounds,
            "deviation_ratio_threshold": self.deviation_ratio_threshold,
            "tie_threshold_small": self.tie_threshold_small,
            "tie_threshold_large": self.tie_threshold_large,
            "preemption_increment_threshold": self.preemption_increment_threshold,
            "deadline_rounds": self.deadline_rounds,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "EngineOptions":
        raw = raw or {}
        return cls(
            starvation_rounds=int(raw.get("starvation_rounds", 5)),
            deviation_ratio_threshold=float(raw.get("deviation_ratio_threshold", 0.5)),
            tie_threshold_small=int(raw.get("tie_threshold_small", 1)),
            tie_threshold_large=int(raw.get("tie_threshold_large", 2)),
            preemption_increment_threshold=(
                None
                if raw.get("preemption_increment_threshold") is None
                else int(raw["preemption_increment_threshold"])
            ),
            deadline_rounds=int(raw.get("deadline_rounds", 100_000)),
        )


@dataclass
class _Preview:
    allocations: dict[int, int] = field(default_factory=dict)
    sequence: list[tuple[int, int]] = field(default_factory=list)
    frames: int = 0


class Engine:
    """A deterministic, fully serializable flow-control engine."""

    def __init__(
        self,
        spec: SessionSpec,
        *,
        options: EngineOptions | None = None,
        bootstrap: bool = True,
    ) -> None:
        self.spec = spec
        self.options = options or EngineOptions()
        self.connection_window = spec.connection.initial_window
        self.streams: dict[int, StreamRuntime] = {
            stream.stream_id: StreamRuntime(
                stream_id=stream.stream_id,
                priority=stream.priority,
                request_body_bytes=stream.request_body_bytes,
                body_remaining=stream.request_body_bytes,
                window=spec.connection.initial_stream_window,
            )
            for stream in spec.streams
        }
        self.current_strategy = ROUND_ROBIN
        self.pinned_strategy: str | None = None
        self.strategy_mode = "auto"
        self.selection = self._empty_selection()
        self.strategy_switches: list[StrategySwitch] = []
        self.frames: list[FrameRecord] = []
        self.transitions: list[Transition] = []
        self.warnings: list[dict[str, Any]] = []
        self.event_cursor = 0
        self.round_index = 0
        self.goaway_seen = False
        self.goaway_last_stream_id: int | None = None
        self.goaway_error_code = 0
        self.bootstrapped = False
        self._closed_warning_keys: set[tuple[Any, ...]] = set()
        if bootstrap:
            self.bootstrap()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def _empty_selection(self) -> StrategySelection:
        return StrategySelection(
            strategy=self.current_strategy,
            base_strategy=ROUND_ROBIN,
            mode="auto",
            tie_affected=False,
            reason_code="not_evaluated",
            reason="no schedulable streams have been evaluated",
            params={
                "stream_count": 0,
                "pending_frames": 0,
                "total_stream_window": 0,
                "average_stream_window": 0.0,
                "connection_window": self.connection_window,
                "scheduling_quantum": self.spec.connection.scheduling_quantum,
                "max_frame_size": self.spec.connection.max_frame_size,
            },
            predicted_round_robin_frames=0,
            predicted_proportional_frames=0,
            frame_difference=0,
            tie_threshold_frames=self.options.tie_threshold_small,
        )

    def to_state_dict(self) -> dict[str, Any]:
        state = {
            "engine_format_version": ENGINE_FORMAT_VERSION,
            "session_id": self.spec.session_id,
            "declared_version": self.spec.declared_version,
            "content_digest": self.spec.content_digest,
            "connection_window": self.connection_window,
            "streams": [self.streams[stream_id].to_dict() for stream_id in sorted(self.streams)],
            "current_strategy": self.current_strategy,
            "pinned_strategy": self.pinned_strategy,
            "strategy_mode": self.strategy_mode,
            "strategy_selection": self.selection.to_dict(),
            "strategy_switches": [switch.to_dict() for switch in self.strategy_switches],
            "frames": [frame.to_dict() for frame in self.frames],
            "transitions": [item.to_dict() for item in self.transitions],
            "warnings": [dict(item) for item in self.warnings],
            "event_cursor": self.event_cursor,
            "round_index": self.round_index,
            "goaway_seen": self.goaway_seen,
            "goaway_last_stream_id": self.goaway_last_stream_id,
            "goaway_error_code": self.goaway_error_code,
            "bootstrapped": self.bootstrapped,
            "options": self.options.to_dict(),
        }
        state["state_digest"] = digest_value(state)
        return state

    @classmethod
    def from_state_dict(
        cls,
        spec: SessionSpec,
        raw: Mapping[str, Any],
        *,
        bootstrap: bool = False,
    ) -> "Engine":
        if int(raw.get("engine_format_version", -1)) != ENGINE_FORMAT_VERSION:
            raise WorkbenchError(
                "E_SNAPSHOT",
                "unsupported engine snapshot format version",
                field="engine_format_version",
            )
        digest_payload = dict(raw)
        expected_digest = digest_payload.pop("state_digest", None)
        if expected_digest != digest_value(digest_payload):
            raise WorkbenchError(
                "E_SNAPSHOT",
                "snapshot state digest does not match its content",
                field="state_digest",
            )
        engine = cls(spec, options=EngineOptions.from_dict(raw.get("options")), bootstrap=False)
        engine.connection_window = int(raw["connection_window"])
        for stream_raw in raw["streams"]:
            runtime = StreamRuntime.from_dict(stream_raw)
            engine.streams[runtime.stream_id] = runtime
        engine.current_strategy = str(raw["current_strategy"])
        engine.pinned_strategy = raw.get("pinned_strategy")
        engine.strategy_mode = str(raw.get("strategy_mode", "auto"))
        engine.selection = StrategySelection.from_dict(raw["strategy_selection"])
        engine.strategy_switches = [StrategySwitch.from_dict(item) for item in raw["strategy_switches"]]
        engine.frames = [FrameRecord.from_dict(item) for item in raw["frames"]]
        engine.transitions = [Transition.from_dict(item) for item in raw["transitions"]]
        engine.warnings = [dict(item) for item in raw["warnings"]]
        engine.event_cursor = int(raw["event_cursor"])
        engine.round_index = int(raw["round_index"])
        engine.goaway_seen = bool(raw["goaway_seen"])
        engine.goaway_last_stream_id = raw.get("goaway_last_stream_id")
        engine.goaway_error_code = int(raw.get("goaway_error_code", 0))
        engine.bootstrapped = bool(raw.get("bootstrapped", True))
        for warning in engine.warnings:
            engine._closed_warning_keys.add(
                (
                    warning.get("type"),
                    warning.get("stream_id"),
                    warning.get("start_round"),
                    warning.get("source_event_seq"),
                )
            )
        if not bootstrap:
            return engine
        return engine

    # ------------------------------------------------------------------
    # Lifecycle and outbound frames
    # ------------------------------------------------------------------
    @property
    def next_event_seq(self) -> int:
        if self.event_cursor < len(self.spec.events):
            return self.spec.events[self.event_cursor].seq
        return self.spec.next_event_seq

    @property
    def last_frame_seq(self) -> int:
        return self.frames[-1].frame_seq if self.frames else 0

    def _active_local_count(self) -> int:
        return sum(1 for stream in self.streams.values() if stream.state in ACTIVE_LOCAL_STATES)

    def _append_frame(self, record: FrameRecord) -> None:
        self.frames.append(record)

    def _record_transition(
        self,
        stream: StreamRuntime,
        target: str,
        *,
        event_seq: int | None,
        frame_seq: int | None,
        reason: str,
    ) -> None:
        item = transition(
            stream.stream_id,
            stream.state,
            target,
            event_seq=event_seq,
            frame_seq=frame_seq,
            reason=reason,
        )
        stream.state = target
        self.transitions.append(item)

    def _emit_headers(self, stream: StreamRuntime, trigger_seq: int | None) -> None:
        if stream.state != IDLE or stream.headers_sent:
            return
        end_stream = stream.body_remaining == 0
        headers = (
            (":method", "GET" if end_stream else "POST"),
            (":path", f"/requests/{stream.stream_id}"),
            (":scheme", "http"),
            (":authority", "local.h2burst"),
            ("content-length", str(stream.request_body_bytes)),
        )
        frame = make_headers(stream.stream_id, headers, end_stream=end_stream)
        frame_seq = self.last_frame_seq + 1
        record = FrameRecord(
            frame_seq=frame_seq,
            event_seq=trigger_seq,
            round_index=self.round_index,
            strategy=self.current_strategy,
            frame_type="HEADERS",
            stream_id=stream.stream_id,
            flags=frame.flags,
            length=len(frame.payload),
            data_bytes=0,
            connection_window_before=self.connection_window,
            connection_window_after=self.connection_window,
            stream_window_before=stream.window,
            stream_window_after=stream.window,
            business_code="HEADERS_SENT",
            end_stream=end_stream,
        )
        self._append_frame(record)
        stream.headers_sent = True
        stream.opened_event_seq = trigger_seq
        self._record_transition(
            stream,
            OPEN,
            event_seq=trigger_seq,
            frame_seq=frame_seq,
            reason="request_headers",
        )
        if end_stream:
            self._record_transition(
                stream,
                HALF_CLOSED_LOCAL,
                event_seq=trigger_seq,
                frame_seq=frame_seq,
                reason="request_body_complete",
            )

    def _open_available(self, trigger_seq: int | None) -> None:
        active = self._active_local_count()
        limit = self.spec.connection.max_concurrent_streams
        if active >= limit:
            return
        ordered = sorted(
            (stream for stream in self.streams.values() if stream.state == IDLE),
            key=lambda stream: (stream.priority, stream.stream_id),
        )
        for stream in ordered:
            if active >= limit:
                break
            if self.goaway_seen and self.goaway_last_stream_id is not None:
                if stream.stream_id > self.goaway_last_stream_id:
                    continue
            before = stream.state
            self._emit_headers(stream, trigger_seq)
            if before == IDLE and stream.state in ACTIVE_LOCAL_STATES:
                active += 1

    def bootstrap(self) -> None:
        if self.bootstrapped:
            return
        trigger = self.spec.events[0].seq if self.spec.events else 0
        self._open_available(trigger)
        self.bootstrapped = True

    # ------------------------------------------------------------------
    # Event application
    # ------------------------------------------------------------------
    def _flow_error(self, event: EventSpec, message: str, field: str) -> WorkbenchError:
        return WorkbenchError(
            "E_FLOW",
            message,
            field=field,
            event_seq=event.seq,
        )

    def _apply_window_update(self, event: EventSpec) -> None:
        assert event.increment is not None
        if event.scope == "connection":
            before = self.connection_window
            if before + event.increment > MAX_WINDOW:
                raise self._flow_error(
                    event,
                    f"connection window {before} + increment {event.increment} exceeds {MAX_WINDOW}",
                    "increment",
                )
            self.connection_window = before + event.increment
            return
        stream = self.streams[event.stream_id]  # validated by the spec parser
        if stream.state == CLOSED:
            raise WorkbenchError(
                "E_STATE",
                f"window update targets closed stream {stream.stream_id}",
                field="stream_state",
                event_seq=event.seq,
                details={"stream_id": stream.stream_id, "from_state": stream.state},
            )
        before = stream.window
        if before + event.increment > MAX_WINDOW:
            raise self._flow_error(
                event,
                f"stream window {before} + increment {event.increment} exceeds {MAX_WINDOW}",
                "increment",
            )
        stream.window = before + event.increment
        threshold = self.options.preemption_increment_threshold
        if threshold is None:
            threshold = self.spec.connection.max_frame_size
        if event.increment >= threshold:
            stream.last_burst_event_seq = event.seq
            stream.last_burst_increment = event.increment

    def _apply_reset(self, event: EventSpec) -> None:
        stream = self.streams[event.stream_id]
        if stream.state == CLOSED:
            raise WorkbenchError(
                "E_STATE",
                f"RST_STREAM targets already closed stream {stream.stream_id}",
                field="stream_state",
                event_seq=event.seq,
                details={
                    "stream_id": stream.stream_id,
                    "from_state": stream.state,
                    "to_state": CLOSED,
                },
            )
        self._record_transition(
            stream,
            CLOSED,
            event_seq=event.seq,
            frame_seq=None,
            reason=f"rst_stream(error_code={event.error_code})",
        )

    def _apply_goaway(self, event: EventSpec) -> None:
        assert event.last_stream_id is not None
        self.goaway_seen = True
        self.goaway_last_stream_id = event.last_stream_id
        self.goaway_error_code = event.error_code
        for stream_id in sorted(self.streams):
            stream = self.streams[stream_id]
            if stream_id > event.last_stream_id and stream.state != CLOSED:
                self._record_transition(
                    stream,
                    CLOSED,
                    event_seq=event.seq,
                    frame_seq=None,
                    reason=f"goaway_not_processed(error_code={event.error_code})",
                )

    def apply_event(self, event: EventSpec) -> None:
        if event.kind == "window_update":
            self._apply_window_update(event)
        elif event.kind == "rst_stream":
            self._apply_reset(event)
        elif event.kind == "goaway":
            self._apply_goaway(event)
        elif event.kind == "priority_change":
            assert event.priority is not None
            self.streams[event.stream_id].priority = event.priority
        elif event.kind == "strategy_change":
            assert event.strategy is not None
            self.pinned_strategy = event.strategy
            self.strategy_mode = "pinned"
        else:  # pragma: no cover - parser prevents this
            raise WorkbenchError("E_SPEC", f"unsupported event kind {event.kind}", field="type", event_seq=event.seq)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    def _eligible(self) -> list[StreamRuntime]:
        candidates = [
            stream
            for stream in self.streams.values()
            if stream.state in (OPEN, HALF_CLOSED_REMOTE)
            and stream.body_remaining > 0
            and stream.window > 0
            and self.connection_window > 0
            and not (self.goaway_seen and stream.state == CLOSED)
        ]
        return sorted(candidates, key=lambda stream: (stream.priority, stream.stream_id))

    def _pending_frames(self, streams: Iterable[StreamRuntime]) -> int:
        max_frame = self.spec.connection.max_frame_size
        return sum(math.ceil(stream.body_remaining / max_frame) for stream in streams)

    def _selection_params(self, eligible: list[StreamRuntime]) -> dict[str, Any]:
        total_window = sum(stream.window for stream in eligible)
        count = len(eligible)
        return {
            "stream_count": count,
            "pending_frames": self._pending_frames(eligible),
            "total_stream_window": total_window,
            "average_stream_window": round(total_window / count, 3) if count else 0.0,
            "connection_window": self.connection_window,
            "scheduling_quantum": self.spec.connection.scheduling_quantum,
            "max_frame_size": self.spec.connection.max_frame_size,
        }

    def _scale_base_strategy(self, params: Mapping[str, Any]) -> tuple[str, str, str]:
        count = int(params["stream_count"])
        pending = int(params["pending_frames"])
        average = float(params["average_stream_window"])
        max_frame = int(params["max_frame_size"])
        if count >= 128 or pending >= 256:
            return (
                PROPORTIONAL,
                "scale_high_cardinality",
                f"stream_count {count} or pending_frames {pending} crosses the large-scale threshold",
            )
        if average >= max_frame * 3 and count >= 16:
            return (
                PROPORTIONAL,
                "scale_deep_window_ratio",
                f"average_stream_window {average} >= 3 * max_frame_size {max_frame} with {count} streams",
            )
        return (
            ROUND_ROBIN,
            "scale_small_equal_frames",
            f"stream_count {count}, pending_frames {pending}, average_window {average} favor equal frame turns",
        )

    def _preview(self, strategy: str, eligible: list[StreamRuntime], quantum: int) -> _Preview:
        if not eligible:
            return _Preview()
        max_frame = self.spec.connection.max_frame_size
        order = sorted(eligible, key=lambda stream: (stream.priority, stream.stream_id))
        if strategy == ROUND_ROBIN:
            weights = {stream.stream_id: 1 for stream in eligible}
        else:
            weights = {stream.stream_id: max(0, stream.window) for stream in eligible}
        caps = {
            stream.stream_id: min(
                math.ceil(stream.body_remaining / max_frame),
                math.ceil(stream.window / max_frame),
            )
            for stream in eligible
        }
        counts = {stream.stream_id: 0 for stream in eligible}
        remaining_slots = quantum
        active = [stream.stream_id for stream in order if caps[stream.stream_id] > 0]
        while remaining_slots > 0 and active:
            weight_sum = sum(weights[stream_id] for stream_id in active)
            if weight_sum <= 0:
                active_weights = {stream_id: 1 for stream_id in active}
                weight_sum = len(active)
            else:
                active_weights = weights
            base: dict[int, int] = {}
            remainders: list[tuple[int, int, int]] = []
            used = 0
            priority_by_id = {stream.stream_id: stream.priority for stream in eligible}
            for stream_id in active:
                quotient, remainder = divmod(remaining_slots * active_weights[stream_id], weight_sum)
                base[stream_id] = quotient
                used += quotient
                remainders.append((remainder, priority_by_id[stream_id], stream_id))
            leftover = remaining_slots - used
            remainders.sort(key=lambda item: (-item[0], item[1], item[2]))
            for _, _, stream_id in remainders[:leftover]:
                base[stream_id] += 1
            allocated_now = 0
            for stream_id in active:
                room = caps[stream_id] - counts[stream_id]
                assigned = min(base[stream_id], room)
                if assigned > 0:
                    counts[stream_id] += assigned
                    allocated_now += assigned
            if allocated_now == 0:
                break
            remaining_slots -= allocated_now
            active = [stream_id for stream_id in active if counts[stream_id] < caps[stream_id]]

        # Interleave the fixed slot counts in stable priority/stream order. This
        # is the actual on-wire slot order, not merely an aggregate entitlement.
        allocations: dict[int, int] = {}
        sequence: list[tuple[int, int]] = []
        remaining = {stream.stream_id: stream.body_remaining for stream in eligible}
        windows = {stream.stream_id: stream.window for stream in eligible}
        connection = self.connection_window
        max_count = max(counts.values(), default=0)
        for _cycle in range(max_count):
            for stream in order:
                stream_id = stream.stream_id
                if counts[stream_id] <= 0:
                    continue
                counts[stream_id] -= 1
                size = min(max_frame, remaining[stream_id], windows[stream_id], connection)
                if size <= 0:
                    continue
                allocations[stream_id] = allocations.get(stream_id, 0) + size
                sequence.append((stream_id, size))
                remaining[stream_id] -= size
                windows[stream_id] -= size
                connection -= size
        return _Preview(allocations=allocations, sequence=sequence, frames=len(sequence))

    def _select_strategy(self, eligible: list[StreamRuntime], trigger_seq: int | None) -> tuple[StrategySelection, _Preview, _Preview]:
        params = self._selection_params(eligible)
        base, base_reason_code, base_reason = self._scale_base_strategy(params)
        quantum = self.spec.connection.scheduling_quantum
        rr_preview = self._preview(ROUND_ROBIN, eligible, quantum)
        prop_preview = self._preview(PROPORTIONAL, eligible, quantum)
        difference = abs(rr_preview.frames - prop_preview.frames)
        tie_threshold = (
            self.options.tie_threshold_large
            if int(params["stream_count"]) >= 128 or int(params["pending_frames"]) >= 256
            else self.options.tie_threshold_small
        )
        tie_affected = difference <= tie_threshold
        if self.pinned_strategy is not None:
            chosen = self.pinned_strategy
            mode = "pinned"
            tie_affected = False
            reason_code = "explicit_strategy_event"
            reason = "an explicit strategy_change event pins the scheduling policy"
        elif tie_affected:
            chosen = base
            mode = "auto"
            reason_code = f"retain_tie_{base}"
            reason = (
                f"predicted-frame difference {difference} <= tie threshold {tie_threshold}; "
                f"retain scale-rule strategy {base}"
            )
        else:
            chosen = base
            mode = "auto"
            reason_code = base_reason_code
            reason = base_reason
        selection = StrategySelection(
            strategy=chosen,
            base_strategy=base,
            mode=mode,
            tie_affected=tie_affected,
            reason_code=reason_code,
            reason=reason,
            params=dict(params),
            predicted_round_robin_frames=rr_preview.frames,
            predicted_proportional_frames=prop_preview.frames,
            frame_difference=difference,
            tie_threshold_frames=tie_threshold,
        )
        return selection, rr_preview, prop_preview

    def _record_switch(
        self,
        old_strategy: str,
        new_strategy: str,
        selection: StrategySelection,
        old_preview: _Preview,
        new_preview: _Preview,
        trigger_seq: int | None,
    ) -> None:
        deviations = []
        for stream_id in sorted(self.streams):
            expected = old_preview.allocations.get(stream_id, 0)
            actual = new_preview.allocations.get(stream_id, 0)
            if expected or actual:
                deviations.append(
                    {
                        "stream_id": stream_id,
                        "expected_before_bytes": expected,
                        "actual_after_bytes": actual,
                        "deviation_bytes": actual - expected,
                    }
                )
        self.strategy_switches.append(
            StrategySwitch(
                round_index=self.round_index,
                event_seq=trigger_seq,
                from_strategy=old_strategy,
                to_strategy=new_strategy,
                reason_code=selection.reason_code,
                reason=selection.reason,
                stream_deviations=deviations,
            )
        )

    def _send_data(self, stream: StreamRuntime, size: int, trigger_seq: int | None) -> FrameRecord:
        connection_before = self.connection_window
        stream_before = stream.window
        if size > connection_before or size > stream_before or size > stream.body_remaining:
            raise WorkbenchError(
                "E_FLOW",
                "internal DATA allocation exceeds an available window",
                field="data_bytes",
                event_seq=trigger_seq,
                details={
                    "stream_id": stream.stream_id,
                    "size": size,
                    "connection_window": connection_before,
                    "stream_window": stream_before,
                    "body_remaining": stream.body_remaining,
                },
            )
        stream.body_remaining -= size
        self.connection_window -= size
        stream.window -= size
        stream.granted_bytes += size
        stream.granted_frames += 1
        end_stream = stream.body_remaining == 0
        frame_seq = self.last_frame_seq + 1
        record = FrameRecord(
            frame_seq=frame_seq,
            event_seq=trigger_seq,
            round_index=self.round_index,
            strategy=self.current_strategy,
            frame_type="DATA",
            stream_id=stream.stream_id,
            flags=FLAG_END_STREAM if end_stream else 0,
            length=size,
            data_bytes=size,
            connection_window_before=connection_before,
            connection_window_after=self.connection_window,
            stream_window_before=stream_before,
            stream_window_after=stream.window,
            business_code="BODY_COMPLETE" if end_stream else "DATA_SENT",
            end_stream=end_stream,
        )
        self._append_frame(record)
        stream.last_grant_frame_seq = frame_seq
        if end_stream:
            self._record_transition(
                stream,
                HALF_CLOSED_LOCAL,
                event_seq=trigger_seq,
                frame_seq=frame_seq,
                reason="request_body_complete",
            )
        return record

    @staticmethod
    def _apportion(
        total: int,
        streams: list[StreamRuntime],
        weights: Mapping[int, int],
    ) -> dict[int, int]:
        if total <= 0 or not streams:
            return {stream.stream_id: 0 for stream in streams}
        weight_sum = sum(max(0, weights.get(stream.stream_id, 0)) for stream in streams)
        if weight_sum <= 0:
            weights = {stream.stream_id: 1 for stream in streams}
            weight_sum = len(streams)
        result: dict[int, int] = {}
        remainders: list[tuple[int, int, int]] = []
        used = 0
        for stream in streams:
            weight = max(0, weights.get(stream.stream_id, 0))
            quotient, remainder = divmod(total * weight, weight_sum)
            result[stream.stream_id] = quotient
            used += quotient
            remainders.append((remainder, stream.priority, stream.stream_id))
        leftover = total - used
        remainders.sort(key=lambda item: (-item[0], item[1], item[2]))
        for _, _, stream_id in remainders[:leftover]:
            result[stream_id] += 1
        return result

    def _update_fairness(
        self,
        eligible: list[StreamRuntime],
        actual_bytes: int,
        strategy: str,
        round_start_windows: Mapping[int, int],
    ) -> None:
        if actual_bytes <= 0:
            return
        if strategy == ROUND_ROBIN:
            weights = {stream.stream_id: 1 for stream in eligible}
        else:
            weights = {
                stream.stream_id: max(1, round_start_windows.get(stream.stream_id, 0))
                for stream in eligible
            }
        expected = self._apportion(actual_bytes, eligible, weights)
        for stream in eligible:
            stream.expected_bytes += expected.get(stream.stream_id, 0)

    def _episode_warning(
        self,
        stream: StreamRuntime,
        granted_this_round: set[int],
        trigger_seq: int | None,
    ) -> None:
        granted_streams = [self.streams[stream_id] for stream_id in granted_this_round]
        beneficiaries = [item for item in granted_streams if item.priority < stream.priority]
        source: StreamRuntime | None = None
        for candidate in sorted(
            beneficiaries,
            key=lambda item: (-item.last_burst_increment, item.stream_id),
        ):
            if candidate.last_burst_event_seq is None:
                continue
            if stream.starve_start_event_seq is not None and candidate.last_burst_event_seq < stream.starve_start_event_seq:
                continue
            source = candidate
            break
        deviation_ratio = (
            (stream.expected_bytes - stream.granted_bytes) / stream.expected_bytes
            if stream.expected_bytes > 0
            else 0.0
        )
        warning: dict[str, Any] = {
            "type": "starvation",
            "stream_id": stream.stream_id,
            "priority": stream.priority,
            "start_round": stream.starve_start_round,
            "detected_round": self.round_index,
            "start_event_seq": stream.starve_start_event_seq,
            "detected_event_seq": trigger_seq,
            "gap_rounds": stream.no_grant_rounds,
            "threshold_rounds": self.options.starvation_rounds,
            "deviation_ratio": round(deviation_ratio, 6),
            "deviation_ratio_threshold": self.options.deviation_ratio_threshold,
            "source_event_seq": None,
            "beneficiary_stream_id": None,
            "beneficiary_priority": None,
            "burst_increment": 0,
        }
        if source is not None and deviation_ratio > self.options.deviation_ratio_threshold:
            warning.update(
                {
                    "type": "window_burst_preemption",
                    "source_event_seq": source.last_burst_event_seq,
                    "beneficiary_stream_id": source.stream_id,
                    "beneficiary_priority": source.priority,
                    "burst_increment": source.last_burst_increment,
                }
            )
        key = (
            warning["type"],
            stream.stream_id,
            warning["start_round"],
            warning["source_event_seq"],
        )
        if key not in self._closed_warning_keys:
            self._closed_warning_keys.add(key)
            self.warnings.append(warning)

    def schedule_round(self, trigger_seq: int | None) -> bool:
        self._open_available(trigger_seq)
        eligible = self._eligible()
        if not eligible:
            return False
        self.round_index += 1
        selection, rr_preview, prop_preview = self._select_strategy(eligible, trigger_seq)
        old_strategy = self.current_strategy
        if selection.strategy != old_strategy:
            old_preview = rr_preview if old_strategy == ROUND_ROBIN else prop_preview
            new_preview = rr_preview if selection.strategy == ROUND_ROBIN else prop_preview
            self._record_switch(
                old_strategy,
                selection.strategy,
                selection,
                old_preview,
                new_preview,
                trigger_seq,
            )
        self.current_strategy = selection.strategy
        self.selection = selection
        selected_preview = rr_preview if selection.strategy == ROUND_ROBIN else prop_preview

        round_start_windows = {stream.stream_id: stream.window for stream in eligible}
        actual_by_stream: dict[int, int] = {}
        frames_sent = 0
        for stream_id, size in selected_preview.sequence:
            stream = self.streams[stream_id]
            size = min(
                size,
                stream.body_remaining,
                stream.window,
                self.connection_window,
            )
            if frames_sent >= self.spec.connection.scheduling_quantum:
                break
            if size <= 0:
                continue
            record = self._send_data(stream, size, trigger_seq)
            actual_by_stream[stream_id] = actual_by_stream.get(stream_id, 0) + record.data_bytes
            frames_sent += 1

        actual_bytes = sum(actual_by_stream.values())
        self._update_fairness(eligible, actual_bytes, selection.strategy, round_start_windows)
        granted_ids = {stream_id for stream_id, bytes_sent in actual_by_stream.items() if bytes_sent > 0}
        for stream in eligible:
            if stream.stream_id in granted_ids:
                stream.no_grant_rounds = 0
                stream.starve_start_round = None
                stream.starve_start_event_seq = None
            else:
                if stream.no_grant_rounds == 0:
                    stream.starve_start_round = self.round_index
                    stream.starve_start_event_seq = trigger_seq
                stream.no_grant_rounds += 1
                stream.max_grant_gap_rounds = max(stream.max_grant_gap_rounds, stream.no_grant_rounds)
                if stream.no_grant_rounds == self.options.starvation_rounds:
                    self._episode_warning(stream, granted_ids, trigger_seq)
        # Preserve deterministic insertion order for diagnostics and snapshots.
        return frames_sent > 0

    # ------------------------------------------------------------------
    # Script execution and reporting
    # ------------------------------------------------------------------
    def step(self) -> bool:
        if self.event_cursor >= len(self.spec.events):
            return False
        event = self.spec.events[self.event_cursor]
        self.apply_event(event)
        self.event_cursor += 1
        self.schedule_round(event.seq)
        return True

    def run(self, *, drain: bool = True) -> "Engine":
        while self.step():
            pass
        if drain:
            last_seq = self.spec.events[-1].seq if self.spec.events else None
            guard = 0
            while guard < self.options.deadline_rounds:
                if not self.schedule_round(last_seq):
                    break
                guard += 1
            if guard >= self.options.deadline_rounds:
                raise WorkbenchError(
                    "E_STATE",
                    "scheduler deadline exceeded without reaching quiescence",
                    field="deadline_rounds",
                    event_seq=last_seq,
                )
        return self

    def run_remaining(self, *, drain: bool = True) -> "Engine":
        return self.run(drain=drain)

    def fairness_report(self) -> list[dict[str, Any]]:
        report: list[dict[str, Any]] = []
        for stream_id in sorted(self.streams):
            stream = self.streams[stream_id]
            deviation = stream.granted_bytes - stream.expected_bytes
            ratio = (
                deviation / stream.expected_bytes
                if stream.expected_bytes > 0
                else (0.0 if stream.granted_bytes == 0 else 1.0)
            )
            report.append(
                {
                    "stream_id": stream_id,
                    "priority": stream.priority,
                    "expected_bytes": stream.expected_bytes,
                    "actual_bytes": stream.granted_bytes,
                    "deviation_bytes": deviation,
                    "deviation_ratio": round(ratio, 6),
                    "granted_frames": stream.granted_frames,
                    "max_grant_gap_rounds": stream.max_grant_gap_rounds,
                    "state": stream.state,
                }
            )
        return report

    def starvation_warnings(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.warnings]

    def preemption_warnings(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.warnings if item["type"] == "window_burst_preemption"]

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.spec.session_id,
            "declared_version": self.spec.declared_version,
            "content_digest": self.spec.content_digest,
            "streams": len(self.streams),
            "events": len(self.spec.events),
            "scheduling_quantum": self.spec.connection.scheduling_quantum,
            "frames": len(self.frames),
            "transitions": len(self.transitions),
            "warnings": len(self.warnings),
            "preemption_warnings": len(self.preemption_warnings()),
            "strategy_switches": len(self.strategy_switches),
            "current_strategy": self.current_strategy,
            "strategy_selection": self.selection.to_dict(),
            "connection_window": self.connection_window,
            "next_event_seq": self.next_event_seq,
            "run_digest": self.run_digest(),
        }

    def run_digest(self) -> str:
        return digest_value(
            {
                "content_digest": self.spec.content_digest,
                "declared_version": self.spec.declared_version,
                "connection_window": self.connection_window,
                "streams": [self.streams[stream_id].to_dict() for stream_id in sorted(self.streams)],
                "current_strategy": self.current_strategy,
                "pinned_strategy": self.pinned_strategy,
                "strategy_switches": [switch.to_dict() for switch in self.strategy_switches],
                "frames": [frame.to_dict() for frame in self.frames],
                "transitions": [item.to_dict() for item in self.transitions],
                "warnings": [dict(item) for item in self.warnings],
                "event_cursor": self.event_cursor,
                "round_index": self.round_index,
                "goaway_seen": self.goaway_seen,
                "goaway_last_stream_id": self.goaway_last_stream_id,
                "goaway_error_code": self.goaway_error_code,
            }
        )


def run_session(spec: SessionSpec, *, options: EngineOptions | None = None) -> Engine:
    return Engine(spec, options=options).run()
