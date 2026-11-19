"""Deterministic offline HTTP/2 multiplexing state machine."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from .declaration import FrameScript, POLICIES, RequestSpec, SessionDeclaration
from .errors import FrameError, ToolError
from .frames import Frame, HeaderField, PrioritySpec, Setting, encode_frame
from .ledger import Ledger
from .priority import PriorityNode, PriorityTree, fair_quotas, quota_rows


STREAM_STATES = (
    "idle",
    "open",
    "half_closed_local",
    "half_closed_remote",
    "closed",
)

ACTIVE_STATES = {"open", "half_closed_local", "half_closed_remote"}
LOCAL_SEND_STATES = {"open", "half_closed_remote"}
REMOTE_RECV_STATES = {"open", "half_closed_local"}

MAX_WINDOW = (1 << 31) - 1


@dataclass
class StreamRuntime:
    spec: RequestSpec
    state: str = "idle"
    send_window: int = 0
    recv_window: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    headers_sent: bool = False
    local_closed: bool = False
    remote_closed: bool = False

    @property
    def stream_id(self) -> int:
        return self.spec.stream_id

    @property
    def body_bytes(self) -> int:
        return self.spec.body_bytes

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.body_bytes - self.bytes_sent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "state": self.state,
            "send_window": self.send_window,
            "recv_window": self.recv_window,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "headers_sent": self.headers_sent,
            "local_closed": self.local_closed,
            "remote_closed": self.remote_closed,
        }


class H2Session:
    """A fully local, clock-deterministic HTTP/2 session simulator."""

    def __init__(
        self,
        declaration: SessionDeclaration,
        script: FrameScript,
        *,
        strategy: str = "weighted_round_robin",
        starvation_rounds: int = 3,
    ) -> None:
        if strategy not in POLICIES:
            raise ValueError(f"strategy must be one of {', '.join(POLICIES)}")
        if script.declaration_version != declaration.declaration_version:
            raise ValueError("script and declaration versions differ")
        self.declaration = declaration
        self.script = script
        self.strategy = strategy
        self.initial_strategy = strategy
        self.starvation_rounds = max(1, int(starvation_rounds))
        self.clock_ms = 0
        self.frame_seq = 0
        self.script_index = 0
        self.schedule_round = 0
        self.started = False
        self.finalized = False
        self.goaway_received = False
        self.goaway_last_stream_id: int | None = None
        self.goaway_error_code: int | None = None
        self.peer_initial_stream_window = declaration.server.stream_window
        self.peer_max_frame_size = declaration.max_frame_size
        self.peer_max_concurrent = declaration.max_concurrent_streams
        self.send_conn_window = declaration.server.connection_window
        self.recv_conn_window = declaration.client.connection_window
        self.streams: dict[int, StreamRuntime] = {
            item.stream_id: StreamRuntime(
                spec=item,
                send_window=declaration.server.stream_window,
                recv_window=declaration.client.stream_window,
            )
            for item in declaration.requests
        }
        self.queue: list[int] = sorted(self.streams)
        self.priority = PriorityTree(
            PriorityNode(item.stream_id, item.parent_id, item.weight)
            for item in declaration.requests
        )
        self.ledger = Ledger()
        self._open_window_waits: dict[str, dict[str, Any]] = {}
        self._active_queue_wait: dict[str, Any] | None = None
        self._current_round: dict[str, Any] | None = None
        self._priority_update_seq = 0
        self.restored_from: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public progression API
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.started:
            return
        if self.restored_from is not None:
            self.started = True
            return
        self.started = True
        settings = (
            Setting(2, 0),
            Setting(3, self.declaration.max_concurrent_streams),
            Setting(4, self.declaration.client.stream_window),
            Setting(5, self.declaration.max_frame_size),
        )
        self._send_local(Frame.settings_frame(settings), script_index=-1, reason="client_settings")

    def process_next_event(self) -> bool:
        self.start()
        if self.script_index >= len(self.script.events):
            self.finalize()
            return False
        event = self.script.events[self.script_index]
        if event.kind == "recv":
            assert event.frame is not None
            self._receive_frame(event.frame, event.index)
        elif event.kind == "schedule":
            self._run_schedule_event(event.index, int(event.max_frames or 1))
        elif event.kind == "switch_strategy":
            self._switch_strategy(str(event.strategy), event.index)
        else:  # pragma: no cover - loader validates
            raise AssertionError(event.kind)
        self.script_index += 1
        if self.script_index >= len(self.script.events):
            self.finalize()
        return True

    def run(self, *, stop_after_events: int | None = None) -> "H2Session":
        self.start()
        target = len(self.script.events) if stop_after_events is None else min(
            len(self.script.events), max(0, stop_after_events)
        )
        while self.script_index < target:
            self.process_next_event()
        if stop_after_events is None or self.script_index >= len(self.script.events):
            self.finalize()
        return self

    def set_strategy(self, strategy: str, *, script_index: int | None = None) -> None:
        self._switch_strategy(strategy, self.script_index if script_index is None else script_index)

    def finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        if self._current_round is not None:
            self._finish_round()
        for wait in self._open_window_waits.values():
            wait["end_ms"] = self.clock_ms
            wait["end_seq"] = self.ledger.next_seq - 1
            wait["closed_by"] = "session_finalized"
        self._open_window_waits.clear()
        if self._active_queue_wait is not None:
            self._active_queue_wait["end_ms"] = self.clock_ms
            self._active_queue_wait["end_seq"] = self.ledger.next_seq - 1
            self._active_queue_wait["closed_by"] = "session_finalized"
            self._active_queue_wait = None

    # ------------------------------------------------------------------
    # Frame receive handling
    # ------------------------------------------------------------------
    def _receive_frame(self, frame: Frame, script_index: int) -> None:
        before = self._snapshot_windows(frame.stream_id)
        state_before = self._state_of(frame.stream_id)
        code = "OK"
        reason = None
        transition_events: list[tuple[int, str, str]] = []
        try:
            if frame.type == "SETTINGS":
                self._receive_settings(frame)
            elif frame.type == "HEADERS":
                transition_events = self._receive_headers(frame)
            elif frame.type == "DATA":
                transition_events = self._receive_data(frame)
            elif frame.type == "WINDOW_UPDATE":
                self._receive_window_update(frame, script_index)
            elif frame.type == "RST_STREAM":
                transition_events = self._receive_rst_stream(frame)
            elif frame.type == "GOAWAY":
                self._receive_goaway(frame)
            else:  # pragma: no cover - codec validates
                raise FrameError("unsupported frame type at runtime", frame_type=frame.type, offset=0)
        except ToolError as exc:
            code = exc.code
            reason = exc.to_dict()
        self._record_frame(
            direction="recv",
            frame=frame,
            script_index=script_index,
            before=before,
            state_before=state_before,
            code=code,
            reason=reason,
        )
        for _, target_state, trigger in transition_events:
            self._append_transition(frame.stream_id, state_before, target_state, trigger, script_index)
            if target_state == "closed":
                self._close_queue_wait("max_concurrent_slot_released")
                self._close_window_wait(
                    f"stream_send_window_exhausted:{frame.stream_id}",
                    f"stream_closed:{frame.stream_id}",
                )
        if frame.type == "SETTINGS" and not frame.flag("ACK") and code == "OK":
            self._send_local(Frame.settings_frame(ack=True), script_index=script_index, reason="settings_ack")

    def _receive_settings(self, frame: Frame) -> None:
        if frame.flag("ACK"):
            if frame.settings:
                raise FrameError("SETTINGS ACK payload must be empty", frame_type="SETTINGS", offset=9)
            return
        for setting in frame.settings:
            if setting.identifier == 4 and setting.value > MAX_WINDOW:
                raise FrameError(
                    "SETTINGS_INITIAL_WINDOW_SIZE exceeds 2^31-1",
                    frame_type="SETTINGS",
                    offset=9,
                )
            if setting.identifier == 5 and not 16384 <= setting.value <= 16777215:
                raise FrameError(
                    "SETTINGS_MAX_FRAME_SIZE outside 16384..16777215",
                    frame_type="SETTINGS",
                    offset=9,
                )
        for setting in frame.settings:
            if setting.identifier == 3:
                self.peer_max_concurrent = setting.value
            elif setting.identifier == 4:
                delta = setting.value - self.peer_initial_stream_window
                for stream in self.streams.values():
                    stream.send_window = max(0, min(MAX_WINDOW, stream.send_window + delta))
                self.peer_initial_stream_window = setting.value
            elif setting.identifier == 5:
                self.peer_max_frame_size = setting.value
            # HEADER_TABLE_SIZE and ENABLE_PUSH are accepted as no-ops.

    def _receive_headers(self, frame: Frame) -> list[tuple[int, str, str]]:
        stream = self.streams.get(frame.stream_id)
        if stream is None or stream.state in {"idle", "closed"}:
            raise ToolError(
                "E_STREAM",
                "HEADERS received for an idle or closed stream",
                stream_id=frame.stream_id,
                details={"state": stream.state if stream else "absent"},
            )
        if stream.state == "half_closed_remote":
            raise ToolError(
                "E_STREAM",
                "remote side is already closed on this stream",
                stream_id=frame.stream_id,
            )
        transitions: list[tuple[int, str, str]] = []
        if frame.priority is not None:
            active = [sid for sid, item in sorted(self.streams.items()) if item.state != "closed"]
            try:
                change = self.priority.update(
                    frame.stream_id,
                    parent_id=frame.priority.depends_on,
                    weight=frame.priority.weight,
                    exclusive=frame.priority.exclusive,
                    active_streams=active,
                )
            except FrameError as exc:
                raise ToolError(
                    "E_FRAME",
                    exc.message,
                    frame_type="HEADERS",
                    offset=9,
                    stream_id=frame.stream_id,
                    details={"priority_basis": exc.to_dict()},
                ) from exc
            self._priority_update_seq += 1
            self.ledger.priority_events.append(
                {
                    "seq": self._priority_update_seq,
                    "frame_seq": self.frame_seq + 1,
                    "clock_ms": self.clock_ms,
                    "script_index": self.script_index,
                    "source": "peer_headers",
                    **change,
                }
            )
        if frame.flag("END_STREAM"):
            stream.remote_closed = True
            target = "closed" if stream.local_closed else "half_closed_remote"
            stream.state = target
            transitions.append((frame.stream_id, target, "recv_headers_end_stream"))
        return transitions

    def _receive_data(self, frame: Frame) -> list[tuple[int, str, str]]:
        stream = self.streams.get(frame.stream_id)
        if stream is None or stream.state in {"idle", "closed"}:
            raise ToolError(
                "E_STREAM",
                "DATA received for an idle or closed stream",
                stream_id=frame.stream_id,
                details={"state": stream.state if stream else "absent"},
            )
        if stream.state == "half_closed_remote":
            raise ToolError(
                "E_STREAM",
                "remote side is already closed on this stream",
                stream_id=frame.stream_id,
            )
        amount = len(frame.body)
        if amount > self.recv_conn_window or amount > stream.recv_window:
            raise ToolError(
                "E_FRAME",
                "received DATA exceeds a connection or stream receive window",
                frame_type="DATA",
                offset=9,
                stream_id=frame.stream_id,
                details={
                    "amount": amount,
                    "connection_window": self.recv_conn_window,
                    "stream_window": stream.recv_window,
                },
            )
        self.recv_conn_window -= amount
        stream.recv_window -= amount
        stream.bytes_received += amount
        transitions: list[tuple[int, str, str]] = []
        if frame.flag("END_STREAM"):
            stream.remote_closed = True
            target = "closed" if stream.local_closed else "half_closed_remote"
            stream.state = target
            transitions.append((frame.stream_id, target, "recv_data_end_stream"))
        return transitions

    def _receive_window_update(self, frame: Frame, script_index: int) -> None:
        increment = int(frame.increment or 0)
        if frame.stream_id == 0:
            if self.send_conn_window + increment > MAX_WINDOW:
                raise ToolError(
                    "E_FRAME",
                    "connection WINDOW_UPDATE would exceed 2^31-1",
                    frame_type="WINDOW_UPDATE",
                    offset=9,
                    stream_id=0,
                )
            self.send_conn_window += increment
            self._close_window_wait("connection_send_window_exhausted", "WINDOW_UPDATE(0)")
            return
        stream = self.streams.get(frame.stream_id)
        if stream is None or stream.state in {"idle", "closed"}:
            raise ToolError(
                "E_STREAM",
                "WINDOW_UPDATE received for an idle or closed stream",
                stream_id=frame.stream_id,
                details={"state": stream.state if stream else "absent"},
            )
        if stream.send_window + increment > MAX_WINDOW:
            raise ToolError(
                "E_FRAME",
                "stream WINDOW_UPDATE would exceed 2^31-1",
                frame_type="WINDOW_UPDATE",
                offset=9,
                stream_id=frame.stream_id,
            )
        stream.send_window += increment
        self._close_window_wait(f"stream_send_window_exhausted:{frame.stream_id}", f"WINDOW_UPDATE({frame.stream_id})")

    def _receive_rst_stream(self, frame: Frame) -> list[tuple[int, str, str]]:
        stream = self.streams.get(frame.stream_id)
        if stream is None or stream.state in {"idle", "closed"}:
            raise ToolError(
                "E_STREAM",
                "RST_STREAM received for an idle or closed stream",
                stream_id=frame.stream_id,
                details={"state": stream.state if stream else "absent"},
            )
        stream.local_closed = True
        stream.remote_closed = True
        stream.state = "closed"
        self._close_window_wait(
            f"stream_send_window_exhausted:{frame.stream_id}",
            "RST_STREAM",
        )
        changes = self.priority.reparent_children(frame.stream_id)
        for change in changes:
            self.ledger.priority_events.append(
                {
                    "seq": self._priority_update_seq + 1,
                    "frame_seq": self.frame_seq + 1,
                    "clock_ms": self.clock_ms,
                    "script_index": self.script_index,
                    "source": "parent_rst_stream",
                    **change,
                }
            )
            self._priority_update_seq += 1
        self._close_queue_wait("max_concurrent_slot_released")
        return [(frame.stream_id, "closed", "recv_rst_stream")]

    def _receive_goaway(self, frame: Frame) -> None:
        self.goaway_received = True
        self.goaway_last_stream_id = int(frame.last_stream_id or 0)
        self.goaway_error_code = int(frame.error_code or 0)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    def _run_schedule_event(self, script_index: int, max_frames: int) -> None:
        self._begin_round(script_index)
        for _ in range(max_frames):
            if not self._schedule_one(script_index):
                break
        self._finish_round()

    def _begin_round(self, script_index: int) -> None:
        if self._current_round is not None:
            self._finish_round()
        self.schedule_round += 1
        eligible = sorted(
            sid
            for sid, stream in self.streams.items()
            if stream.state in LOCAL_SEND_STATES
            and stream.remaining_bytes > 0
            and stream.send_window > 0
        )
        self._current_round = {
            "round_id": self.schedule_round,
            "script_index": script_index,
            "clock_ms": self.clock_ms,
            "eligible_streams": eligible,
            "sent_streams": [],
            "data_bytes": 0,
        }

    def _finish_round(self) -> None:
        if self._current_round is None:
            return
        current = self._current_round
        if current["data_bytes"] > 0:
            sent = sorted(set(current["sent_streams"]))
            current["missed_streams"] = [
                sid for sid in current["eligible_streams"] if sid not in sent
            ]
        else:
            current["missed_streams"] = []
        self.ledger.schedule_rounds.append(current)
        self._current_round = None

    def _schedule_one(self, script_index: int) -> bool:
        if self.goaway_received:
            return False
        active_count = sum(1 for stream in self.streams.values() if stream.state in ACTIVE_STATES)
        if self.queue:
            if active_count < self.active_stream_limit:
                if self._active_queue_wait is not None:
                    self._close_queue_wait("max_concurrent_slot_released")
                stream_id = self.queue.pop(0)
                self._send_headers(stream_id, script_index)
                return True
            self._open_queue_wait(
                script_index,
                reason="MAX_CONCURRENT_STREAMS",
                detail={
                    "limit": self.active_stream_limit,
                    "active": active_count,
                    "queued_stream_ids": list(self.queue),
                },
            )

        candidates = [
            stream
            for stream in self.streams.values()
            if stream.state in LOCAL_SEND_STATES and stream.remaining_bytes > 0
        ]
        if not candidates:
            return False
        candidates.sort(key=lambda item: item.stream_id)
        if self.send_conn_window <= 0:
            self._open_window_wait(
                script_index,
                "connection_send_window_exhausted",
                [item.stream_id for item in candidates],
            )
            return False
        window_candidates = [item for item in candidates if item.send_window > 0]
        if not window_candidates:
            for stream in candidates:
                self._open_window_wait(
                    script_index,
                    f"stream_send_window_exhausted:{stream.stream_id}",
                    [stream.stream_id],
                )
            return False
        selected = self._select_stream(window_candidates)
        amount = min(
            selected.remaining_bytes,
            self.send_conn_window,
            selected.send_window,
            self.peer_max_frame_size,
            self.declaration.max_frame_size,
        )
        if amount <= 0:
            return False
        self._send_data(selected, amount, script_index)
        return True

    @property
    def active_stream_limit(self) -> int:
        return min(self.declaration.max_concurrent_streams, self.peer_max_concurrent)

    def _select_stream(self, candidates: list[StreamRuntime]) -> StreamRuntime:
        if self.strategy == "window_priority":
            return sorted(
                candidates,
                key=lambda item: (-item.send_window, -self.priority.weight_of(item.stream_id), item.stream_id),
            )[0]
        shares = self.priority.shares()
        fallback = Fraction(1, max(1, len(candidates)))

        def key(stream: StreamRuntime) -> tuple[Fraction, int, int]:
            share = shares.get(stream.stream_id, fallback)
            if share <= 0:
                share = fallback
            return (Fraction(stream.bytes_sent, 1) / share, -self.priority.weight_of(stream.stream_id), stream.stream_id)

        return sorted(candidates, key=key)[0]

    def _send_headers(self, stream_id: int, script_index: int) -> None:
        stream = self.streams[stream_id]
        if stream.state != "idle":
            raise RuntimeError(f"stream {stream_id} is not idle")
        end_stream = stream.body_bytes == 0
        headers = (
            HeaderField.indexed(2),
            HeaderField.indexed(6),
            HeaderField.literal_indexed_name(4, f"/stream/{stream_id}".encode("ascii")),
            HeaderField.literal_indexed_name(1, b"localhost"),
        )
        frame = Frame.headers_frame(
            stream_id,
            headers,
            end_stream=end_stream,
            priority=PrioritySpec(
                depends_on=self.priority.parent_of(stream_id),
                weight=self.priority.weight_of(stream_id),
            ),
        )
        before = self._snapshot_windows(stream_id)
        state_before = stream.state
        stream.headers_sent = True
        stream.state = "half_closed_local" if end_stream else "open"
        if end_stream:
            stream.local_closed = True
        self._send_local(frame, script_index=script_index, reason="open_stream", before=before, state_before=state_before)
        self._append_transition(stream_id, state_before, "open", "send_headers", script_index)
        if end_stream:
            self._append_transition(stream_id, "open", "half_closed_local", "send_headers_end_stream", script_index)

    def _send_data(self, stream: StreamRuntime, amount: int, script_index: int) -> None:
        before = self._snapshot_windows(stream.stream_id)
        state_before = stream.state
        end_stream = amount == stream.remaining_bytes
        body = bytes([0x41 + (stream.stream_id % 26)]) * amount
        frame = Frame.data_frame(stream.stream_id, body, end_stream=end_stream)
        self.send_conn_window -= amount
        stream.send_window -= amount
        stream.bytes_sent += amount
        if self.send_conn_window < 0 or stream.send_window < 0:  # invariant guard
            raise AssertionError("flow-control windows must never become negative")
        if end_stream:
            stream.local_closed = True
            stream.state = "closed" if stream.remote_closed else "half_closed_local"
        self._send_local(
            frame,
            script_index=script_index,
            reason="data",
            before=before,
            state_before=state_before,
            flow_bytes=amount,
        )
        if self._current_round is not None:
            self._current_round["sent_streams"].append(stream.stream_id)
            self._current_round["data_bytes"] += amount
        if end_stream:
            target = "closed" if state_before == "half_closed_remote" else "half_closed_local"
            self._append_transition(stream.stream_id, state_before, target, "send_data_end_stream", script_index)
            if target == "closed":
                self._close_queue_wait("max_concurrent_slot_released")

    def _switch_strategy(self, strategy: str, script_index: int) -> None:
        if strategy not in POLICIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        old = self.strategy
        self.strategy = strategy
        self.ledger.append_control(
            {
                "clock_ms": self.clock_ms,
                "script_index": script_index,
                "kind": "switch_strategy",
                "old": old,
                "new": strategy,
            }
        )
        self.clock_ms += 1

    # ------------------------------------------------------------------
    # Sending and ledger
    # ------------------------------------------------------------------
    def _send_local(
        self,
        frame: Frame,
        *,
        script_index: int,
        reason: str,
        before: dict[str, Any] | None = None,
        state_before: str | None = None,
        flow_bytes: int = 0,
    ) -> None:
        local_max = min(self.declaration.max_frame_size, self.peer_max_frame_size)
        wire = encode_frame(frame, max_frame_size=local_max, preserve_raw=False)
        if before is None:
            before = self._snapshot_windows(frame.stream_id)
        if state_before is None:
            state_before = self._state_of(frame.stream_id)
        self._record_frame(
            direction="send",
            frame=frame,
            script_index=script_index,
            before=before,
            state_before=state_before,
            code="OK",
            reason={"message": reason} if reason else None,
            wire_bytes=len(wire),
            flow_bytes=flow_bytes,
        )

    def _record_frame(
        self,
        *,
        direction: str,
        frame: Frame,
        script_index: int,
        before: dict[str, Any],
        state_before: str | None,
        code: str,
        reason: dict[str, Any] | None,
        wire_bytes: int | None = None,
        flow_bytes: int | None = None,
    ) -> dict[str, Any]:
        self.frame_seq += 1
        after = self._snapshot_windows(frame.stream_id)
        state_after = self._state_of(frame.stream_id)
        payload_bytes = len(frame.payload)
        audited_flow_bytes = len(frame.body) if frame.type == "DATA" else 0
        if flow_bytes is not None:
            audited_flow_bytes = flow_bytes
        if wire_bytes is None:
            try:
                wire_bytes = len(encode_frame(frame, preserve_raw=True))
            except ToolError:
                wire_bytes = 9 + len(frame.payload)
        record = self.ledger.append_record(
            {
                "frame_seq": self.frame_seq,
                "clock_ms": self.clock_ms,
                "script_index": script_index,
                "direction": direction,
                "frame_type": frame.type,
                "flags": frame.flag_names(),
                "target_stream_id": frame.stream_id,
                "payload_bytes": payload_bytes,
                "flow_bytes": audited_flow_bytes,
                "wire_bytes": wire_bytes,
                "conn_send_window_before": before["conn_send_window"],
                "conn_send_window_after": after["conn_send_window"],
                "stream_send_window_before": before["stream_send_window"],
                "stream_send_window_after": after["stream_send_window"],
                "conn_recv_window_before": before["conn_recv_window"],
                "conn_recv_window_after": after["conn_recv_window"],
                "stream_recv_window_before": before["stream_recv_window"],
                "stream_recv_window_after": after["stream_recv_window"],
                "state_before": state_before,
                "state_after": state_after,
                "code": code,
                "reason": reason,
            }
        )
        self.clock_ms += 1
        return record

    def _append_transition(
        self,
        stream_id: int,
        state_before: str | None,
        state_after: str,
        trigger: str,
        script_index: int,
    ) -> None:
        if state_before == state_after:
            return
        self.ledger.append_transition(
            {
                "frame_seq": self.frame_seq,
                "clock_ms": self.clock_ms - 1,
                "script_index": script_index,
                "stream_id": stream_id,
                "from": state_before,
                "to": state_after,
                "trigger": trigger,
            }
        )

    def _snapshot_windows(self, stream_id: int) -> dict[str, int | None]:
        stream = self.streams.get(stream_id) if stream_id else None
        return {
            "conn_send_window": self.send_conn_window,
            "conn_recv_window": self.recv_conn_window,
            "stream_send_window": stream.send_window if stream else None,
            "stream_recv_window": stream.recv_window if stream else None,
        }

    def _state_of(self, stream_id: int) -> str | None:
        if not stream_id:
            return None
        stream = self.streams.get(stream_id)
        return stream.state if stream else "absent"

    # ------------------------------------------------------------------
    # Wait accounting
    # ------------------------------------------------------------------
    def _open_window_wait(self, script_index: int, reason: str, stream_ids: list[int]) -> None:
        existing = self._open_window_waits.get(reason)
        if existing is None:
            existing = {
                "wait_id": self.ledger.next_wait(),
                "reason": reason,
                "affected_streams": sorted(set(stream_ids)),
                "start_ms": self.clock_ms,
                "start_seq": self.ledger.next_seq,
                "end_ms": None,
                "end_seq": None,
                "closed_by": None,
                "script_index": script_index,
            }
            self._open_window_waits[reason] = existing
            self.ledger.empty_waits.append(existing)
        else:
            existing["affected_streams"] = sorted(
                set(existing["affected_streams"]) | set(stream_ids)
            )
            existing["end_seq"] = self.ledger.next_seq

    def _close_window_wait(self, reason: str, closed_by: str) -> None:
        wait = self._open_window_waits.pop(reason, None)
        if wait is None:
            return
        wait["end_ms"] = self.clock_ms + 1
        wait["end_seq"] = self.ledger.next_seq
        wait["closed_by"] = closed_by

    def _open_queue_wait(self, script_index: int, *, reason: str, detail: dict[str, Any]) -> None:
        if self._active_queue_wait is None:
            self._active_queue_wait = {
                "wait_id": self.ledger.next_queue_wait(),
                "reason": reason,
                "detail": detail,
                "start_ms": self.clock_ms,
                "start_seq": self.ledger.next_seq,
                "end_ms": None,
                "end_seq": None,
                "closed_by": None,
                "script_index": script_index,
            }
            self.ledger.queue_waits.append(self._active_queue_wait)

    def _close_queue_wait(self, closed_by: str) -> None:
        if self._active_queue_wait is None:
            return
        self._active_queue_wait["end_ms"] = self.clock_ms
        self._active_queue_wait["end_seq"] = self.ledger.next_seq
        self._active_queue_wait["closed_by"] = closed_by
        self._active_queue_wait = None

    # ------------------------------------------------------------------
    # Diagnostics, state export, and summary
    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, Any]:
        body_bytes = {sid: stream.body_bytes for sid, stream in sorted(self.streams.items())}
        actual_bytes = {sid: stream.bytes_sent for sid, stream in sorted(self.streams.items())}
        quota = quota_rows(self.priority, body_bytes, actual_bytes)
        starvation = self._starvation_spans()
        return {
            "policy": self.strategy,
            "priority_rows": quota,
            "starvation_threshold_rounds": self.starvation_rounds,
            "starvation": starvation,
            "priority_events": list(self.ledger.priority_events),
        }

    def _starvation_spans(self) -> list[dict[str, Any]]:
        data_rounds = [item for item in self.ledger.schedule_rounds if item.get("data_bytes", 0) > 0]
        spans: list[dict[str, Any]] = []
        for stream_id in sorted(self.streams):
            current_start: int | None = None
            previous_round: int | None = None
            for item in data_rounds:
                round_id = int(item["round_id"])
                missed = stream_id in item.get("missed_streams", [])
                consecutive = previous_round is None or round_id == previous_round + 1
                if missed and consecutive:
                    if current_start is None:
                        current_start = round_id
                elif missed and not consecutive:
                    if current_start is not None:
                        self._maybe_starvation_span(spans, stream_id, current_start, previous_round)
                    current_start = round_id
                else:
                    if current_start is not None:
                        self._maybe_starvation_span(spans, stream_id, current_start, previous_round)
                    current_start = None
                previous_round = round_id
            if current_start is not None:
                self._maybe_starvation_span(spans, stream_id, current_start, previous_round)
        return sorted(spans, key=lambda item: (item["start_round"], item["stream_id"]))

    def _maybe_starvation_span(
        self,
        output: list[dict[str, Any]],
        stream_id: int,
        start_round: int,
        end_round: int | None,
    ) -> None:
        if end_round is None:
            return
        count = end_round - start_round + 1
        if count >= self.starvation_rounds:
            output.append(
                {
                    "stream_id": stream_id,
                    "start_round": start_round,
                    "end_round": end_round,
                    "round_count": count,
                }
            )

    def completion_report(self) -> dict[str, Any]:
        pending_local = sorted(
            sid
            for sid, stream in self.streams.items()
            if stream.state != "closed" and stream.bytes_sent < stream.body_bytes
        )
        non_closed = sorted(
            sid for sid, stream in self.streams.items() if stream.state != "closed"
        )
        complete = not pending_local and not non_closed and not self.queue
        script_exhausted = self.script_index >= len(self.script.events)
        truncated = bool(self.script.expect_complete and script_exhausted and not complete)
        return {
            "complete": complete,
            "script_exhausted": script_exhausted,
            "script_tail_truncated": truncated,
            "pending_local_stream_ids": pending_local,
            "non_closed_stream_ids": non_closed,
            "queued_stream_ids": list(self.queue),
            "goaway_received": self.goaway_received,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "declaration_version": self.declaration.declaration_version,
            "declaration_digest": self.declaration.content_digest,
            "script_digest": self.script.source_digest,
            "strategy": self.strategy,
            "initial_strategy": self.initial_strategy,
            "clock_mode": "virtual_monotonic_ms",
            "clock_ms": self.clock_ms,
            "script_index": self.script_index,
            "frame_count": self.frame_seq,
            "ledger_records": len(self.ledger.records),
            "streams": [
                {
                    **stream.to_dict(),
                    "weight": self.priority.weight_of(stream.stream_id),
                    "parent_id": self.priority.parent_of(stream.stream_id),
                }
                for _, stream in sorted(self.streams.items())
            ],
            "empty_waits": list(self.ledger.empty_waits),
            "queue_waits": list(self.ledger.queue_waits),
            "completion": self.completion_report(),
            "diagnostics": self.diagnostics(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "initial_strategy": self.initial_strategy,
            "clock_mode": "virtual_monotonic_ms",
            "starvation_rounds": self.starvation_rounds,
            "clock_ms": self.clock_ms,
            "frame_seq": self.frame_seq,
            "script_index": self.script_index,
            "schedule_round": self.schedule_round,
            "started": self.started,
            "finalized": self.finalized,
            "goaway_received": self.goaway_received,
            "goaway_last_stream_id": self.goaway_last_stream_id,
            "goaway_error_code": self.goaway_error_code,
            "peer_initial_stream_window": self.peer_initial_stream_window,
            "peer_max_frame_size": self.peer_max_frame_size,
            "peer_max_concurrent": self.peer_max_concurrent,
            "send_conn_window": self.send_conn_window,
            "recv_conn_window": self.recv_conn_window,
            "priority_update_seq": self._priority_update_seq,
            "queue": list(self.queue),
            "streams": [self.streams[sid].to_dict() for sid in sorted(self.streams)],
            "priority": self.priority.to_dict(),
            "ledger": self.ledger.to_dict(),
            "open_window_waits": [
                reason for reason in sorted(self._open_window_waits)
            ],
            "open_queue_wait": self._active_queue_wait is not None,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.strategy = str(state["strategy"])
        self.initial_strategy = str(state.get("initial_strategy", state["strategy"]))
        if self.strategy not in POLICIES:
            raise ValueError("snapshot strategy is not supported")
        if self.initial_strategy not in POLICIES:
            raise ValueError("snapshot initial strategy is not supported")
        self.starvation_rounds = int(state.get("starvation_rounds", 3))
        self.clock_ms = int(state["clock_ms"])
        self.frame_seq = int(state["frame_seq"])
        self.script_index = int(state["script_index"])
        self.schedule_round = int(state["schedule_round"])
        self.started = bool(state["started"])
        self.finalized = bool(state["finalized"])
        self.goaway_received = bool(state["goaway_received"])
        self.goaway_last_stream_id = state.get("goaway_last_stream_id")
        self.goaway_error_code = state.get("goaway_error_code")
        self.peer_initial_stream_window = int(state["peer_initial_stream_window"])
        self.peer_max_frame_size = int(state["peer_max_frame_size"])
        self.peer_max_concurrent = int(state["peer_max_concurrent"])
        self.send_conn_window = int(state["send_conn_window"])
        self.recv_conn_window = int(state["recv_conn_window"])
        self._priority_update_seq = int(state.get("priority_update_seq", 0))
        self.queue = [int(value) for value in state["queue"]]
        for stream_value in state["streams"]:
            stream_id = int(stream_value["stream_id"])
            stream = self.streams[stream_id]
            stream.state = str(stream_value["state"])
            stream.send_window = int(stream_value["send_window"])
            stream.recv_window = int(stream_value["recv_window"])
            stream.bytes_sent = int(stream_value["bytes_sent"])
            stream.bytes_received = int(stream_value["bytes_received"])
            stream.headers_sent = bool(stream_value["headers_sent"])
            stream.local_closed = bool(stream_value["local_closed"])
            stream.remote_closed = bool(stream_value["remote_closed"])
        self.priority = PriorityTree.from_dict(state["priority"])
        self.ledger = Ledger.from_dict(state["ledger"])
        self._open_window_waits = {}
        for wait in self.ledger.empty_waits:
            if wait.get("end_ms") is None:
                self._open_window_waits[str(wait["reason"])] = wait
        self._active_queue_wait = None
        for wait in self.ledger.queue_waits:
            if wait.get("end_ms") is None:
                self._active_queue_wait = wait
        self.finalized = False
        self.restored_from = {"script_index": self.script_index}
