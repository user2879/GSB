"""Real loopback parallel forwarding engine."""

from __future__ import annotations

import base64
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .budget import BudgetManager, ClockNs, SleepFn
from .codec import SequenceAllocator, encode_message
from .config import Route, TunnelConfig
from .errors import ErrorCode, TunnelError
from .logging import StructuredLogger
from .metrics import Ledger, SessionRecord
from .peer import LENGTH_PREFIX, read_json_line
from .session import SessionState

INBOUND_IDLE_TIMEOUT_SECONDS = 30.0
MAX_INBOUND_PAYLOAD = 64 * 1024 * 1024


def _clock_ms(clock_ns: ClockNs) -> int:
    return clock_ns() // 1_000_000


class TunnelEngine:
    def __init__(
        self,
        config: TunnelConfig,
        *,
        ledger: Ledger | None = None,
        log_path: str | Path | None = None,
        clock_ns: ClockNs = time.monotonic_ns,
        sleep_fn: SleepFn = time.sleep,
        sequence_watermarks: dict[str, int] | None = None,
        budget_usage: dict[str, Any] | None = None,
        restored_sessions: list[SessionState] | None = None,
    ) -> None:
        self.config = config
        self.routes = config.route_map()
        self.peers = config.peer_map()
        self.ledger = ledger if ledger is not None else Ledger()
        self.logger = StructuredLogger(log_path)
        self.clock_ns = clock_ns
        self.sequences = SequenceAllocator(sequence_watermarks)
        self.budget = BudgetManager(
            config.global_limits,
            config.routes,
            clock_ns=clock_ns,
            sleep_fn=sleep_fn,
            usage=budget_usage,
        )
        self._lock = threading.Lock()
        self._listeners: dict[str, socket.socket] = {}
        self._accept_threads: dict[str, threading.Thread] = {}
        self._session_threads: set[threading.Thread] = set()
        self._connections: set[socket.socket] = set()
        self._stopping = threading.Event()
        self._started = False
        self._sessions: dict[str, SessionState] = {}
        self._route_counters = {route.id: 0 for route in config.routes}
        if restored_sessions:
            for state in restored_sessions:
                self._sessions[state.session_id] = state
                suffix = state.session_id.rsplit("-", 1)[-1]
                if suffix.isdigit():
                    self._route_counters[state.route_id] = max(
                        self._route_counters.get(state.route_id, 0), int(suffix)
                    )

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            return
        bound: dict[str, socket.socket] = {}
        try:
            for route in self.config.routes:
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((route.listen.host, route.listen.port))
                listener.listen(64)
                listener.settimeout(0.2)
                bound[route.id] = listener
        except OSError as exc:
            for listener in bound.values():
                listener.close()
            raise TunnelError(
                ErrorCode.E_CONFIG,
                f"入站回环端口无法监听: {exc}",
                field="routes.listen.port",
                details={"port": getattr(exc, "port", None)},
            ) from exc
        self._listeners = bound
        self._started = True
        self._stopping.clear()
        for route in self.config.routes:
            thread = threading.Thread(
                target=self._accept_loop,
                args=(route,),
                name=f"accept-{route.id}",
                daemon=True,
            )
            self._accept_threads[route.id] = thread
            thread.start()
            self.logger.event(
                "engine.listening",
                route_id=route.id,
                host=route.listen.host,
                port=route.listen.port,
                target_port=route.target.port,
                mtu=route.mtu,
                bandwidth_kbps=route.bandwidth_kbps,
                max_sessions=route.max_sessions,
            )

    def stop(self, *, wait_seconds: float = 3.0) -> None:
        self._stopping.set()
        for listener in self._listeners.values():
            try:
                listener.close()
            except OSError:
                pass
        self._listeners.clear()
        with self._lock:
            threads = list(self._accept_threads.values())
            self._accept_threads.clear()
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        with self._lock:
            session_threads = list(self._session_threads)
        deadline = time.monotonic() + max(0.0, wait_seconds)
        for thread in session_threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if thread.is_alive():
                thread.join(timeout=remaining)
        with self._lock:
            connections = list(self._connections)
        for conn in connections:
            try:
                conn.close()
            except OSError:
                pass
        self._started = False
        self.logger.event("engine.stopped")

    def _accept_loop(self, route: Route) -> None:
        listener = self._listeners.get(route.id)
        if listener is None:
            return
        while not self._stopping.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self._lock:
                self._route_counters[route.id] += 1
                ordinal = self._route_counters[route.id]
                session_id = f"{route.id}-{ordinal:06d}"
                sequence = self.sequences.allocate(route.id)
                state = SessionState(
                    session_id=session_id,
                    route_id=route.id,
                    peer_id=route.peer_id,
                    sequence=sequence,
                    start_ms=_clock_ms(self.clock_ns),
                )
                self._sessions[session_id] = state
            self.logger.event(
                "session.accepted",
                route_id=route.id,
                session_id=session_id,
                sequence=sequence,
            )
            thread = threading.Thread(
                target=self._connection_worker,
                args=(conn, route, state),
                name=f"session-{session_id}",
                daemon=True,
            )
            with self._lock:
                self._session_threads.add(thread)
                self._connections.add(conn)
            thread.start()

    def _connection_worker(
        self, conn: socket.socket, route: Route, state: SessionState
    ) -> None:
        response: dict[str, Any] = {}
        try:
            self._admit_and_read(route, state, conn)
            response = self._result_payload(state)
        except TunnelError as exc:
            self._fail_if_needed(state, exc)
            response = self._result_payload(state)
            self.logger.error(
                "session.error",
                exc,
                route_id=route.id,
                session_id=state.session_id,
            )
        except Exception as exc:
            wrapped = TunnelError(
                ErrorCode.E_STATE,
                f"会话内部错误: {exc}",
                tunnel_id=route.id,
                session_id=state.session_id,
            )
            self._fail_if_needed(state, wrapped)
            response = self._result_payload(state)
            self.logger.error(
                "session.internal_error",
                wrapped,
                route_id=route.id,
                session_id=state.session_id,
            )
        finally:
            try:
                conn.sendall(
                    (
                        json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                )
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
            with self._lock:
                self._connections.discard(conn)
                self._session_threads.discard(threading.current_thread())

    def _admit_and_read(
        self, route: Route, state: SessionState, conn: socket.socket
    ) -> None:
        observer = self._queue_observer(state)
        self.budget.acquire_session(route.id, observer=observer)
        try:
            state.processing_started_ms = _clock_ms(self.clock_ns)
            state.status = "RUNNING"
            self.logger.event(
                "session.running",
                route_id=route.id,
                session_id=state.session_id,
                sequence=state.sequence,
            )
            conn.settimeout(INBOUND_IDLE_TIMEOUT_SECONDS)
            payload = self._read_inbound(conn)
            state.payload_bytes = len(payload)
            state.payload_b64 = base64.b64encode(payload).decode("ascii")
            self._execute_payload(route, state, payload)
        finally:
            state.finish_queue(_clock_ms(self.clock_ns))
            self.budget.release_session(route.id)

    def _read_inbound(self, conn: socket.socket) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = conn.recv(65536)
            except socket.timeout as exc:
                raise TunnelError(
                    ErrorCode.E_TIMEOUT,
                    "入站字节流空闲超时",
                    details={"timeout_seconds": INBOUND_IDLE_TIMEOUT_SECONDS},
                ) from exc
            except OSError as exc:
                raise TunnelError(ErrorCode.E_PEER, f"入站字节流读取失败: {exc}") from exc
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > MAX_INBOUND_PAYLOAD:
                raise TunnelError(
                    ErrorCode.E_ENCAP,
                    "单会话负载超过上限",
                    details={"max_bytes": MAX_INBOUND_PAYLOAD},
                )
            chunks.append(chunk)

    def forward_direct(
        self,
        route_id: str,
        payload: bytes,
        *,
        session_id: str | None = None,
        sequence: int | None = None,
        replay_of: str | None = None,
    ) -> SessionState:
        """Drive one session without an inbound socket (used by replay tools)."""

        route = self.routes.get(route_id)
        if route is None:
            raise TunnelError(ErrorCode.E_CONFIG, "未知隧道标识", field="route_id")
        with self._lock:
            if session_id is None:
                self._route_counters[route_id] += 1
                session_id = f"{route_id}-{self._route_counters[route_id]:06d}"
            if session_id in self._sessions:
                raise TunnelError(
                    ErrorCode.E_STATE,
                    "会话标识已存在",
                    tunnel_id=route_id,
                    session_id=session_id,
                )
            if sequence is None:
                sequence = self.sequences.allocate(route_id)
            else:
                self.sequences.observe(route_id, sequence)
            state = SessionState(
                session_id=session_id,
                route_id=route_id,
                peer_id=route.peer_id,
                sequence=sequence,
                start_ms=_clock_ms(self.clock_ns),
                replay_of=replay_of,
            )
            self._sessions[session_id] = state
        self.budget.acquire_session(route_id, observer=self._queue_observer(state))
        try:
            state.processing_started_ms = _clock_ms(self.clock_ns)
            state.status = "RUNNING"
            state.payload_bytes = len(payload)
            state.payload_b64 = base64.b64encode(payload).decode("ascii")
            self._execute_payload(route, state, payload)
        except TunnelError as exc:
            self._fail_if_needed(state, exc)
        finally:
            state.finish_queue(_clock_ms(self.clock_ns))
            self.budget.release_session(route_id)
        return state

    def _queue_observer(self, state: SessionState) -> Callable[[str, int], None]:
        def observe(event: str, now_ms: int) -> None:
            if event == "start":
                state.begin_queue(now_ms)
                if state.status == "QUEUED":
                    self.logger.event(
                        "session.queued",
                        route_id=state.route_id,
                        session_id=state.session_id,
                        start_ms=now_ms,
                    )
            elif event == "end":
                start = state._queue_start_ms
                state.end_queue(now_ms)
                self.logger.event(
                    "session.dequeued",
                    route_id=state.route_id,
                    session_id=state.session_id,
                    start_ms=start,
                    end_ms=now_ms,
                    duration_ms=(now_ms - start) if start is not None else 0,
                )

        return observe

    def _execute_payload(
        self, route: Route, state: SessionState, payload: bytes
    ) -> None:
        encoded = encode_message(
            tunnel_id=route.id,
            sequence=state.sequence,
            cipher=route.cipher,
            mtu=route.mtu,
            payload=payload,
        )
        state.encapsulation_digest = encoded.payload_digest
        state.fragment_count = len(encoded.frames)
        self.logger.event(
            "session.forwarding",
            route_id=route.id,
            session_id=state.session_id,
            sequence=state.sequence,
            payload_digest=encoded.payload_digest,
            fragmented=encoded.fragmented,
            fragment_count=len(encoded.frames),
        )
        wire_sent = 0
        peer_reply: dict[str, Any]
        try:
            try:
                peer = socket.create_connection(
                    (route.target.host, route.target.port),
                    timeout=self.config.global_limits.connect_timeout_ms / 1000,
                )
            except (socket.timeout, TimeoutError) as exc:
                raise TunnelError(
                    ErrorCode.E_PEER,
                    "本地模拟对端端口不可达",
                    tunnel_id=route.id,
                    session_id=state.session_id,
                    details={
                        "host": route.target.host,
                        "port": route.target.port,
                        "connect_timeout_ms": self.config.global_limits.connect_timeout_ms,
                    },
                ) from exc
            except ConnectionRefusedError as exc:
                raise TunnelError(
                    ErrorCode.E_PEER,
                    "本地模拟对端端口不可达",
                    tunnel_id=route.id,
                    session_id=state.session_id,
                    details={"host": route.target.host, "port": route.target.port},
                ) from exc
            with peer:
                peer.settimeout(self.config.global_limits.peer_timeout_ms / 1000)
                for frame in encoded.frames:
                    wire = LENGTH_PREFIX.pack(len(frame)) + frame
                    sent = self._send_budgeted(route, state, peer, wire)
                    wire_sent += sent
                state.forwarded_bytes = len(payload)
                try:
                    peer.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                peer_reply = read_json_line(peer)
        except socket.timeout as exc:
            raise TunnelError(
                ErrorCode.E_TIMEOUT,
                "等待本地模拟对端响应超时",
                tunnel_id=route.id,
                session_id=state.session_id,
                details={"peer_timeout_ms": self.config.global_limits.peer_timeout_ms},
            ) from exc
        except OSError as exc:
            raise TunnelError(
                ErrorCode.E_PEER,
                f"本地模拟对端链路失败: {exc}",
                tunnel_id=route.id,
                session_id=state.session_id,
            ) from exc
        finally:
            state.wire_bytes = wire_sent

        reply_code = str(peer_reply.get("code", ErrorCode.E_PEER.value))
        state.peer_payload_digest = peer_reply.get("payload_digest")
        if reply_code != ErrorCode.OK.value:
            mapped = (
                reply_code
                if reply_code
                in {
                    ErrorCode.E_ENCAP.value,
                    ErrorCode.E_FRAG.value,
                    ErrorCode.E_PEER.value,
                    ErrorCode.E_TIMEOUT.value,
                    ErrorCode.E_DIGEST.value,
                }
                else ErrorCode.E_PEER.value
            )
            raise TunnelError(
                mapped,
                str(peer_reply.get("message", "对端返回失败")),
                tunnel_id=route.id,
                session_id=state.session_id,
                offset=peer_reply.get("error_offset"),
                details={"peer_reply": peer_reply},
            )
        if state.peer_payload_digest != encoded.payload_digest:
            raise TunnelError(
                ErrorCode.E_DIGEST,
                "对端回送摘要与本地封装摘要不一致",
                tunnel_id=route.id,
                session_id=state.session_id,
                details={
                    "expected": encoded.payload_digest,
                    "actual": state.peer_payload_digest,
                },
            )
        peer_bytes = peer_reply.get("payload_bytes")
        if isinstance(peer_bytes, int) and peer_bytes != len(payload):
            raise TunnelError(
                ErrorCode.E_PEER,
                "对端接收字节数与发送负载不一致",
                tunnel_id=route.id,
                session_id=state.session_id,
                details={"expected": len(payload), "actual": peer_bytes},
            )
        self._complete(state, ErrorCode.OK.value)

    def _send_budgeted(
        self,
        route: Route,
        state: SessionState,
        peer: socket.socket,
        wire: bytes,
    ) -> int:
        chunk_size = self.budget.max_bandwidth_chunk(route.id)
        observer = self._queue_observer(state)
        offset = 0
        while offset < len(wire):
            chunk = wire[offset : offset + chunk_size]
            self.budget.acquire_bandwidth(route.id, len(chunk), observer=observer)
            try:
                peer.sendall(chunk)
            except socket.timeout as exc:
                raise TunnelError(
                    ErrorCode.E_TIMEOUT,
                    "向对端写入超时",
                    tunnel_id=route.id,
                    session_id=state.session_id,
                ) from exc
            except OSError as exc:
                raise TunnelError(
                    ErrorCode.E_PEER,
                    f"向对端写入失败: {exc}",
                    tunnel_id=route.id,
                    session_id=state.session_id,
                ) from exc
            offset += len(chunk)
        return offset

    def _complete(self, state: SessionState, code: str) -> None:
        if state.status in {"COMPLETED", "FAILED"}:
            return
        state.business_code = code
        state.status = "COMPLETED" if code == ErrorCode.OK.value else "FAILED"
        state.end_ms = _clock_ms(self.clock_ns)
        state.finish_queue(_clock_ms(self.clock_ns))
        record = SessionRecord(
            session_id=state.session_id,
            route_id=state.route_id,
            peer_id=state.peer_id,
            sequence=state.sequence,
            start_ms=state.start_ms,
            end_ms=max(state.start_ms, state.end_ms),
            payload_bytes=state.payload_bytes,
            forwarded_bytes=state.forwarded_bytes,
            wire_bytes=state.wire_bytes,
            queue_events=tuple(state.queue_events),
            status=state.status,
            business_code=state.business_code,
            encapsulation_digest=state.encapsulation_digest,
            fragment_count=state.fragment_count,
            peer_payload_digest=state.peer_payload_digest,
            replay_of=state.replay_of,
            error_detail=state.error,
        )
        self.ledger.record(record)
        self.logger.event(
            "session.completed",
            code=code,
            route_id=state.route_id,
            session_id=state.session_id,
            sequence=state.sequence,
            payload_bytes=state.payload_bytes,
            forwarded_bytes=state.forwarded_bytes,
            wire_bytes=state.wire_bytes,
            queue_count=state.queue_count,
            total_queue_ms=state.total_queue_ms,
            payload_digest=state.encapsulation_digest,
        )

    def _fail_if_needed(self, state: SessionState, exc: TunnelError) -> None:
        if state.status in {"COMPLETED", "FAILED"}:
            return
        state.error = exc.to_dict()
        self._complete(state, exc.code)
        try:
            self.logger.event(
                "session.failed",
                code=exc.code,
                route_id=state.route_id,
                session_id=state.session_id,
                sequence=state.sequence,
                error=exc.to_dict(),
            )
        except Exception:
            pass

    def _result_payload(self, state: SessionState) -> dict[str, Any]:
        return {
            "session_id": state.session_id,
            "route_id": state.route_id,
            "sequence": state.sequence,
            "status": state.status,
            "code": state.business_code,
            "payload_bytes": state.payload_bytes,
            "wire_bytes": state.wire_bytes,
            "queue_count": state.queue_count,
            "total_queue_ms": state.total_queue_ms,
            "payload_digest": state.encapsulation_digest,
            "peer_payload_digest": state.peer_payload_digest,
            "fragment_count": state.fragment_count,
            "error": state.error,
        }

    def sessions_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            states = sorted(
                self._sessions.values(), key=lambda item: (item.start_ms, item.session_id)
            )
            return [state.to_dict() for state in states]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "routes": [route.to_dict() for route in self.config.routes],
            "budget": self.budget.snapshot().to_dict(),
            "sessions": self.sessions_snapshot(),
            "ledger": {
                "records": self.ledger.to_dicts(),
                "statistics": self.ledger.statistics(),
                "digest": self.ledger.digest(),
            },
        }


def send_inbound_payload(
    route: Route,
    payload: bytes,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Client helper: send bytes then wait for the engine's JSON outcome."""

    with socket.create_connection((route.listen.host, route.listen.port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        return read_json_line(sock)
