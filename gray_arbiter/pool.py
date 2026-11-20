"""Real loopback connection pools, lifecycle state, and borrow arbitration."""

from __future__ import annotations

import hashlib
import http.client
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .errors import BusinessError, StructuredLogger
from .model import (
    BorrowRecord,
    Config,
    ConnectionEvent,
    PoolBatchStats,
    PoolSpec,
    PoolTotals,
    WatermarkView,
)


def mono_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class ConnectionRuntime:
    connection_id: str
    pool_id: str
    spec: PoolSpec
    http_conn: http.client.HTTPConnection
    lender_pool: str | None = None
    borrow_id: str | None = None
    state: str = "new"
    in_use: bool = False
    total_requests: int = 0
    last_transition_ms: int = field(default_factory=mono_ms)
    broken: bool = False

    @property
    def owned(self) -> bool:
        return self.lender_pool is None


@dataclass
class AcquiredConnection:
    connection: ConnectionRuntime
    reuse_sequence: int
    queue_started_mono_ms: int
    queue_ended_mono_ms: int
    wait_ms: int
    borrow_id: str | None
    lender_pool: str | None
    borrow_deadline_mono_ms: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection.connection_id,
            "pool_id": self.connection.pool_id,
            "reuse_sequence": self.reuse_sequence,
            "queue_started_mono_ms": self.queue_started_mono_ms,
            "queue_ended_mono_ms": self.queue_ended_mono_ms,
            "wait_ms": self.wait_ms,
            "borrow_id": self.borrow_id,
            "lender_pool": self.lender_pool,
            "borrow_deadline_mono_ms": self.borrow_deadline_mono_ms,
        }


@dataclass
class _PoolState:
    spec: PoolSpec
    connections: dict[str, ConnectionRuntime] = field(default_factory=dict)
    idle: list[str] = field(default_factory=list)
    lent_borrow_ids: set[str] = field(default_factory=set)
    totals: PoolTotals = field(default_factory=PoolTotals)
    watermark_samples: list[int] = field(default_factory=list)
    detail_samples: list[dict[str, int]] = field(default_factory=list)
    sample_start: int = 0
    detail_sample_start: int = 0

    @property
    def active(self) -> int:
        return len(self.connections)

    @property
    def idle_count(self) -> int:
        return sum(1 for conn_id in self.idle if conn_id in self.connections)

    @property
    def in_use(self) -> int:
        return sum(1 for conn in self.connections.values() if conn.in_use)

    @property
    def borrowed_in(self) -> int:
        return sum(1 for conn in self.connections.values() if not conn.owned)

    @property
    def owned_active(self) -> int:
        return sum(1 for conn in self.connections.values() if conn.owned)

    @property
    def lent_out(self) -> int:
        return len(self.lent_borrow_ids)

    @property
    def watermark(self) -> int:
        return self.active + self.lent_out


class PoolManager:
    """Thread-safe pool manager.

    ``active_connections`` counts all established loopback connections assigned
    to a pool.  A borrowed connection is assigned to the borrower (it connects
    to the borrower's endpoint) and consumes one unit of the lender's free
    capacity.  Consequently the lender invariant is always::

        lender.owned_active + lender.lent_out <= lender.max_connections

    A borrower may temporarily exceed its own ``max_connections`` by the number
    of borrowed slots, which is the purpose of borrowing.  Mirror pools are
    excluded from both sides of this mechanism.
    """

    def __init__(
        self,
        config: Config,
        logger: StructuredLogger | None = None,
        *,
        endpoint_overrides: Mapping[str, tuple[str, int]] | None = None,
        start_reaper: bool = True,
    ) -> None:
        self.config = config
        self.logger = logger or StructuredLogger()
        self.endpoint_overrides = dict(endpoint_overrides or {})
        self.condition = threading.Condition(threading.RLock())
        self.states: dict[str, _PoolState] = {
            pool_id: _PoolState(spec=spec) for pool_id, spec in sorted(config.pools.items())
        }
        self.mirror_pools = {rule.mirror_pool for rule in config.rules.values()}
        self.connection_counter = 0
        self.borrow_counter = 0
        self.event_counter = 0
        self.events: list[ConnectionEvent] = []
        self.event_start = 0
        self.borrows: dict[str, BorrowRecord] = {}
        self.resource_change_callback: Callable[[dict[str, Any]], None] | None = None
        self.batch_id: str | None = None
        self._batch_finalized = False
        self._stop_event = threading.Event()
        self._reaper_thread: threading.Thread | None = None
        with self.condition:
            for state in self.states.values():
                self._sample_locked(state)
        if start_reaper:
            self._start_reaper()

    # ------------------------------------------------------------------
    # Lifecycle and snapshots
    # ------------------------------------------------------------------
    def _start_reaper(self) -> None:
        timeouts = [state.spec.idle_timeout_ms for state in self.states.values()]
        interval_ms = max(20, min(250, min(timeouts) if timeouts else 250))

        def loop() -> None:
            while not self._stop_event.wait(interval_ms / 1000.0):
                try:
                    self.reap_idle()
                except Exception:
                    # Reaping is a safety net and must never kill the process.
                    pass

        self._reaper_thread = threading.Thread(
            target=loop,
            name="loopback-pool-reaper",
            daemon=True,
        )
        self._reaper_thread.start()

    def begin_batch(self, batch_id: str) -> None:
        with self.condition:
            self.batch_id = batch_id
            self._batch_finalized = False
            self.event_start = len(self.events)
            for state in self.states.values():
                state.totals = PoolTotals()
                state.sample_start = len(state.watermark_samples)
                state.detail_sample_start = len(state.detail_samples)
                self._sample_locked(state)

    def finalize_batch(self) -> tuple[list[PoolBatchStats], list[ConnectionEvent]]:
        with self.condition:
            if self._batch_finalized:
                events = self.events[self.event_start :]
                stats = self._stats_from_state(finalize=False)
                return stats, list(events)
            for state in self.states.values():
                self._sample_locked(state)
            self._batch_finalized = True
            stats = self._stats_from_state(finalize=True)
            events = list(self.events[self.event_start :])
            return stats, events

    def _stats_from_state(self, *, finalize: bool) -> list[PoolBatchStats]:
        result: list[PoolBatchStats] = []
        for pool_id in sorted(self.states):
            state = self.states[pool_id]
            samples = state.watermark_samples[state.sample_start :]
            if not samples:
                samples = [state.watermark]
            total = sum(samples)
            average_scaled = (total * 1000) // len(samples)
            result.append(
                PoolBatchStats(
                    pool_id=pool_id,
                    max_connections=state.spec.max_connections,
                    active_connections=state.active,
                    in_use_connections=state.in_use,
                    idle_connections=state.idle_count,
                    lent_out=state.lent_out,
                    borrowed_in=state.borrowed_in,
                    reuse_count=state.totals.reuse_count,
                    borrow_count=state.totals.borrow_count,
                    closed_count=state.totals.closed_count,
                    sample_count=len(samples),
                    average_watermark_scaled=average_scaled,
                    peak_watermark=max(samples),
                )
            )
        return result

    def batch_borrows(self) -> list[BorrowRecord]:
        if self.batch_id is None:
            return []
        return sorted(
            (item for item in self.borrows.values() if item.batch_id == self.batch_id),
            key=lambda item: item.borrow_id,
        )

    def batch_watermark_series(self) -> dict[str, list[dict[str, int]]]:
        with self.condition:
            return {
                pool_id: list(state.detail_samples[state.detail_sample_start :])
                for pool_id, state in sorted(self.states.items())
            }

    def capture_watermarks(self) -> dict[str, dict[str, Any]]:
        with self.condition:
            return {
                pool_id: self._view_locked(state).to_dict()
                for pool_id, state in sorted(self.states.items())
            }

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            pools = []
            for pool_id, state in sorted(self.states.items()):
                view = self._view_locked(state)
                pools.append(
                    {
                        **view.to_dict(),
                        "max_connections": state.spec.max_connections,
                        "max_requests_per_connection": state.spec.max_requests_per_connection,
                        "idle_timeout_ms": state.spec.idle_timeout_ms,
                        "lendable_to": list(state.spec.lendable_to),
                        "endpoint": self._endpoint_locked(pool_id),
                        "reuse_count": state.totals.reuse_count,
                        "borrow_count": state.totals.borrow_count,
                        "closed_count": state.totals.closed_count,
                    }
                )
            return {
                "config_version": self.config.version,
                "pools": pools,
                "connections": sorted(
                    (
                        {
                            "connection_id": conn.connection_id,
                            "pool_id": conn.pool_id,
                            "state": conn.state,
                            "in_use": conn.in_use,
                            "borrow_id": conn.borrow_id,
                            "lender_pool": conn.lender_pool,
                            "total_requests": conn.total_requests,
                        }
                        for conn in (
                            item
                            for state in self.states.values()
                            for item in state.connections.values()
                        )
                    ),
                    key=lambda item: item["connection_id"],
                ),
                "active_borrows": sorted(
                    (item.to_dict() for item in self.borrows.values() if item.state == "active"),
                    key=lambda item: item["borrow_id"],
                ),
                "borrows": sorted(
                    (item.to_dict() for item in self.borrows.values()),
                    key=lambda item: item["borrow_id"],
                ),
                "connection_events": [event.to_dict() for event in self.events],
            }

    def _view_locked(self, state: _PoolState) -> WatermarkView:
        return WatermarkView(
            pool_id=state.spec.pool_id,
            active_connections=state.active,
            in_use_connections=state.in_use,
            idle_connections=state.idle_count,
            lent_out=state.lent_out,
            borrowed_in=state.borrowed_in,
            watermark=state.watermark,
        )

    def request_watermarks(self) -> dict[str, dict[str, Any]]:
        return self.capture_watermarks()

    # ------------------------------------------------------------------
    # Acquisition, borrowing, and queueing
    # ------------------------------------------------------------------
    def acquire(
        self,
        pool_id: str,
        request_id: str,
        batch_id: str,
        *,
        allow_borrow: bool,
        block: bool,
        timeout_ms: int | None = None,
        http_timeout_ms: int = 2000,
    ) -> AcquiredConnection:
        if pool_id not in self.states:
            raise BusinessError(
                "E_POOL",
                "unknown pool for acquisition",
                field="pool_id",
                request_id=request_id,
                pool_id=pool_id,
                phase="acquire",
            )
        started = mono_ms()
        queue_deadline = started + (timeout_ms if timeout_ms is not None else self.config.queue_timeout_ms)
        with self.condition:
            while True:
                now = mono_ms()
                self._close_expired_idle_locked(now)
                state = self.states[pool_id]

                # Idle connections are always preferred over creating or
                # borrowing.  They remain counted in active_connections.
                idle_conn = self._pop_usable_idle_locked(state, now)
                if idle_conn is not None:
                    return self._activate_existing_locked(
                        idle_conn,
                        request_id=request_id,
                        queue_started=started,
                        queue_ended=now,
                    )

                if state.active < state.spec.max_connections:
                    conn = self._create_owned_connection_locked(
                        state,
                        request_id=request_id,
                        http_timeout_ms=http_timeout_ms,
                        queue_started=started,
                    )
                    return self._activate_new_locked(
                        conn,
                        request_id=request_id,
                        queue_started=started,
                        queue_ended=mono_ms(),
                        borrow_id=None,
                        borrow_deadline_mono_ms=None,
                    )

                if allow_borrow and pool_id not in self.mirror_pools:
                    lender = self._choose_lender_locked(pool_id)
                    if lender is not None:
                        conn, borrow = self._create_borrowed_connection_locked(
                            borrower=self.states[pool_id],
                            lender=lender,
                            request_id=request_id,
                            batch_id=batch_id,
                            http_timeout_ms=http_timeout_ms,
                        )
                        return self._activate_new_locked(
                            conn,
                            request_id=request_id,
                            queue_started=started,
                            queue_ended=mono_ms(),
                            borrow_id=borrow.borrow_id,
                            borrow_deadline_mono_ms=borrow.deadline_mono_ms,
                        )

                if not block:
                    code = "E_MIRROR" if pool_id in self.mirror_pools else "E_QUEUE"
                    raise BusinessError(
                        code,
                        "pool exhausted and non-blocking acquisition requested",
                        field="pool.active_connections",
                        request_id=request_id,
                        pool_id=pool_id,
                        phase="acquire",
                        details={
                            "queue_started_mono_ms": started,
                            "queue_ended_mono_ms": mono_ms(),
                            "wait_ms": max(0, mono_ms() - started),
                            "available_borrow": False,
                        },
                    )

                now = mono_ms()
                if now >= queue_deadline:
                    raise BusinessError(
                        "E_QUEUE",
                        "timed out waiting for a connection or borrow slot",
                        field="pool.active_connections",
                        request_id=request_id,
                        pool_id=pool_id,
                        phase="acquire",
                        details={
                            "queue_started_mono_ms": started,
                            "queue_ended_mono_ms": now,
                            "wait_ms": max(0, now - started),
                        },
                    )
                self.condition.wait(timeout=min(0.05, max(0.001, (queue_deadline - now) / 1000.0)))

    def _activate_existing_locked(
        self,
        conn: ConnectionRuntime,
        *,
        request_id: str,
        queue_started: int,
        queue_ended: int,
    ) -> AcquiredConnection:
        reuse_sequence = conn.total_requests
        conn.total_requests += 1
        conn.in_use = True
        conn.state = "reused"
        conn.last_transition_ms = queue_ended
        self.states[conn.pool_id].totals.reuse_count += 1
        self._record_event_locked(conn, "reused", "request_acquisition")
        self._sample_pool_locked(self.states[conn.pool_id])
        return AcquiredConnection(
            connection=conn,
            reuse_sequence=reuse_sequence,
            queue_started_mono_ms=queue_started,
            queue_ended_mono_ms=queue_ended,
            wait_ms=max(0, queue_ended - queue_started),
            borrow_id=None,
            lender_pool=None,
            borrow_deadline_mono_ms=None,
        )

    def _activate_new_locked(
        self,
        conn: ConnectionRuntime,
        *,
        request_id: str,
        queue_started: int,
        queue_ended: int,
        borrow_id: str | None,
        borrow_deadline_mono_ms: int | None = None,
    ) -> AcquiredConnection:
        reuse_sequence = conn.total_requests
        conn.total_requests += 1
        return AcquiredConnection(
            connection=conn,
            reuse_sequence=reuse_sequence,
            queue_started_mono_ms=queue_started,
            queue_ended_mono_ms=queue_ended,
            wait_ms=max(0, queue_ended - queue_started),
            borrow_id=borrow_id,
            lender_pool=conn.lender_pool,
            borrow_deadline_mono_ms=borrow_deadline_mono_ms,
        )

    def _endpoint_locked(self, pool_id: str) -> dict[str, Any]:
        if pool_id in self.endpoint_overrides:
            host, port = self.endpoint_overrides[pool_id]
        else:
            host, port = self.states[pool_id].spec.host, self.states[pool_id].spec.port
        return {"host": host, "port": port}

    def _new_http_connection(self, pool_id: str, timeout_ms: int) -> http.client.HTTPConnection:
        endpoint = self._endpoint_locked(pool_id)
        return http.client.HTTPConnection(
            endpoint["host"],
            endpoint["port"],
            timeout=max(0.001, timeout_ms / 1000.0),
        )

    def _create_owned_connection_locked(
        self,
        state: _PoolState,
        *,
        request_id: str,
        http_timeout_ms: int,
        queue_started: int,
    ) -> ConnectionRuntime:
        self.connection_counter += 1
        conn_id = f"{state.spec.pool_id}-v{self.config.version}-c{self.connection_counter:06d}"
        http_conn = self._new_http_connection(state.spec.pool_id, http_timeout_ms)
        conn = ConnectionRuntime(
            connection_id=conn_id,
            pool_id=state.spec.pool_id,
            spec=state.spec,
            http_conn=http_conn,
            state="established",
            in_use=True,
        )
        state.connections[conn_id] = conn
        self._sample_pool_locked(state)
        try:
            http_conn.connect()
        except (OSError, socket.timeout, http.client.HTTPException) as exc:
            state.connections.pop(conn_id, None)
            conn.state = "closed"
            conn.in_use = False
            self._record_event_locked(conn, "closed", "connect_failed")
            self._sample_pool_locked(state)
            self.condition.notify_all()
            raise BusinessError(
                "E_TARGET",
                "loopback target is unreachable during connection establishment",
                field="target",
                request_id=request_id,
                pool_id=state.spec.pool_id,
                phase="establish",
                details={
                    "exception": type(exc).__name__,
                    "endpoint": self._endpoint_locked(state.spec.pool_id),
                    "queue_started_mono_ms": queue_started,
                    "queue_ended_mono_ms": mono_ms(),
                    "wait_ms": max(0, mono_ms() - queue_started),
                },
            )
        state.totals.created_count += 1
        self._record_event_locked(conn, "established", "connection_created")
        self._sample_pool_locked(state)
        return conn

    def _create_borrowed_connection_locked(
        self,
        *,
        borrower: _PoolState,
        lender: _PoolState,
        request_id: str,
        batch_id: str,
        http_timeout_ms: int,
    ) -> tuple[ConnectionRuntime, BorrowRecord]:
        self.connection_counter += 1
        self.borrow_counter += 1
        now = mono_ms()
        borrow_id = f"b-v{self.config.version}-{self.borrow_counter:06d}"
        borrow = BorrowRecord(
            borrow_id=borrow_id,
            version=self.config.version,
            lender_pool=lender.spec.pool_id,
            borrower_pool=borrower.spec.pool_id,
            amount=1,
            request_id=request_id,
            batch_id=batch_id,
            created_mono_ms=now,
            deadline_mono_ms=now + self.config.borrow_ttl_ms,
        )
        self.borrows[borrow_id] = borrow
        lender.lent_borrow_ids.add(borrow_id)
        lender.totals.borrow_count += 1
        borrower.totals.borrow_count += 1
        self._sample_pool_locked(lender)
        conn_id = (
            f"{borrower.spec.pool_id}-v{self.config.version}-c{self.connection_counter:06d}"
        )
        http_conn = self._new_http_connection(borrower.spec.pool_id, http_timeout_ms)
        conn = ConnectionRuntime(
            connection_id=conn_id,
            pool_id=borrower.spec.pool_id,
            spec=borrower.spec,
            http_conn=http_conn,
            lender_pool=lender.spec.pool_id,
            borrow_id=borrow_id,
            state="established",
            in_use=True,
        )
        borrower.connections[conn_id] = conn
        self._sample_pool_locked(borrower)
        try:
            http_conn.connect()
        except (OSError, socket.timeout, http.client.HTTPException) as exc:
            borrower.connections.pop(conn_id, None)
            lender.lent_borrow_ids.discard(borrow_id)
            borrow.state = "returned_failed"
            borrow.returned_mono_ms = mono_ms()
            conn.state = "closed"
            conn.in_use = False
            self._record_event_locked(conn, "closed", "borrow_connect_failed")
            self._sample_pool_locked(borrower)
            self._sample_pool_locked(lender)
            self.condition.notify_all()
            raise BusinessError(
                "E_TARGET",
                "borrowed loopback target is unreachable during establishment",
                field="target",
                request_id=request_id,
                pool_id=borrower.spec.pool_id,
                phase="establish",
                details={
                    "exception": type(exc).__name__,
                    "borrow_id": borrow_id,
                    "lender_pool": lender.spec.pool_id,
                    "endpoint": self._endpoint_locked(borrower.spec.pool_id),
                },
            )
        self._record_event_locked(conn, "established", "borrowed_connection_created")
        self._sample_pool_locked(borrower)
        self._sample_pool_locked(lender)
        return conn, borrow

    def _choose_lender_locked(self, borrower_pool: str) -> _PoolState | None:
        candidates: list[_PoolState] = []
        for lender_id, state in sorted(self.states.items()):
            if lender_id == borrower_pool:
                continue
            if lender_id in self.mirror_pools:
                continue
            if borrower_pool not in state.spec.lendable_to:
                continue
            if state.owned_active + state.lent_out >= state.spec.max_connections:
                continue
            candidates.append(state)
        if not candidates:
            return None
        # Stable explicit policy: largest free capacity, then lexicographic ID.
        return sorted(
            candidates,
            key=lambda state: (
                -(state.spec.max_connections - state.owned_active - state.lent_out),
                state.spec.pool_id,
            ),
        )[0]

    def _pop_usable_idle_locked(
        self, state: _PoolState, now: int
    ) -> ConnectionRuntime | None:
        while state.idle:
            conn_id = state.idle.pop(0)
            conn = state.connections.get(conn_id)
            if conn is None or conn.state != "idle" or conn.in_use:
                continue
            if conn.http_conn.sock is None:
                self._close_connection_locked(conn, "socket_closed")
                continue
            if now - conn.last_transition_ms >= state.spec.idle_timeout_ms:
                self._close_connection_locked(conn, "idle_timeout")
                continue
            if conn.total_requests >= state.spec.max_requests_per_connection:
                self._close_connection_locked(conn, "reuse_limit")
                continue
            return conn
        return None

    def _close_expired_idle_locked(self, now: int) -> None:
        for state in self.states.values():
            for conn_id in list(state.idle):
                conn = state.connections.get(conn_id)
                if conn is None or conn.in_use:
                    continue
                if now - conn.last_transition_ms >= state.spec.idle_timeout_ms:
                    self._close_connection_locked(conn, "idle_timeout")

    def reap_idle(self) -> None:
        with self.condition:
            self._close_expired_idle_locked(mono_ms())

    def _close_connection_locked(self, conn: ConnectionRuntime, reason: str) -> None:
        state = self.states.get(conn.pool_id)
        if state is None or conn.connection_id not in state.connections:
            return
        if conn.connection_id in state.idle:
            state.idle.remove(conn.connection_id)
        try:
            conn.http_conn.close()
        except Exception:
            pass
        state.connections.pop(conn.connection_id, None)
        conn.state = "closed"
        conn.in_use = False
        state.totals.closed_count += 1
        self._record_event_locked(conn, "closed", reason)
        if conn.lender_pool is not None and conn.borrow_id is not None:
            lender = self.states.get(conn.lender_pool)
            if lender is not None:
                lender.lent_borrow_ids.discard(conn.borrow_id)
                self._sample_pool_locked(lender)
            borrow = self.borrows.get(conn.borrow_id)
            if borrow is not None and borrow.state == "active":
                borrow.returned_mono_ms = mono_ms()
                borrow.state = "returned"
        self._sample_pool_locked(state)
        self.condition.notify_all()

    # ------------------------------------------------------------------
    # HTTP execution and release
    # ------------------------------------------------------------------
    def execute_http(
        self,
        acquired: AcquiredConnection,
        path: str,
        request_id: str,
        *,
        timeout_ms: int,
        phase: str,
    ) -> dict[str, Any]:
        conn = acquired.connection
        if conn.broken:
            raise BusinessError(
                "E_TARGET",
                "connection was marked broken",
                field="connection.state",
                request_id=request_id,
                pool_id=conn.pool_id,
                phase=phase,
            )
        timeout_seconds = max(0.001, timeout_ms / 1000.0)
        conn.http_conn.timeout = timeout_seconds
        if conn.http_conn.sock is not None:
            conn.http_conn.sock.settimeout(timeout_seconds)
        headers = {
            "Host": f"127.0.0.1:{conn.spec.port}",
            "Connection": "keep-alive",
            "X-Request-Id": request_id,
            "X-Pool-Id": conn.pool_id,
        }
        started = time.perf_counter_ns()
        try:
            conn.http_conn.request("GET", path, headers=headers)
            response = conn.http_conn.getresponse()
            body = response.read()
            elapsed_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)
            connection_alive = conn.http_conn.sock is not None
            if not connection_alive:
                conn.broken = True
            summary = {
                "sha256_16": hashlib.sha256(body).hexdigest()[:16],
                "bytes": len(body),
                "content_type": response.getheader("Content-Type", ""),
            }
            return {
                "status": "ok" if 200 <= response.status < 500 else "http_error",
                "http_status": response.status,
                "response_summary": summary,
                "elapsed_ms": elapsed_ms,
                "peer": "127.0.0.1",
                "connection_alive": connection_alive,
            }
        except socket.timeout as exc:
            conn.broken = True
            elapsed_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)
            raise BusinessError(
                "E_TIMEOUT",
                "loopback request timed out on a real socket",
                field="socket.timeout",
                request_id=request_id,
                pool_id=conn.pool_id,
                phase=phase,
                details={
                    "timeout_ms": timeout_ms,
                    "elapsed_ms": elapsed_ms,
                    "exception": type(exc).__name__,
                },
            )
        except (OSError, http.client.HTTPException) as exc:
            conn.broken = True
            elapsed_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)
            raise BusinessError(
                "E_TARGET",
                "loopback request failed",
                field="target",
                request_id=request_id,
                pool_id=conn.pool_id,
                phase=phase,
                details={
                    "elapsed_ms": elapsed_ms,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )

    def release(
        self,
        acquired: AcquiredConnection,
        request_id: str,
        *,
        error: BusinessError | None = None,
    ) -> None:
        conn = acquired.connection
        with self.condition:
            state = self.states.get(conn.pool_id)
            if state is None or conn.connection_id not in state.connections:
                # A version adjudication may already have returned/invalidated
                # this connection.  The request result remains observable in the
                # ledger, but accounting must not be decremented twice.
                return
            conn.in_use = False
            if conn.lender_pool is not None:
                borrow = self.borrows.get(conn.borrow_id or "")
                if borrow is not None and borrow.state == "active":
                    borrow.returned_mono_ms = mono_ms()
                    borrow.state = "returned" if error is None else "returned_with_error"
                self._close_connection_locked(
                    conn,
                    "borrow_return_error" if error is not None else "borrow_return",
                )
                return
            if error is not None or conn.broken:
                self._close_connection_locked(conn, "request_error")
                return
            if conn.total_requests >= state.spec.max_requests_per_connection:
                self._close_connection_locked(conn, "reuse_limit")
                return
            if state.spec.idle_timeout_ms <= 0:
                self._close_connection_locked(conn, "idle_timeout_zero")
                return
            conn.state = "idle"
            conn.last_transition_ms = mono_ms()
            if conn.connection_id not in state.idle:
                state.idle.append(conn.connection_id)
            self._record_event_locked(conn, "idle", "request_released")
            self._sample_pool_locked(state)
            self.condition.notify_all()

    # ------------------------------------------------------------------
    # Version replacement and shutdown
    # ------------------------------------------------------------------
    def adjudicate_borrows(self, policy: str, new_config: Config) -> list[dict[str, Any]]:
        if policy not in {"force_return", "recompute", "invalidate"}:
            raise ValueError("borrow policy must be force_return, recompute, or invalidate")
        dispositions: list[dict[str, Any]] = []
        with self.condition:
            for borrow in sorted(self.borrows.values(), key=lambda item: item.borrow_id):
                if borrow.state != "active":
                    continue
                conn = self._connection_for_borrow_locked(borrow.borrow_id)
                action = policy
                reason = "explicit_version_policy"
                capacity_recheck: dict[str, Any] | None = None
                old_deadline = borrow.deadline_mono_ms
                if policy == "recompute":
                    borrower_spec = new_config.pools.get(borrow.borrower_pool)
                    lender_spec = new_config.pools.get(borrow.lender_pool)
                    borrower_state = self.states.get(borrow.borrower_pool)
                    lender_state = self.states.get(borrow.lender_pool)
                    permitted = (
                        borrower_spec is not None
                        and lender_spec is not None
                        and borrower_state is not None
                        and lender_state is not None
                        and borrower_state.active <= borrower_spec.max_connections
                        and lender_state.owned_active + lender_state.lent_out
                        <= lender_spec.max_connections
                    )
                    capacity_recheck = {
                        "borrower_active": None if borrower_state is None else borrower_state.active,
                        "borrower_new_max": None if borrower_spec is None else borrower_spec.max_connections,
                        "lender_committed": None
                        if lender_state is None
                        else lender_state.owned_active + lender_state.lent_out,
                        "lender_new_max": None if lender_spec is None else lender_spec.max_connections,
                        "permitted": permitted,
                    }
                    borrow.deadline_mono_ms = mono_ms() + new_config.borrow_ttl_ms
                    action = "recomputed_return"
                    reason = (
                        "new_capacity_permits_release_before_manager_replacement"
                        if permitted
                        else "new_capacity_requires_release_before_manager_replacement"
                    )
                borrow.disposition = action
                if action in {"force_return", "invalidate", "recomputed_return"}:
                    if action == "invalidate":
                        borrow.state = "invalidated"
                    else:
                        borrow.state = "forced_return" if action == "force_return" else "returned"
                    borrow.returned_mono_ms = mono_ms()
                    if conn is not None:
                        self._close_connection_locked(conn, f"version_{action}")
                dispositions.append(
                    {
                        "borrow_id": borrow.borrow_id,
                        "lender_pool": borrow.lender_pool,
                        "borrower_pool": borrow.borrower_pool,
                        "action": action,
                        "reason": reason,
                        "old_deadline_mono_ms": old_deadline,
                        "new_deadline_mono_ms": borrow.deadline_mono_ms,
                        "capacity_recheck": capacity_recheck,
                    }
                )
        return dispositions

    def _connection_for_borrow_locked(self, borrow_id: str) -> ConnectionRuntime | None:
        for state in self.states.values():
            for conn in state.connections.values():
                if conn.borrow_id == borrow_id:
                    return conn
        return None

    def close_all(self, reason: str = "shutdown") -> None:
        with self.condition:
            for state in self.states.values():
                for conn in list(state.connections.values()):
                    if conn.in_use:
                        conn.broken = True
                    self._close_connection_locked(conn, reason)

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=1.0)
        self.close_all("shutdown")

    # ------------------------------------------------------------------
    # Methods used by tests and diagnostics
    # ------------------------------------------------------------------
    def invariant_violations(self) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        with self.condition:
            for pool_id, state in sorted(self.states.items()):
                committed = state.owned_active + state.lent_out
                if committed > state.spec.max_connections:
                    violations.append(
                        {
                            "pool_id": pool_id,
                            "kind": "lender_watermark_overflow",
                            "owned_active": state.owned_active,
                            "lent_out": state.lent_out,
                            "max_connections": state.spec.max_connections,
                        }
                    )
                for conn in state.connections.values():
                    if conn.state not in {"established", "idle", "reused", "closed"}:
                        violations.append(
                            {
                                "pool_id": pool_id,
                                "kind": "invalid_connection_state",
                                "connection_id": conn.connection_id,
                                "state": conn.state,
                            }
                        )
            return violations

    def force_close_connections(self, reason: str = "test_force_close") -> None:
        self.close_all(reason)

    def _record_event_locked(self, conn: ConnectionRuntime, state: str, reason: str) -> None:
        self.event_counter += 1
        self.events.append(
            ConnectionEvent(
                event_seq=self.event_counter,
                connection_id=conn.connection_id,
                pool_id=conn.pool_id,
                state=state,
                reuse_sequence=(conn.total_requests - 1) if conn.total_requests else None,
                reason=reason,
                borrow_id=conn.borrow_id,
                lender_pool=conn.lender_pool,
            )
        )

    def _sample_pool_locked(self, state: _PoolState) -> None:
        state.watermark_samples.append(state.watermark)
        state.detail_samples.append(
            {
                "active_connections": state.active,
                "in_use_connections": state.in_use,
                "idle_connections": state.idle_count,
                "lent_out": state.lent_out,
                "borrowed_in": state.borrowed_in,
                "watermark": state.watermark,
            }
        )
        if self.resource_change_callback is not None:
            try:
                self.resource_change_callback(self._resource_snapshot_locked())
            except Exception:
                # Diagnostic durability must not break request forwarding.
                pass

    def _resource_snapshot_locked(self) -> dict[str, Any]:
        return {
            "connections": [
                {
                    "connection_id": conn.connection_id,
                    "pool_id": conn.pool_id,
                    "state": conn.state,
                    "in_use": conn.in_use,
                    "borrow_id": conn.borrow_id,
                    "lender_pool": conn.lender_pool,
                }
                for state in self.states.values()
                for conn in state.connections.values()
                if conn.state != "closed"
            ],
            "borrows": [
                item.to_dict()
                for item in self.borrows.values()
                if item.state == "active"
            ],
            "captured_mono_ms": mono_ms(),
        }

    def _sample_locked(self, state: _PoolState) -> None:
        self._sample_pool_locked(state)
