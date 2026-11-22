"""Loopback TCP coordinator with JSON-lines protocol.

Only 127.0.0.1 high ports are used and the port is probed before binding.
Three constraints govern parallel grants:

* a global cap on simultaneously running grants (semaphore),
* a per-pool quota on simultaneously running grants (per-pool semaphore),
* a single-grant socket timeout on the client (real ``socket.settimeout``).

Waiting is produced by genuine socket timeout/blocking behaviour, never by
sleeping to fake latency.  Retries carry the same request id so adjudication
stays idempotent and a grant's semantics never change across attempts.
"""
from __future__ import annotations

import json
import socket
import threading

from .errors import E_TIMEOUT
from .model import EV_QUEUE, Request

LOOPBACK = "127.0.0.1"
HIGH_PORT_FLOOR = 20000
HIGH_PORT_CEIL = 60000


def is_port_free(host: str = LOOPBACK, port: int = 0) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def pick_free_port(preferred: int | None = None) -> int:
    if preferred is not None:
        if not (HIGH_PORT_FLOOR <= preferred <= HIGH_PORT_CEIL):
            raise ValueError("port must be a high ephemeral-style port")
        if not is_port_free(port=preferred):
            raise OSError(f"preferred port {preferred} is not free")
        return preferred
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind((LOOPBACK, 0))
    port = listener.getsockname()[1]
    listener.close()
    if not (HIGH_PORT_FLOOR <= port <= HIGH_PORT_CEIL):
        return pick_free_port(None)
    return port


class CoordinatorServer:
    def __init__(self, engine, clock, pools: dict, *, global_cap: int,
                 host: str = LOOPBACK, port: int | None = None) -> None:
        self.engine = engine
        self.clock = clock
        self.pools = pools
        self.global_cap = global_cap
        self.host = host
        self.port = pick_free_port(port)
        self._engine_lock = threading.Lock()
        self._global_slots = threading.BoundedSemaphore(global_cap)
        self._pool_slots = {
            pid: threading.BoundedSemaphore(pool.quota)
            for pid, pool in pools.items()}
        self.queue_stats: list[dict] = []
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((self.host, self.port))
        self._listener.listen(16)
        self._listener.settimeout(0.2)
        self._thread = threading.Thread(
            target=self._accept_loop, name="coordinator-accept", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------- accept
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,),
                             name="coordinator-conn", daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            buffer = b""
            while not self._stop.is_set():
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        message = json.loads(line.decode("utf-8"))
                        response = self._dispatch(message)
                    except Exception as exc:  # protocol-level safety
                        response = {"ok": False, "code": "E_INTERNAL",
                                    "message": str(exc)}
                    conn.sendall(
                        (json.dumps(response, sort_keys=True) + "\n")
                        .encode("utf-8"))
        except OSError:
            return
        finally:
            conn.close()

    # ---------------------------------------------------------- scheduling
    def _dispatch(self, message: dict) -> dict:
        request = Request(
            message["request_id"], message["key"], message["pool_id"],
            message["revision"], message["priority"], message["arrive_ms"])
        kind = message.get("kind", "GRANT")
        now_ms = self.clock.now_ms()
        if kind == "RENEW":
            # Renewals do not consume grant slots; ownership is checked inside
            # the engine against the explicit state-machine permission table.
            with self._engine_lock:
                self.engine.tick(now_ms)
                decision = self.engine.submit(request, now_ms)
            return self._pack(decision, waited_ms=0)

        wait_start = self.clock.now_ms()
        # Global grant cap: blocks while the maximum number of grants run.
        self._global_slots.acquire()
        pool_slot = self._pool_slots[request.pool_id]
        try:
            # Per-pool quota: the pool-specific semaphore produces genuine
            # queueing; no artificial sleeps are used to simulate waiting.
            acquired = pool_slot.acquire(timeout=0.05)
            if not acquired:
                self._record_queue(request, wait_start, "waiting")
                pool_slot.acquire()
            wait_end = self.clock.now_ms()
            waited_ms = wait_end - wait_start
            if not acquired:
                self._record_queue(request, wait_end, "acquired", waited_ms)
                with self._engine_lock:
                    self.engine._emit(
                        EV_QUEUE, wait_end, request_id=request.request_id,
                        key=request.key, pool_id=request.pool_id,
                        detail={"wait_ms": waited_ms,
                                "wait_start_ms": wait_start,
                                "wait_end_ms": wait_end})
            with self._engine_lock:
                self.engine.tick(self.clock.now_ms())
                decision = self.engine.submit(request, self.clock.now_ms())
            return self._pack(decision, waited_ms, wait_start, wait_end)
        finally:
            try:
                pool_slot.release()
            except ValueError:
                pass
            self._global_slots.release()

    def _record_queue(self, request: Request, at_ms: int, phase: str,
                      waited_ms: int | None = None) -> None:
        self.queue_stats.append({
            "request_id": request.request_id, "pool_id": request.pool_id,
            "at_ms": at_ms, "phase": phase,
            "waited_ms": waited_ms,
        })

    @staticmethod
    def _pack(decision, waited_ms: int, wait_start: int | None = None,
              wait_end: int | None = None) -> dict:
        payload = decision.to_dict()
        payload["waited_ms"] = waited_ms
        if wait_start is not None:
            payload["wait_start_ms"] = wait_start
            payload["wait_end_ms"] = wait_end
        payload["ok"] = decision.accepted
        return payload


class LeaseClient:
    """JSON-lines client enforcing a real per-grant socket timeout."""

    def __init__(self, host: str, port: int, timeout_ms: int,
                 max_retries: int = 2) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries

    def send(self, message: dict) -> dict:
        attempts = 0
        last_error: dict | None = None
        while attempts <= self.max_retries:
            attempts += 1
            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn.settimeout(self.timeout_ms / 1000.0)
            try:
                conn.connect((self.host, self.port))
                conn.sendall(
                    (json.dumps(message, sort_keys=True) + "\n")
                    .encode("utf-8"))
                buffer = b""
                while b"\n" not in buffer:
                    buffer += conn.recv(65536)
                line = buffer.split(b"\n", 1)[0]
                response = json.loads(line.decode("utf-8"))
                response["attempts"] = attempts
                return response
            except (socket.timeout, TimeoutError) as exc:
                last_error = {"ok": False, "code": E_TIMEOUT,
                              "message": str(exc), "attempts": attempts,
                              "request_id": message["request_id"]}
                continue
            finally:
                conn.close()
        assert last_error is not None
        return last_error
