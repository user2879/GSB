from __future__ import annotations

import json
import socketserver
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import BusinessError, LeaseExpired, LeaseStale


def now_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass
class LeaseRecord:
    lease_id: str
    lease_key: str
    generation: int
    holder: str
    scopes: list[str]
    slots: int
    issued_ms: int
    expires_ms: int
    ttl_ms: int
    renewal_count: int = 0
    state: str = "active"

    def public(self) -> dict[str, Any]:
        return asdict(self)


class LeaseExecutor:
    """Thread-safe local lease authority and concurrency slot reaper."""

    def __init__(
        self,
        max_slots: int = 2,
        reaper_interval_ms: int = 5,
        clock: Any = now_ms,
    ) -> None:
        self.max_slots = max_slots
        self.reaper_interval_ms = reaper_interval_ms
        self._clock = clock
        self._lock = threading.RLock()
        self._leases: dict[str, LeaseRecord] = {}
        self._by_key: dict[str, str] = {}
        self._generation = 0
        self._events: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop, name="lease-reaper", daemon=True
        )
        self._reaper_thread.start()

    def close(self) -> None:
        self._stop.set()

    def _now(self) -> int:
        return int(self._clock())

    def _event(self, action: str, **details: Any) -> None:
        event = {"at_ms": self._now(), "action": action, **details}
        self._events.append(event)

    def reap_expired(self) -> list[dict[str, Any]]:
        with self._lock:
            current = self._now()
            reaped: list[dict[str, Any]] = []
            for record in list(self._leases.values()):
                if record.state == "active" and current >= record.expires_ms:
                    record.state = "expired"
                    reaped.append(record.public())
                    self._event(
                        "reap",
                        lease_id=record.lease_id,
                        lease_key=record.lease_key,
                        generation=record.generation,
                        expires_ms=record.expires_ms,
                        slots_released=record.slots,
                    )
            return reaped

    def used_slots(self) -> int:
        with self._lock:
            return sum(record.slots for record in self._leases.values() if record.state == "active")

    def acquire(
        self,
        lease_key: str,
        scopes: list[str] | tuple[str, ...],
        ttl_ms: int,
        holder: str = "fixed-render",
        slots: int = 1,
    ) -> dict[str, Any]:
        with self._lock:
            self.reap_expired()
            existing_id = self._by_key.get(lease_key)
            if existing_id:
                existing = self._leases[existing_id]
                if existing.state == "active":
                    return {**existing.public(), "idempotent": True}
            used = self.used_slots()
            if used + slots > self.max_slots:
                raise BusinessError(
                    "E_LEASE_BUSY",
                    f"no concurrency slot: used={used}, requested={slots}, max={self.max_slots}",
                    details={"used_slots": used, "max_slots": self.max_slots},
                )
            self._generation += 1
            current = self._now()
            record = LeaseRecord(
                lease_id=f"lease-{uuid.uuid4().hex[:12]}",
                lease_key=lease_key,
                generation=self._generation,
                holder=holder,
                scopes=list(scopes),
                slots=slots,
                issued_ms=current,
                expires_ms=current + ttl_ms,
                ttl_ms=ttl_ms,
            )
            self._leases[record.lease_id] = record
            self._by_key[lease_key] = record.lease_id
            self._event(
                "acquire",
                lease_id=record.lease_id,
                lease_key=lease_key,
                generation=record.generation,
                issued_ms=record.issued_ms,
                expires_ms=record.expires_ms,
                scopes=record.scopes,
                slots=slots,
            )
            return record.public()

    def _recognized(self, lease_id: str, generation: int) -> LeaseRecord:
        record = self._leases.get(lease_id)
        if record is None:
            raise LeaseStale("lease id is not recognized", lease_id=lease_id, generation=generation)
        if record.generation != generation:
            raise LeaseStale(
                "presented generation does not match recognized generation",
                lease_id=lease_id,
                presented_generation=generation,
                recognized_generation=record.generation,
            )
        return record

    def authorize(self, lease_id: str, generation: int) -> dict[str, Any]:
        with self._lock:
            self.reap_expired()
            record = self._recognized(lease_id, generation)
            if record.state != "active" or self._now() >= record.expires_ms:
                record.state = "expired"
                raise LeaseExpired(
                    "lease expired and its concurrency slot was reclaimed",
                    lease_id=lease_id,
                    generation=generation,
                    expires_ms=record.expires_ms,
                    now_ms=self._now(),
                )
            return record.public()

    def renew(self, lease_id: str, generation: int, ttl_ms: int | None = None) -> dict[str, Any]:
        with self._lock:
            record = self._recognized(lease_id, generation)
            self.reap_expired()
            if record.state != "active" or self._now() >= record.expires_ms:
                record.state = "expired"
                raise LeaseExpired(
                    "cannot renew expired lease; slot has already been reclaimed",
                    lease_id=lease_id,
                    generation=generation,
                    expires_ms=record.expires_ms,
                )
            record.ttl_ms = ttl_ms or record.ttl_ms
            record.expires_ms = self._now() + record.ttl_ms
            record.renewal_count += 1
            self._event(
                "renew",
                lease_id=lease_id,
                generation=generation,
                expires_ms=record.expires_ms,
                renewal_count=record.renewal_count,
            )
            return record.public()

    def release(self, lease_id: str, generation: int) -> dict[str, Any]:
        with self._lock:
            record = self._recognized(lease_id, generation)
            if record.state == "active":
                record.state = "released"
                self._event(
                    "release",
                    lease_id=lease_id,
                    generation=generation,
                    slots_released=record.slots,
                )
            return record.public()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self.reap_expired()
            return {
                "now_ms": self._now(),
                "max_slots": self.max_slots,
                "used_slots": self.used_slots(),
                "generation_counter": self._generation,
                "leases": [record.public() for record in self._leases.values()],
                "events": list(self._events),
            }

    def _reaper_loop(self) -> None:
        while not self._stop.wait(max(0.001, self.reaper_interval_ms / 1000.0)):
            try:
                self.reap_expired()
            except Exception:
                # Reaper must stay alive; request handlers report concrete errors.
                pass


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        executor: LeaseExecutor = self.server.executor  # type: ignore[attr-defined]
        for line in self.rfile:
            try:
                request = json.loads(line.decode("utf-8"))
                action = request.pop("action")
                method = {
                    "acquire": executor.acquire,
                    "renew": executor.renew,
                    "release": executor.release,
                    "authorize": executor.authorize,
                    "snapshot": executor.snapshot,
                    "reap": executor.reap_expired,
                }[action]
                result = method(**request)
                response = {"ok": True, "result": result}
            except BusinessError as exc:
                response = {"ok": False, "code": exc.code, "message": str(exc), "details": exc.details}
            except Exception as exc:  # malformed request is a client error
                response = {"ok": False, "code": "E_BAD_REQUEST", "message": str(exc)}
            self.wfile.write((json.dumps(response, sort_keys=True) + "\n").encode("utf-8"))
            self.wfile.flush()


class LeaseServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, max_slots: int = 2) -> None:
        self.executor = LeaseExecutor(max_slots=max_slots)
        self.server = socketserver.ThreadingTCPServer((host, port), _Handler)
        self.server.executor = self.executor  # type: ignore[attr-defined]
        self.server.daemon_threads = True
        self.host, self.port = self.server.server_address[:2]
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="lease-http-json", daemon=True
        )

    def __enter__(self) -> "LeaseServer":
        self.start()
        return self

    def start(self) -> "LeaseServer":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.executor.close()


class LeaseClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port

    def call(self, action: str, **payload: Any) -> dict[str, Any]:
        import socket

        with socket.create_connection((self.host, self.port), timeout=2.0) as sock:
            sock_file = sock.makefile("rwb")
            sock_file.write((json.dumps({"action": action, **payload}, sort_keys=True) + "\n").encode())
            sock_file.flush()
            line = sock_file.readline()
        response = json.loads(line.decode("utf-8"))
        if not response.get("ok"):
            code = response.get("code", "E_LEASE")
            if code == "E_LEASE_STALE":
                raise LeaseStale(response.get("message", code), **response.get("details", {}))
            if code == "E_LEASE_EXPIRED":
                raise LeaseExpired(response.get("message", code), **response.get("details", {}))
            raise BusinessError(
                code,
                response.get("message", code),
                details=response.get("details", {}),
            )
        return response["result"]

    def acquire(self, **payload: Any) -> dict[str, Any]:
        return self.call("acquire", **payload)

    def renew(self, lease_id: str, generation: int, ttl_ms: int | None = None) -> dict[str, Any]:
        return self.call("renew", lease_id=lease_id, generation=generation, ttl_ms=ttl_ms)

    def authorize(self, lease_id: str, generation: int) -> dict[str, Any]:
        return self.call("authorize", lease_id=lease_id, generation=generation)

    def release(self, lease_id: str, generation: int) -> dict[str, Any]:
        return self.call("release", lease_id=lease_id, generation=generation)

    def snapshot(self) -> dict[str, Any]:
        return self.call("snapshot")
