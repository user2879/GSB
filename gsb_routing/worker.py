"""Physical-database worker and parent-side socket client.

Each physical database runs in a separate child process, listens on its
declared 127.0.0.1 port, and speaks newline-delimited JSON.  The parent opens a
fresh TCP connection for every RPC and performs a timestamp-free handshake on
that same socket before sending the operation.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import E_LEASE, E_PORT, E_REPLY, E_TIMEOUT, E_WORKER, RouteError
from .probe import numpy_route_results
from .spec import DatabaseSpec, Limits, RoutingSpec


def _encode_line(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _read_line(sock: socket.socket, *, max_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("peer closed the socket before a complete JSON line")
        newline = chunk.find(b"\n")
        if newline >= 0:
            chunks.append(chunk[:newline])
            total += newline
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("JSON line exceeds the maximum size")
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid JSON line") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON line must contain an object")
    return value


class _LeaseStore:
    def __init__(self, route_version: int) -> None:
        self.route_version = route_version
        self._leases: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _expire_locked(self, now_ns: int) -> None:
        for lease in self._leases.values():
            if lease["status"] == "active" and now_ns >= lease["deadline_ns"]:
                lease["status"] = "expired"
                lease["decision_reason"] = "local_worker_monotonic_deadline"

    def issue(self, request: dict[str, Any]) -> dict[str, Any]:
        now_ns = time.monotonic_ns()
        lease_id = request.get("lease_id")
        if not isinstance(lease_id, str) or not lease_id:
            raise RouteError(E_LEASE, "request.lease_id", "lease id must be a non-empty string")
        route_version = request.get("route_version")
        if route_version != self.route_version:
            raise RouteError(
                E_LEASE,
                "request.route_version",
                "lease route version does not match worker",
                {"expected": self.route_version, "actual": route_version},
            )
        ttl_ms = request.get("ttl_ms")
        if not isinstance(ttl_ms, int) or isinstance(ttl_ms, bool) or ttl_ms <= 0:
            raise RouteError(E_LEASE, "request.ttl_ms", "ttl_ms must be a positive integer")
        event_seq = request.get("event_seq")
        if not isinstance(event_seq, int) or isinstance(event_seq, bool):
            raise RouteError(E_LEASE, "request.event_seq", "event_seq must be an integer")
        with self._lock:
            self._expire_locked(now_ns)
            existing = self._leases.get(lease_id)
            if existing is not None and existing["status"] == "active":
                raise RouteError(E_LEASE, "request.lease_id", "lease id is already active", {"lease_id": lease_id})
            self._leases[lease_id] = {
                "lease_id": lease_id,
                "event_seq": event_seq,
                "route_version": route_version,
                "version": 1,
                "deadline_ns": now_ns + ttl_ms * 1_000_000,
                "status": "active",
                "decision_reason": None,
            }
            return {"lease_id": lease_id, "lease_version": 1, "route_version": route_version, "ttl_ms": ttl_ms}

    def renew(self, request: dict[str, Any]) -> dict[str, Any]:
        now_ns = time.monotonic_ns()
        lease_id = request.get("lease_id")
        expected_version = request.get("expected_lease_version")
        ttl_ms = request.get("ttl_ms")
        route_version = request.get("route_version")
        if not isinstance(ttl_ms, int) or isinstance(ttl_ms, bool) or ttl_ms <= 0:
            raise RouteError(E_LEASE, "request.ttl_ms", "ttl_ms must be a positive integer")
        with self._lock:
            self._expire_locked(now_ns)
            lease = self._leases.get(lease_id)
            if lease is None:
                raise RouteError(E_LEASE, "request.lease_id", "unknown lease", {"lease_id": lease_id})
            if route_version != self.route_version or lease["route_version"] != self.route_version:
                raise RouteError(
                    E_LEASE,
                    "request.route_version",
                    "lease belongs to another route version",
                    {"expected": self.route_version, "actual": route_version},
                )
            if lease["status"] != "active":
                raise RouteError(
                    E_LEASE,
                    "request.lease_id",
                    "lease is no longer active",
                    {"status": lease["status"], "lease_id": lease_id},
                )
            if expected_version != lease["version"]:
                raise RouteError(
                    E_LEASE,
                    "request.expected_lease_version",
                    "optimistic lease version mismatch; deadline was not extended",
                    {"expected": lease["version"], "actual": expected_version, "lease_id": lease_id},
                )
            lease["version"] += 1
            lease["deadline_ns"] = now_ns + ttl_ms * 1_000_000
            return {
                "lease_id": lease_id,
                "lease_version": lease["version"],
                "route_version": lease["route_version"],
                "ttl_ms": ttl_ms,
            }

    def release(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        expected_version = request.get("lease_version")
        with self._lock:
            self._expire_locked(time.monotonic_ns())
            lease = self._leases.get(lease_id)
            if lease is not None and lease["status"] == "active":
                if expected_version is not None and expected_version != lease["version"]:
                    raise RouteError(
                        E_LEASE,
                        "request.lease_version",
                        "optimistic lease version mismatch",
                        {"expected": lease["version"], "actual": expected_version},
                    )
                lease["status"] = request.get("outcome", "completed")
                lease["decision_reason"] = "worker_release"
            return {"lease_id": lease_id, "status": lease["status"] if lease else "unknown"}

    def validate_for_probe(self, lease_id: Any, expected_version: Any, route_version: Any) -> None:
        now_ns = time.monotonic_ns()
        with self._lock:
            self._expire_locked(now_ns)
            lease = self._leases.get(lease_id)
            if lease is None:
                raise RouteError(E_LEASE, "request.lease_id", "unknown lease", {"lease_id": lease_id})
            if lease["status"] != "active":
                raise RouteError(
                    E_LEASE,
                    "request.lease_id",
                    "lease is not active",
                    {"status": lease["status"], "lease_id": lease_id},
                )
            if lease["route_version"] != self.route_version or route_version != self.route_version:
                raise RouteError(
                    E_LEASE,
                    "request.route_version",
                    "probe route version does not match the lease",
                    {"expected": self.route_version, "actual": route_version},
                )
            if expected_version != lease["version"]:
                raise RouteError(
                    E_LEASE,
                    "request.lease_version",
                    "probe lease version mismatch",
                    {"expected": lease["version"], "actual": expected_version},
                )

    def configure(self, route_version: int, plan: dict[str, Any]) -> list[dict[str, Any]]:
        now_ns = time.monotonic_ns()
        with self._lock:
            self._expire_locked(now_ns)
            if route_version < self.route_version:
                raise RouteError(
                    E_LEASE,
                    "request.route_version",
                    "worker route version cannot move backwards",
                    {"current": self.route_version, "requested": route_version},
                )
            old_version = self.route_version
            revoked: list[dict[str, Any]] = []
            if route_version > old_version:
                for lease in self._leases.values():
                    if lease["status"] == "active":
                        lease["status"] = "version_invalidated"
                        lease["decision_reason"] = "route_version_changed"
                        revoked.append(
                            {
                                "lease_id": lease["lease_id"],
                                "lease_version": lease["version"],
                                "route_version": old_version,
                            }
                        )
            self.route_version = route_version
            return revoked


class _WorkerRuntime:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.database_id = config["db_id"]
        self.host = config["host"]
        self.port = config["port"]
        self.db_seed = config["seed"]
        self.read_timeout_ms = config["read_timeout_ms"]
        self.plan = config["plan"]
        self.route_version = int(self.plan["route_version"])
        self.lease_store = _LeaseStore(self.route_version)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._socket: socket.socket | None = None

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        if request_type == "probe":
            with self._lock:
                plan = self.plan
                route_version = self.route_version
            if request.get("route_version") != route_version:
                raise RouteError(
                    E_LEASE,
                    "request.route_version",
                    "probe route version does not match worker",
                    {"expected": route_version, "actual": request.get("route_version")},
                )
            self.lease_store.validate_for_probe(
                request.get("lease_id"), request.get("lease_version"), request.get("route_version")
            )
            batch_id = request.get("batch_id")
            if not isinstance(batch_id, str) or not batch_id:
                raise RouteError(E_REPLY, "request.batch_id", "batch id must be a non-empty string")
            started_ns = time.perf_counter_ns()
            result = numpy_route_results(
                plan,
                request.get("keys", []),
                database_id=self.database_id,
                batch_id=batch_id,
            )
            return {
                "ok": True,
                "type": "probe_ack",
                "batch_id": batch_id,
                "database_id": self.database_id,
                "route_version": route_version,
                "hits": result["results"],
                "probe_digest": result["probe_digest"],
                "fixedpoint_checksum_u64": result["fixedpoint_checksum_u64"],
                "elapsed_us": max(0, (time.perf_counter_ns() - started_ns) // 1000),
            }
        if request_type == "issue_lease":
            return {"ok": True, "type": "lease_ack", **self.lease_store.issue(request)}
        if request_type == "renew_lease":
            return {"ok": True, "type": "lease_ack", **self.lease_store.renew(request)}
        if request_type == "release_lease":
            return {"ok": True, "type": "lease_ack", **self.lease_store.release(request)}
        if request_type == "configure":
            new_plan = request.get("plan")
            if not isinstance(new_plan, dict):
                raise RouteError(E_REPLY, "request.plan", "plan must be an object")
            new_version = request.get("route_version")
            if not isinstance(new_version, int) or isinstance(new_version, bool):
                raise RouteError(E_REPLY, "request.route_version", "route version must be an integer")
            if new_plan.get("route_version") != new_version:
                raise RouteError(E_REPLY, "request.plan.route_version", "plan and request versions differ")
            revoked = self.lease_store.configure(new_version, new_plan)
            with self._lock:
                self.plan = new_plan
                self.route_version = new_version
            return {
                "ok": True,
                "type": "configure_ack",
                "database_id": self.database_id,
                "route_version": new_version,
                "revoked": revoked,
            }
        if request_type == "ping":
            return {"ok": True, "type": "pong", "database_id": self.database_id, "route_version": self.route_version}
        raise RouteError(E_REPLY, "request.type", "unsupported worker request", {"type": request_type})

    def _serve_connection(self, connection: socket.socket) -> None:
        try:
            connection.settimeout(self.read_timeout_ms / 1000.0)
            handshake = _read_line(connection)
            if handshake.get("type") != "handshake":
                raise RouteError(E_REPLY, "handshake.type", "expected handshake")
            nonce = handshake.get("nonce")
            connection.sendall(
                _encode_line(
                    {
                        "ok": True,
                        "type": "handshake_ack",
                        "database_id": self.database_id,
                        "port": self.port,
                        "route_version": self.route_version,
                        "nonce": nonce,
                    }
                )
            )
            while not self._stop.is_set():
                try:
                    request = _read_line(connection)
                except socket.timeout:
                    raise
                if request.get("type") == "close":
                    break
                try:
                    response = self.handle_request(request)
                    connection.sendall(_encode_line(response))
                except RouteError as exc:
                    connection.sendall(_encode_line(exc.to_dict()))
                except Exception as exc:  # keep the worker alive and return a typed error
                    connection.sendall(
                        _encode_line(
                            RouteError(
                                E_REPLY,
                                "request",
                                "worker internal request failure",
                                {"error": type(exc).__name__, "message": str(exc)},
                            ).to_dict()
                        )
                    )
        except socket.timeout:
            try:
                connection.sendall(
                    _encode_line(
                        RouteError(
                            E_TIMEOUT,
                            "handshake" if "handshake" not in locals() else "request",
                            "worker socket read timeout",
                        ).to_dict()
                    )
                )
            except OSError:
                pass
        except (ConnectionError, ValueError, OSError):
            pass
        finally:
            try:
                connection.close()
            except OSError:
                pass

    def serve_forever(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
            sock.listen(128)
        except OSError as exc:
            print(
                json.dumps(
                    {
                        "event": "startup_error",
                        "code": E_PORT,
                        "path": f"databases[port={self.port}].port",
                        "reason": "cannot bind loopback port",
                        "details": {"port": self.port, "errno": getattr(exc, "errno", None)},
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
            sock.close()
            return 2
        self._socket = sock
        sock.settimeout(0.25)
        print(
            json.dumps(
                {
                    "event": "ready",
                    "database_id": self.database_id,
                    "host": self.host,
                    "port": self.port,
                    "route_version": self.route_version,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )

        def stop_handler(signum: int, frame: Any) -> None:
            self._stop.set()
            try:
                sock.close()
            except OSError:
                pass

        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        while not self._stop.is_set():
            try:
                connection, _ = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(target=self._serve_connection, args=(connection,), daemon=True)
            thread.start()
        return 0


def worker_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gsb-routing-worker")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        runtime = _WorkerRuntime(config)
        return runtime.serve_forever()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "startup_error",
                    "code": E_WORKER,
                    "path": "worker",
                    "reason": "worker failed during startup",
                    "details": {"error": type(exc).__name__, "message": str(exc)},
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
        return 3


@dataclass
class WorkerProcessInfo:
    database_id: str
    pid: int
    port: int
    route_version: int
    config_path: str


class WorkerClient:
    """Parent-side client owning one child process."""

    def __init__(
        self,
        database: DatabaseSpec,
        spec: RoutingSpec,
        *,
        workdir: str | Path,
    ) -> None:
        self.database = database
        self.spec = spec
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.limits = spec.limits
        self.plan = __import__("gsb_routing.probe", fromlist=["build_route_plan"]).build_route_plan(spec)
        self.process: subprocess.Popen[bytes] | None = None
        self.info: WorkerProcessInfo | None = None
        self._stderr_path: Path | None = None
        self._config_path: Path | None = None

    @property
    def port(self) -> int:
        return self.database.port

    def _read_startup(self, deadline: float) -> dict[str, Any]:
        assert self.process is not None and self.process.stdout is not None
        stdout = self.process.stdout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RouteError(
                    E_TIMEOUT,
                    f"databases[id={self.database.id}].port",
                    "worker startup/pipe handshake timeout",
                    {"port": self.port},
                )
            ready, _, _ = select.select([stdout], [], [], remaining)
            if not ready:
                if self.process.poll() is not None:
                    break
                continue
            line = stdout.readline()
            if not line:
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("event") == "startup_error":
                raise RouteError(
                    value.get("code", E_WORKER),
                    value.get("path", "worker"),
                    value.get("reason", "worker startup failed"),
                    value.get("details", {}),
                )
            if value.get("event") == "ready":
                return value
        stderr_text = ""
        if self._stderr_path is not None and self._stderr_path.exists():
            stderr_text = self._stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RouteError(
            E_WORKER,
            f"databases[id={self.database.id}].port",
            "worker exited before reporting readiness",
            {"port": self.port, "stderr_tail": stderr_text},
        )

    def start(self) -> WorkerProcessInfo:
        if self.process is not None and self.process.poll() is None:
            assert self.info is not None
            return self.info
        config = {
            "db_id": self.database.id,
            "host": self.database.host,
            "port": self.database.port,
            "seed": self.database.seed,
            "route_version": self.spec.route_version,
            "read_timeout_ms": self.spec.limits.socket_read_timeout_ms,
            "plan": self.plan,
        }
        fd, config_path = tempfile.mkstemp(prefix=f"gsb-worker-{self.database.id}-", suffix=".json", dir=str(self.workdir))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        self._config_path = Path(config_path)
        stderr_file = tempfile.NamedTemporaryFile(
            prefix=f"gsb-worker-{self.database.id}-",
            suffix=".stderr.log",
            dir=str(self.workdir),
            delete=False,
        )
        stderr_file.close()
        self._stderr_path = Path(stderr_file.name)
        package_root = str(Path(__file__).resolve().parent.parent)
        env = dict(os.environ)
        env["PYTHONPATH"] = package_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with self._stderr_path.open("wb") as stderr_handle:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "gsb_routing.worker", "--config", config_path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                env=env,
            )
        deadline = time.monotonic() + self.limits.handshake_timeout_ms / 1000.0
        try:
            ready = self._read_startup(deadline)
        except Exception:
            self.kill()
            raise
        if ready.get("database_id") != self.database.id or int(ready.get("port", -1)) != self.database.port:
            self.kill()
            raise RouteError(
                E_REPLY,
                "worker.ready",
                "worker readiness binding mismatch",
                {"database_id": self.database.id, "port": self.database.port},
            )
        nonce = f"startup-{os.getpid()}-{self.database.id}"
        try:
            self._rpc({"type": "ping"}, connect_timeout=self.limits.handshake_timeout_ms, nonce=nonce)
        except Exception:
            self.kill()
            raise
        assert self.process is not None
        self.info = WorkerProcessInfo(
            database_id=self.database.id,
            pid=self.process.pid,
            port=self.database.port,
            route_version=self.spec.route_version,
            config_path=config_path,
        )
        return self.info

    def _rpc(
        self,
        payload: dict[str, Any],
        *,
        connect_timeout: int | None = None,
        nonce: str | None = None,
    ) -> dict[str, Any]:
        connect_seconds = (connect_timeout or self.limits.handshake_timeout_ms) / 1000.0
        read_seconds = self.limits.socket_read_timeout_ms / 1000.0
        if nonce is None:
            nonce = f"{os.getpid()}-{time.monotonic_ns()}"
        try:
            socket_obj = socket.create_connection((self.database.host, self.database.port), timeout=connect_seconds)
        except socket.timeout as exc:
            raise RouteError(
                E_TIMEOUT,
                f"databases[id={self.database.id}].port",
                "TCP connect timeout",
                {"port": self.database.port},
            ) from exc
        except OSError as exc:
            raise RouteError(
                E_WORKER,
                f"databases[id={self.database.id}].port",
                "worker TCP connection failed",
                {"port": self.database.port, "error": type(exc).__name__},
            ) from exc
        try:
            socket_obj.settimeout(read_seconds)
            socket_obj.sendall(_encode_line({"type": "handshake", "nonce": nonce}))
            handshake = _read_line(socket_obj)
            if handshake.get("type") != "handshake_ack" or handshake.get("nonce") != nonce:
                raise RouteError(E_REPLY, "handshake", "invalid or unbound handshake reply")
            if handshake.get("database_id") != self.database.id:
                raise RouteError(
                    E_REPLY,
                    "handshake.database_id",
                    "handshake database binding mismatch",
                    {"expected": self.database.id, "actual": handshake.get("database_id")},
                )
            socket_obj.sendall(_encode_line(payload))
            response = _read_line(socket_obj)
        except socket.timeout as exc:
            phase = "handshake" if "handshake" not in locals() else "read"
            raise RouteError(
                E_TIMEOUT,
                f"databases[id={self.database.id}].{phase}",
                "socket handshake/read timeout",
                {"port": self.database.port, "phase": phase},
            ) from exc
        except ConnectionError as exc:
            raise RouteError(
                E_WORKER,
                f"databases[id={self.database.id}].socket",
                "worker closed the TCP connection",
                {"port": self.database.port},
            ) from exc
        except ValueError as exc:
            raise RouteError(E_REPLY, "reply", "invalid JSON reply from worker") from exc
        finally:
            try:
                socket_obj.close()
            except OSError:
                pass
        if not response.get("ok"):
            raise RouteError(
                response.get("code", E_REPLY),
                response.get("path", "response"),
                response.get("reason", "worker rejected request"),
                response.get("details", {}),
            )
        return response

    def probe(
        self,
        *,
        batch_id: str,
        keys: list[dict[str, Any]],
        route_version: int,
        lease_id: str,
        lease_version: int,
    ) -> dict[str, Any]:
        response = self._rpc(
            {
                "type": "probe",
                "batch_id": batch_id,
                "keys": keys,
                "route_version": route_version,
                "lease_id": lease_id,
                "lease_version": lease_version,
            }
        )
        if (
            response.get("database_id") != self.database.id
            or int(response.get("route_version", -1)) != route_version
        ):
            raise RouteError(
                E_REPLY,
                "reply.binding",
                "probe receipt is not bound to the database and route version",
                {
                    "database_id": response.get("database_id"),
                    "route_version": response.get("route_version"),
                },
            )
        return response

    def issue_lease(self, *, lease_id: str, event_seq: int, route_version: int, ttl_ms: int) -> dict[str, Any]:
        return self._rpc(
            {
                "type": "issue_lease",
                "lease_id": lease_id,
                "event_seq": event_seq,
                "route_version": route_version,
                "ttl_ms": ttl_ms,
            }
        )

    def renew_lease(self, *, lease_id: str, expected_lease_version: int, route_version: int, ttl_ms: int) -> dict[str, Any]:
        return self._rpc(
            {
                "type": "renew_lease",
                "lease_id": lease_id,
                "expected_lease_version": expected_lease_version,
                "route_version": route_version,
                "ttl_ms": ttl_ms,
            }
        )

    def release_lease(self, *, lease_id: str, lease_version: int, outcome: str) -> dict[str, Any]:
        return self._rpc(
            {
                "type": "release_lease",
                "lease_id": lease_id,
                "lease_version": lease_version,
                "outcome": outcome,
            }
        )

    def configure(self, spec: RoutingSpec) -> dict[str, Any]:
        from .probe import build_route_plan

        new_plan = build_route_plan(spec)
        response = self._rpc({"type": "configure", "route_version": spec.route_version, "plan": new_plan})
        self.plan = new_plan
        self.spec = spec
        if self.info is not None:
            self.info.route_version = spec.route_version
        return response

    def kill(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=max(0.5, self.limits.handshake_timeout_ms / 1000.0))
        except subprocess.TimeoutExpired:
            pass
        if self.process.stdout is not None:
            try:
                self.process.stdout.close()
            except OSError:
                pass
        self.process = None
        self.info = None

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=max(0.5, self.limits.handshake_timeout_ms / 1000.0))
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1.0)
        if self.process.stdout is not None:
            try:
                self.process.stdout.close()
            except OSError:
                pass
        self.process = None
        self.info = None

    def restart(self, spec: RoutingSpec | None = None) -> WorkerProcessInfo:
        self.kill()
        if spec is not None:
            self.spec = spec
            from .probe import build_route_plan

            self.plan = build_route_plan(spec)
        return self.start()

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


class WorkerManager:
    def __init__(self, spec: RoutingSpec, *, workdir: str | Path) -> None:
        self.spec = spec
        self.workdir = Path(workdir)
        self.clients: dict[str, WorkerClient] = {
            database.id: WorkerClient(database, spec, workdir=self.workdir)
            for database in sorted(spec.databases, key=lambda item: item.id)
        }

    def __enter__(self) -> "WorkerManager":
        self.start_all()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop_all()

    def start_all(self) -> dict[str, WorkerProcessInfo]:
        started: dict[str, WorkerProcessInfo] = {}
        try:
            for database_id in sorted(self.clients):
                started[database_id] = self.clients[database_id].start()
        except Exception:
            self.stop_all()
            raise
        return started

    def stop_all(self) -> None:
        for client in self.clients.values():
            try:
                client.stop()
            except Exception:
                try:
                    client.kill()
                except Exception:
                    pass

    def kill(self, database_id: str) -> None:
        self.clients[database_id].kill()

    def restart(self, database_id: str, spec: RoutingSpec | None = None) -> WorkerProcessInfo:
        return self.clients[database_id].restart(spec)

    def restart_dead(self) -> dict[str, WorkerProcessInfo]:
        restarted: dict[str, WorkerProcessInfo] = {}
        for database_id, client in self.clients.items():
            if not client.alive():
                restarted[database_id] = client.restart(self.spec)
        return restarted

    def configure(self, spec: RoutingSpec) -> dict[str, dict[str, Any]]:
        responses: dict[str, dict[str, Any]] = {}
        for database_id in sorted(self.clients):
            responses[database_id] = self.clients[database_id].configure(spec)
        self.spec = spec
        return responses


if __name__ == "__main__":
    raise SystemExit(worker_main())
