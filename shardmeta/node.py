"""Node subprocess, loopback JSON-lines transport and client (requirement 4).

Each node runs in its own process, binds a 127.0.0.1 TCP port and speaks one
JSON object per line. The client maps concrete failures to E_PORT, E_TIMEOUT
and E_REPLY. Digest computation is shared, deterministic NumPy fixed-point code.
"""
from __future__ import annotations

import base64
import json
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, Optional

from .clock import monotonic_ms
from .content import compute_checksum, generate_shard_bytes
from .errors import PortError, ReplyError, TimeoutError
from .lease import LeaseManager
from .models import NodeSpec

REQUIRED_REPLY_FIELDS = ("type", "ok", "node_id", "elapsed_ms")


def _send_line(sock: socket.socket, payload: dict) -> None:
    sock.sendall(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")


def _recv_line(sock: socket.socket) -> dict:
    buffer = bytearray()
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("node closed connection")
        buffer.extend(chunk)
        if b"\n" in buffer:
            line, _, _ = buffer.partition(b"\n")
            return json.loads(line.decode("utf-8"))


class NodeServer:
    """In-process threaded loopback server executed inside the subprocess."""

    def __init__(self, spec: NodeSpec) -> None:
        self.spec = spec
        self.manager = LeaseManager(
            spec.node_id, spec.capacity_slots, spec.lease_limit,
        )
        self._server: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.spec.host, self.spec.port))
        except OSError as exc:
            server.close()
            raise PortError(f"cannot bind {self.spec.host}:{self.spec.port}: {exc}",
                            node=self.spec.node_id, field="port") from exc
        server.listen(16)
        server.settimeout(0.25)
        self._server = server
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            request: dict = {}
            try:
                request = _recv_line(conn)
                reply = self.dispatch(request)
            except Exception as exc:  # noqa: BLE001 - report, never crash the node
                reply = {
                    "type": request.get("type", "unknown"),
                    "ok": False,
                    "node_id": self.spec.node_id,
                    "elapsed_ms": 0,
                    "error": {"code": "E_LEASE", "message": str(exc)},
                }
            _send_line(conn, reply)

    def dispatch(self, request: dict) -> dict:
        start = monotonic_ms()
        kind = request.get("type")
        reply: dict[str, Any] = {
            "type": kind,
            "ok": True,
            "node_id": self.spec.node_id,
            "meta_version": self.spec.meta_version,
        }
        key = request.get("object_key", "")
        shard_index = int(request.get("shard_index", 0))
        if kind == "ping":
            pass
        elif kind == "grant":
            lease = self.manager.grant(key, shard_index, int(request["ttl_ms"]))
            reply["lease"] = lease.to_dict()
        elif kind == "renew":
            lease = self.manager.renew(
                request["lease_id"], int(request["expected_lease_version"]),
                int(request["ttl_ms"]),
            )
            reply["lease"] = lease.to_dict()
        elif kind == "abort":
            lease = self.manager.abort(request["lease_id"])
            reply["lease"] = lease.to_dict()
        elif kind == "upload":
            data = generate_shard_bytes(
                self.spec.seed, int(request["object_seed"]), key, shard_index,
                int(request["size"]),
            )
            checksum = compute_checksum(self.spec.node_id, key, shard_index, data)
            reply["merge_id"] = request["merge_id"]
            reply["checksum"] = checksum.as_dict()
            reply["content_seed"] = int(request["object_seed"])
        elif kind == "restore":
            # Recover lease authority solely from committed ledger records.
            from .models import Lease as _Lease
            restored = 0
            for lease_dict in request.get("leases", []):
                lease = _Lease(**lease_dict)
                self.manager.restore(lease)
                restored += 1
            reply["restored_lease_count"] = restored
            reply["active_count"] = self.manager.state.active_count
        elif kind == "sweep":
            expired = self.manager.sweep()
            reply["expired_lease_ids"] = [lease.lease_id for lease in expired]
            reply["active_count"] = self.manager.state.active_count
        else:
            reply["ok"] = False
            reply["error"] = {"code": "E_REPLY", "message": f"unknown request type {kind}"}
        reply["elapsed_ms"] = monotonic_ms() - start
        return reply

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _run_subprocess(spec_b64: str) -> None:
    spec_dict = json.loads(base64.b64decode(spec_b64.encode("ascii")).decode("utf-8"))
    spec = NodeSpec(**spec_dict)
    server = NodeServer(spec)
    server.start()
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


@dataclass
class NodeProcess:
    spec: NodeSpec
    process: subprocess.Popen
    client: "NodeClient"


class NodeClient:
    """Synchronous one-request-per-connection loopback JSON-lines client."""

    def __init__(self, spec: NodeSpec, timeout_ms: int = 3000) -> None:
        self.spec = spec
        self.timeout = timeout_ms / 1000.0

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect((self.spec.host, self.spec.port))
        except ConnectionRefusedError as exc:
            sock.close()
            raise PortError(f"node {self.spec.node_id} not reachable on "
                            f"{self.spec.host}:{self.spec.port}",
                            node=self.spec.node_id, field="port") from exc
        except socket.timeout as exc:
            sock.close()
            # An un-connectable loopback endpoint is a port/binding problem.
            raise PortError("handshake timed out: port not accepting connections",
                            node=self.spec.node_id, field="port") from exc
        return sock

    def call(self, payload: dict, expect_fields: Optional[tuple[str, ...]] = None) -> dict:
        start = monotonic_ms()
        try:
            sock = self._connect()
            with sock:
                _send_line(sock, payload)
                try:
                    reply = _recv_line(sock)
                except socket.timeout as exc:
                    raise TimeoutError("reply timed out",
                                       node=self.spec.node_id) from exc
                except (json.JSONDecodeError, ConnectionError) as exc:
                    raise ReplyError("malformed reply from node",
                                     node=self.spec.node_id) from exc
        except socket.timeout as exc:
            raise TimeoutError("loopback call timed out",
                               node=self.spec.node_id) from exc
        reply["client_elapsed_ms"] = monotonic_ms() - start
        for field_name in REQUIRED_REPLY_FIELDS:
            if field_name not in reply:
                raise ReplyError(f"reply missing required field {field_name}",
                                 node=self.spec.node_id, field=field_name)
        if expect_fields:
            for field_name in expect_fields:
                if field_name not in reply:
                    raise ReplyError(f"reply missing field {field_name}",
                                     node=self.spec.node_id, field=field_name)
        if not reply.get("ok", False):
            error = reply.get("error") or {}
            raise ReplyError(
                error.get("message", "node returned ok=false"),
                node=self.spec.node_id, detail=error,
            )
        return reply

    def ping(self) -> dict:
        return self.call({"type": "ping"})


class NodeOrchestrator:
    """Spawns and supervises one subprocess per declared node."""

    def __init__(self, specs: dict[str, NodeSpec], timeout_ms: int = 3000,
                 python_executable: Optional[str] = None) -> None:
        self.specs = specs
        self.timeout_ms = timeout_ms
        self._python = python_executable or sys.executable
        self._processes: dict[str, subprocess.Popen] = {}
        self._clients: dict[str, NodeClient] = {}

    def start_all(self) -> None:
        for node_id, spec in self.specs.items():
            spec_b64 = base64.b64encode(
                json.dumps(spec.to_dict(), separators=(",", ":")).encode("utf-8")
            ).decode("ascii")
            process = subprocess.Popen(
                [self._python, "-m", "shardmeta.node", spec_b64],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._processes[node_id] = process
            client = NodeClient(spec, timeout_ms=self.timeout_ms)
            self._await_ready(client, node_id)
            self._clients[node_id] = client

    def _await_ready(self, client: NodeClient, node_id: str) -> None:
        deadline = monotonic_ms() + self.timeout_ms
        last_error: Optional[Exception] = None
        while monotonic_ms() < deadline:
            try:
                client.ping()
                return
            except (PortError, ReplyError, ConnectionError) as exc:
                last_error = exc
        raise TimeoutError(f"node {node_id} did not become ready in time",
                           node=node_id, detail=str(last_error))

    def client(self, node_id: str) -> NodeClient:
        return self._clients[node_id]

    def is_alive(self, node_id: str) -> bool:
        process = self._processes.get(node_id)
        return process is not None and process.poll() is None

    def kill_node(self, node_id: str) -> None:
        process = self._processes.get(node_id)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()

    def restart_node(self, node_id: str, recovery_leases: list[dict] | None = None) -> None:
        self.kill_node(node_id)
        spec = self.specs[node_id]
        spec_b64 = base64.b64encode(
            json.dumps(spec.to_dict(), separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        process = subprocess.Popen(
            [self._python, "-m", "shardmeta.node", spec_b64],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes[node_id] = process
        client = NodeClient(spec, timeout_ms=self.timeout_ms)
        self._await_ready(client, node_id)
        self._clients[node_id] = client
        if recovery_leases is not None:
            client.call(
                {"type": "restore", "leases": recovery_leases},
                expect_fields=("restored_lease_count",),
            )

    def stop_all(self) -> None:
        for process in self._processes.values():
            if process.poll() is None:
                process.terminate()
        for process in self._processes.values():
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._processes.clear()
        self._clients.clear()

    def __enter__(self) -> "NodeOrchestrator":
        self.start_all()
        return self

    def __exit__(self, *exc) -> None:
        self.stop_all()


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m shardmeta.node <base64-spec>")
    _run_subprocess(sys.argv[1])


__all__ = [
    "NodeServer",
    "NodeClient",
    "NodeOrchestrator",
    "NodeProcess",
]
