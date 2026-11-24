from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json
from .errors import CacheLeaseError, HandshakeError, ProtocolError, ReplyError
from .node import PROTOCOL_VERSION, mono_ms
from .spec import Config, NodeSpec


class JsonLineClient:
    def __init__(self, host: str, port: int, timeout_ms: int) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms

    def request(self, payload: dict[str, Any], timeout_ms: int | None = None) -> dict[str, Any]:
        timeout = (self.timeout_ms if timeout_ms is None else timeout_ms) / 1000.0
        timeout = max(0.001, timeout)
        data = (canonical_json(payload) + "\n").encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                sock.sendall(data)
                chunks = bytearray()
                while b"\n" not in chunks:
                    block = sock.recv(65536)
                    if not block:
                        raise ReplyError("node closed connection before replying", _endpoint(self))
                    chunks.extend(block)
                    if len(chunks) > 4 * 1024 * 1024:
                        raise ReplyError("node reply exceeds 4 MiB", _endpoint(self))
                line = bytes(chunks).split(b"\n", 1)[0]
        except socket.timeout as exc:
            raise ReplyError(f"real socket timeout after {timeout_ms or self.timeout_ms} ms", _endpoint(self)) from exc
        except OSError as exc:
            raise ReplyError(f"socket error: {exc}", _endpoint(self)) from exc
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReplyError(f"invalid JSON reply: {exc}", _endpoint(self)) from exc
        if not isinstance(response, dict):
            raise ReplyError("reply must be a JSON object", _endpoint(self))
        return response


def _endpoint(client: JsonLineClient) -> str:
    return f"{client.host}:{client.port}"


def require_fields(payload: dict[str, Any], fields: Iterable[str], location: str) -> None:
    for field_name in fields:
        if field_name not in payload:
            raise ReplyError(f"reply is missing required field {field_name!r}", f"{location}.{field_name}")


class NodeClient:
    def __init__(self, spec: NodeSpec, timeout_ms: int) -> None:
        self.spec = spec
        self.timeout_ms = timeout_ms
        self.transport = JsonLineClient(spec.host, spec.port, timeout_ms)
        self.status_snapshot: dict[str, Any] = {}

    @property
    def node_id(self) -> str:
        return self.spec.node_id

    def handshake(self, deadline_ms: int) -> dict[str, Any]:
        last_error: Exception | None = None
        while mono_ms() < deadline_ms:
            remaining = max(1, deadline_ms - mono_ms())
            timeout = min(self.timeout_ms, remaining)
            try:
                response = self.transport.request({"op": "hello", "protocol_version": PROTOCOL_VERSION}, timeout)
            except ReplyError as exc:
                last_error = exc
                time.sleep(0.01)
                continue
            if not response.get("ok"):
                last_error = HandshakeError(
                    f"node rejected handshake: {response.get('error')} {response.get('message', '')}",
                    self.node_id,
                )
                time.sleep(0.01)
                continue
            require_fields(
                response,
                ("protocol_version", "node_id", "quota_version", "status", "content_rule"),
                f"handshake[{self.node_id}]",
            )
            if response["protocol_version"] != PROTOCOL_VERSION:
                raise HandshakeError(
                    f"protocol version mismatch: {response['protocol_version']} != {PROTOCOL_VERSION}",
                    self.node_id,
                )
            if response["node_id"] != self.node_id:
                raise HandshakeError(f"handshake node id mismatch: {response['node_id']}", self.node_id)
            status = response["status"]
            if not isinstance(status, dict):
                raise ReplyError("handshake status must be an object", f"handshake[{self.node_id}].status")
            require_fields(status, ("quota_version", "watermark", "entries", "leases", "reservations"), f"handshake[{self.node_id}].status")
            if not isinstance(response["content_rule"], str) or not response["content_rule"]:
                raise ReplyError("handshake content_rule must be a non-empty string", f"handshake[{self.node_id}].content_rule")
            self.status_snapshot = status
            return response
        if last_error is not None:
            raise HandshakeError(
                f"handshake failed before timeout: {last_error}", f"{self.spec.host}:{self.spec.port}"
            ) from last_error
        raise HandshakeError("handshake timed out", f"{self.spec.host}:{self.spec.port}")

    def call(self, payload: dict[str, Any], expected: Iterable[str] = ()) -> dict[str, Any]:
        response = self.transport.request(payload)
        if not response.get("ok"):
            code = str(response.get("error", "E_REPLY"))
            raise ProtocolError(
                code,
                str(response.get("message", "node returned an error")),
                f"node[{self.node_id}]",
                response.get("details") or {},
            )
        require_fields(response, expected, f"node[{self.node_id}]")
        require_fields(response, ("elapsed_ms",), f"node[{self.node_id}]")
        elapsed = response["elapsed_ms"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
            raise ReplyError("elapsed_ms must be a non-negative integer", f"node[{self.node_id}].elapsed_ms")
        return response

    def status(self) -> dict[str, Any]:
        response = self.call({"op": "status"}, ("status",))
        status = response["status"]
        if not isinstance(status, dict):
            raise ReplyError("status must be an object", f"node[{self.node_id}].status")
        require_fields(
            status,
            ("quota_version", "capacity_slots", "lease_limit", "watermark", "entries", "leases", "reservations"),
            f"node[{self.node_id}].status",
        )
        self.status_snapshot = status
        return status

    def reserve(
        self,
        artifact_key: str,
        lease_ms: int,
        apply_seq: int,
        grace_ms: int,
        expected_quota_version: int,
        apply_ordinal: int | None = None,
    ) -> dict[str, Any]:
        response = self.call(
            {
                "op": "reserve",
                "artifact_key": artifact_key,
                "lease_ms": lease_ms,
                "apply_seq": apply_seq,
                "grace_ms": grace_ms,
                "expected_quota_version": expected_quota_version,
                "apply_ordinal": apply_seq if apply_ordinal is None else apply_ordinal,
            },
            ("reservation_id", "slot_held", "grace_deadline_ms", "quota_version", "watermark"),
        )
        return response

    def commit(self, reservation_id: str) -> dict[str, Any]:
        return self.call(
            {"op": "commit", "reservation_id": reservation_id},
            ("lease_id", "artifact_key", "digest", "quota_version", "watermark"),
        )

    def rollback(self, reservation_id: str, reason: str) -> dict[str, Any]:
        return self.call(
            {"op": "rollback", "reservation_id": reservation_id, "reason": reason},
            ("rolled_back", "watermark"),
        )

    def read(
        self, artifact_key: str, lease_ms: int, apply_seq: int, apply_ordinal: int | None = None
    ) -> dict[str, Any]:
        return self.call(
            {
                "op": "read",
                "artifact_key": artifact_key,
                "lease_ms": lease_ms,
                "apply_seq": apply_seq,
                "apply_ordinal": apply_seq if apply_ordinal is None else apply_ordinal,
            },
            ("lease_id", "artifact_key", "digest", "quota_version", "watermark"),
        )

    def release(self, lease_id: str) -> dict[str, Any]:
        return self.call({"op": "release", "lease_id": lease_id}, ("lease_id", "released", "watermark"))

    def release_artifact(self, artifact_key: str) -> dict[str, Any]:
        return self.call(
            {"op": "release_artifact", "artifact_key": artifact_key},
            ("released_lease_ids", "freed_slot", "watermark"),
        )

    def set_quota(
        self,
        quota_version: int,
        capacity_slots: int | None,
        lease_limit: int | None,
        policy: str,
    ) -> dict[str, Any]:
        return self.call(
            {
                "op": "set_quota",
                "quota_version": quota_version,
                "capacity_slots": capacity_slots,
                "lease_limit": lease_limit,
                "policy": policy,
            },
            (
                "old_quota_version",
                "quota_version",
                "lease_dispositions",
                "reservation_dispositions",
                "watermark",
            ),
        )

    def reconcile(self, max_committed_seq: int, max_committed_ordinal: int | None = None) -> dict[str, Any]:
        return self.call(
            {
                "op": "reconcile",
                "max_committed_seq": max_committed_seq,
                "max_committed_ordinal": max_committed_seq
                if max_committed_ordinal is None
                else max_committed_ordinal,
            },
            ("unknown_reservations", "unknown_leases", "watermark"),
        )

    def restore(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return self.call({"op": "restore", "snapshot": snapshot}, ("restored", "watermark"))

    def shutdown(self) -> None:
        try:
            self.transport.request({"op": "shutdown"}, self.timeout_ms)
        except CacheLeaseError:
            pass


@dataclass
class NodeProcess:
    spec: NodeSpec
    process: subprocess.Popen[bytes]
    log_path: Path
    ready_path: Path


class NodePool:
    def __init__(self, config: Config, work_dir: str | Path, *, start_timeout_ms: int | None = None) -> None:
        self.config = config
        self.work_dir = Path(work_dir).resolve()
        self.start_timeout_ms = start_timeout_ms or config.runtime.handshake_timeout_ms
        self.processes: dict[str, NodeProcess] = {}
        self.clients: dict[str, NodeClient] = {}
        self.lock = threading.RLock()
        self._started = False

    def start(self) -> "NodePool":
        self.work_dir.mkdir(parents=True, exist_ok=True)
        root = Path(__file__).resolve().parent.parent
        env = dict(__import__("os").environ)
        env["PYTHONPATH"] = str(root) + (__import__("os").pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self._wait_for_ports()
        for spec in self.config.nodes:
            ready_path = self.work_dir / f"{spec.node_id}.ready.json"
            log_path = self.work_dir / f"{spec.node_id}.node.log"
            if ready_path.exists():
                ready_path.unlink()
            log_file = log_path.open("ab", buffering=0)
            command = [
                sys.executable,
                "-m",
                "cachelease.node",
                "--config",
                str(Path(self.config.config_path).resolve()),
                "--node-id",
                spec.node_id,
                "--parent-pid",
                str(__import__("os").getpid()),
                "--ready-file",
                str(ready_path),
            ]
            kwargs: dict[str, Any] = {
                "cwd": str(root),
                "env": env,
                "stdin": subprocess.DEVNULL,
                "stdout": log_file,
                "stderr": subprocess.STDOUT,
            }
            if __import__("os").name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(command, **kwargs)
            log_file.close()
            self.processes[spec.node_id] = NodeProcess(spec, process, log_path, ready_path)
            self.clients[spec.node_id] = NodeClient(spec, self.config.runtime.reply_timeout_ms)
        self._started = True
        deadline = mono_ms() + self.start_timeout_ms
        for spec in self.config.nodes:
            client = self.clients[spec.node_id]
            process = self.processes[spec.node_id]
            while mono_ms() < deadline:
                if process.process.poll() is not None:
                    detail = process.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                    raise HandshakeError(
                        f"node process exited with code {process.process.returncode}: {detail}",
                        spec.node_id,
                    )
                if process.ready_path.exists():
                    break
                time.sleep(0.01)
            try:
                client.handshake(deadline)
            except HandshakeError:
                self.stop()
                raise
        return self

    def _wait_for_ports(self) -> None:
        deadline = mono_ms() + min(self.start_timeout_ms, 3000)
        for spec in self.config.nodes:
            while True:
                if _loopback_port_available(spec.host, spec.port):
                    break
                if mono_ms() >= deadline:
                    raise HandshakeError(
                        "loopback port is already occupied before node startup",
                        f"{spec.host}:{spec.port}",
                    )
                time.sleep(0.02)

    def get(self, node_id: str) -> NodeClient:
        return self.clients[node_id]

    def snapshot(self, node_id: str) -> dict[str, Any]:
        with self.lock:
            return self.clients[node_id].status_snapshot

    def refresh(self, node_id: str) -> dict[str, Any]:
        with self.lock:
            return self.clients[node_id].status()

    def refresh_all(self) -> dict[str, dict[str, Any]]:
        return {spec.node_id: self.refresh(spec.node_id) for spec in self.config.nodes}

    def stop(self) -> None:
        for client in self.clients.values():
            try:
                client.shutdown()
            except Exception:
                pass
        for proc in self.processes.values():
            try:
                proc.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.process.terminate()
                try:
                    proc.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.process.kill()
                    proc.process.wait(timeout=1.0)
        self.clients.clear()
        self.processes.clear()
        self._started = False

    def __enter__(self) -> "NodePool":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def _loopback_port_available(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.05)
        if probe.connect_ex((host, port)) == 0:
            return False
    finally:
        probe.close()
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()
