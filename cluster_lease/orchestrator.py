"""Local subprocess orchestration and newline-delimited JSON protocol client."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from .canonical import PROTOCOL_VERSION, canonical_json, sha256_value
from .errors import LeaseError
from .models import ClusterSpec

MAX_LINE_BYTES = 1_048_576

REQUIRED_REPLY_FIELDS = (
    "ok",
    "type",
    "event_seq",
    "cluster_id",
    "holder_id",
    "event_type",
    "decision",
    "reason",
    "active_holders",
    "used_capacity",
    "capacity",
    "watermark_fixed",
    "watermark_scale",
)
REQUIRED_ACTIVE_FIELDS = (
    "expires_ms",
    "holder_id",
    "last_event_seq",
    "last_renewed_ms",
    "priority",
    "renewal_count",
    "started_ms",
    "state",
)
REQUIRED_TRANSITION_FIELDS = (
    "active_count",
    "after_state",
    "before_state",
    "event_seq",
    "holder_id",
    "reason",
    "timestamp_ms",
)


class JsonLineClient:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buffer = bytearray()

    def send(self, value: dict[str, Any]) -> None:
        self.sock.sendall((canonical_json(value) + "\n").encode("utf-8"))

    def recv(self) -> dict[str, Any]:
        while b"\n" not in self.buffer:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise LeaseError("E_REPLY", "service closed the connection before a complete reply")
            self.buffer.extend(chunk)
            if len(self.buffer) > MAX_LINE_BYTES:
                raise LeaseError("E_REPLY", "reply line exceeds maximum length")
        raw, _, remainder = self.buffer.partition(b"\n")
        self.buffer = bytearray(remainder)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LeaseError("E_REPLY", f"invalid JSON reply: {exc}") from None
        if not isinstance(value, dict):
            raise LeaseError("E_REPLY", "reply must be a JSON object")
        return value


class ServiceProcess:
    """A child process exposing one cluster arbitration service."""

    def __init__(
        self,
        spec: ClusterSpec,
        priorities: dict[str, int],
        *,
        parameter_version: int,
        runtime_dir: str,
        clock_mode: str = "hybrid",
        startup_timeout_ms: int = 5000,
        handshake_timeout_ms: int = 2000,
        log_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.spec = spec
        self.priorities = dict(priorities)
        self.parameter_version = parameter_version
        self.runtime_dir = Path(runtime_dir)
        self.clock_mode = clock_mode
        self.startup_timeout_ms = startup_timeout_ms
        self.handshake_timeout_ms = handshake_timeout_ms
        self.log_callback = log_callback
        self.process: subprocess.Popen[bytes] | None = None
        self.client: JsonLineClient | None = None
        self.stderr_lines: deque[str] = deque(maxlen=200)
        self._reader: threading.Thread | None = None
        self._stderr_lock = threading.Lock()
        safe_cluster_name = sha256_value(spec.cluster_id)[:16]
        self.config_path = self.runtime_dir / f"service-{safe_cluster_name}.json"

    def _dispatch_log_line(self, line: str) -> None:
        if not line:
            return
        with self._stderr_lock:
            self.stderr_lines.append(line)
        if self.log_callback is None:
            return
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            parsed = {"code": "E_SERVICE_LOG", "line": line, "ok": False}
        if isinstance(parsed, dict):
            self.log_callback(self.spec.cluster_id, parsed)

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stderr is not None
        for raw in iter(process.stderr.readline, b""):
            self._dispatch_log_line(raw.decode("utf-8", errors="replace").strip())

    def _service_error(self) -> LeaseError | None:
        with self._stderr_lock:
            lines = list(self.stderr_lines)
        for line in reversed(lines):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and parsed.get("code") == "E_PORT":
                return LeaseError(
                    code="E_PORT",
                    message=parsed.get("message", f"port {self.spec.port} is already in use"),
                    field="port",
                    details={"cluster_id": self.spec.cluster_id, "port": self.spec.port},
                )
        return None

    def start(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if os.name != "nt":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((self.spec.host, self.spec.port))
        except OSError as exc:
            raise LeaseError(
                "E_PORT",
                f"cannot bind {self.spec.host}:{self.spec.port}: {exc}",
                field="port",
                details={"cluster_id": self.spec.cluster_id, "port": self.spec.port},
            ) from None
        finally:
            probe.close()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "capacity": self.spec.capacity,
            "cluster_id": self.spec.cluster_id,
            "clock_mode": self.clock_mode,
            "host": self.spec.host,
            "lease_duration_ms": self.spec.lease_duration_ms,
            "parameter_version": self.parameter_version,
            "parent_pid": os.getpid(),
            "port": self.spec.port,
            "priorities": {key: self.priorities[key] for key in sorted(self.priorities)},
            "renewal_interval_ms": self.spec.renewal_interval_ms,
        }
        self.config_path.write_text(canonical_json(config) + "\n", encoding="utf-8")
        repository_root = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(repository_root)
            if not existing_pythonpath
            else str(repository_root) + os.pathsep + existing_pythonpath
        )
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = subprocess.Popen(
            [sys.executable, "-m", "cluster_lease.service", "--config", str(self.config_path)],
            cwd=str(repository_root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        self._reader = threading.Thread(
            target=self._drain_stderr,
            args=(self.process,),
            name=f"stderr-{self.spec.cluster_id}",
            daemon=True,
        )
        self._reader.start()

        deadline = time.monotonic() + self.startup_timeout_ms / 1000.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                service_error = self._service_error()
                if service_error is not None:
                    raise service_error
                raise LeaseError(
                    "E_HANDSHAKE",
                    f"service for {self.spec.cluster_id} exited before handshake",
                    field="handshake",
                    details={"exit_code": self.process.returncode, "cluster_id": self.spec.cluster_id},
                )
            try:
                sock = socket.create_connection(
                    (self.spec.host, self.spec.port), timeout=min(0.25, self.startup_timeout_ms / 1000.0)
                )
            except OSError as exc:
                last_error = exc
                time.sleep(0.03)
                continue
            try:
                self._handshake(sock)
                return
            except LeaseError:
                sock.close()
                raise
            except Exception:
                sock.close()
                raise
        service_error = self._service_error()
        if service_error is not None:
            raise service_error
        raise LeaseError(
            "E_HANDSHAKE",
            f"timed out waiting for service handshake for {self.spec.cluster_id}",
            field="handshake",
            details={
                "cluster_id": self.spec.cluster_id,
                "port": self.spec.port,
                "last_error": str(last_error) if last_error else None,
            },
        )

    def _handshake(self, sock: socket.socket) -> None:
        sock.settimeout(self.handshake_timeout_ms / 1000.0)
        client = JsonLineClient(sock)
        try:
            client.send(
                {
                    "cluster_id": self.spec.cluster_id,
                    "protocol": PROTOCOL_VERSION,
                    "type": "hello",
                }
            )
            reply = client.recv()
        except (socket.timeout, TimeoutError):
            raise LeaseError(
                "E_HANDSHAKE",
                f"handshake socket timeout after {self.handshake_timeout_ms} ms",
                field="handshake",
                details={"cluster_id": self.spec.cluster_id, "port": self.spec.port},
            ) from None
        required = ("ok", "type", "protocol", "cluster_id")
        for field in required:
            if field not in reply:
                raise LeaseError(
                    "E_REPLY",
                    f"handshake reply is missing required field: {field}",
                    field=f"handshake.{field}",
                    details={"cluster_id": self.spec.cluster_id},
                )
        if reply.get("type") != "hello" or reply.get("ok") is not True:
            raise LeaseError(
                "E_REPLY",
                "handshake reply has an invalid type or ok value",
                field="handshake",
                details={"cluster_id": self.spec.cluster_id, "reply": reply},
            )
        if reply.get("protocol") != PROTOCOL_VERSION:
            raise LeaseError(
                "E_REPLY",
                f"unsupported protocol version: {reply.get('protocol')}",
                field="handshake.protocol",
            )
        self.client = client

    def request(self, value: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            raise LeaseError("E_HANDSHAKE", "service is not connected", field="handshake")
        self.client.send(value)
        try:
            reply = self.client.recv()
        except (socket.timeout, TimeoutError):
            raise LeaseError(
                "E_REPLY",
                "socket timeout while waiting for a decision reply",
                field="reply",
                details={"cluster_id": self.spec.cluster_id},
            ) from None
        if reply.get("ok") is not True:
            if "code" in reply:
                raise LeaseError(
                    str(reply.get("code")),
                    str(reply.get("message", "service rejected the request")),
                    field=str(reply.get("field")) if reply.get("field") else None,
                )
            raise LeaseError("E_REPLY", "service reply has ok=false", field="reply")
        for field in REQUIRED_REPLY_FIELDS:
            if field not in reply:
                raise LeaseError(
                    "E_REPLY",
                    f"decision reply is missing required field: {field}",
                    field=f"reply.{field}",
                    details={"cluster_id": self.spec.cluster_id},
                )
        if not isinstance(reply["active_holders"], list):
            raise LeaseError("E_REPLY", "active_holders must be a list", field="reply.active_holders")
        for index, holder in enumerate(reply["active_holders"]):
            if not isinstance(holder, dict):
                raise LeaseError(
                    "E_REPLY",
                    "active holder entry must be an object",
                    field=f"reply.active_holders[{index}]",
                )
            for field in REQUIRED_ACTIVE_FIELDS:
                if field not in holder:
                    raise LeaseError(
                        "E_REPLY",
                        f"active holder entry is missing required field: {field}",
                        field=f"reply.active_holders[{index}].{field}",
                    )
        transitions = reply.get("transitions")
        if not isinstance(transitions, list):
            raise LeaseError("E_REPLY", "transitions must be a list", field="reply.transitions")
        for index, transition in enumerate(transitions):
            if not isinstance(transition, dict):
                raise LeaseError(
                    "E_REPLY",
                    "transition entry must be an object",
                    field=f"reply.transitions[{index}]",
                )
            for field in REQUIRED_TRANSITION_FIELDS:
                if field not in transition:
                    raise LeaseError(
                        "E_REPLY",
                        f"transition entry is missing required field: {field}",
                        field=f"reply.transitions[{index}].{field}",
                    )
        return reply

    def restore(self, leases: list[dict[str, Any]], last_event_seq: int, last_timestamp_ms: int) -> dict[str, Any]:
        if self.client is None:
            raise LeaseError("E_HANDSHAKE", "service is not connected", field="handshake")
        self.client.send(
            {
                "last_event_seq": last_event_seq,
                "last_timestamp_ms": last_timestamp_ms,
                "leases": leases,
                "type": "restore",
            }
        )
        reply = self.client.recv()
        for field in ("ok", "type", "cluster_id", "active_holders"):
            if field not in reply:
                raise LeaseError(
                    "E_REPLY",
                    f"restore reply is missing required field: {field}",
                    field=f"reply.{field}",
                )
        if reply.get("ok") is not True or reply.get("type") != "restored":
            raise LeaseError("E_REPLY", "restore reply has an invalid type or ok value", field="reply")
        return reply

    def status(self, timestamp_ms: int | None = None) -> dict[str, Any]:
        """Ask the service to evaluate real-clock expiry without a lease event."""

        if self.client is None:
            raise LeaseError("E_HANDSHAKE", "service is not connected", field="handshake")
        request: dict[str, Any] = {"type": "status"}
        if timestamp_ms is not None:
            request["timestamp_ms"] = timestamp_ms
        self.client.send(request)
        reply = self.client.recv()
        for field in ("ok", "type", "cluster_id", "active_holders", "used_capacity"):
            if field not in reply:
                raise LeaseError(
                    "E_REPLY",
                    f"status reply is missing required field: {field}",
                    field=f"reply.{field}",
                )
        if reply.get("ok") is not True or reply.get("type") != "status":
            raise LeaseError("E_REPLY", "status reply has an invalid type or ok value", field="reply")
        return reply

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        if self.client is not None:
            try:
                self.client.send({"type": "shutdown"})
                self.client.sock.settimeout(1.0)
                try:
                    self.client.recv()
                except Exception:
                    pass
            except OSError:
                pass
            try:
                self.client.sock.close()
            except OSError:
                pass
            self.client = None
        try:
            process.wait(timeout=0.8)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        if self._reader is not None:
            self._reader.join(timeout=0.5)
        if process.stderr is not None:
            try:
                process.stderr.close()
            except OSError:
                pass
        self.process = None


class ServiceOrchestrator:
    def __init__(
        self,
        runtime_dir: str,
        *,
        clock_mode: str = "hybrid",
        startup_timeout_ms: int = 5000,
        handshake_timeout_ms: int = 2000,
        log_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.runtime_dir = runtime_dir
        self.clock_mode = clock_mode
        self.startup_timeout_ms = startup_timeout_ms
        self.handshake_timeout_ms = handshake_timeout_ms
        self.log_callback = log_callback
        self.services: dict[str, ServiceProcess] = {}

    def start(
        self,
        specs: list[ClusterSpec],
        priorities_by_cluster: dict[str, dict[str, int]],
        versions: dict[str, int],
    ) -> None:
        self.stop()
        started: list[ServiceProcess] = []
        try:
            for spec in sorted(specs, key=lambda item: item.cluster_id):
                service = ServiceProcess(
                    spec,
                    priorities_by_cluster[spec.cluster_id],
                    parameter_version=versions[spec.cluster_id],
                    runtime_dir=self.runtime_dir,
                    clock_mode=self.clock_mode,
                    startup_timeout_ms=self.startup_timeout_ms,
                    handshake_timeout_ms=self.handshake_timeout_ms,
                    log_callback=self.log_callback,
                )
                try:
                    service.start()
                except Exception:
                    service.close()
                    raise
                started.append(service)
                self.services[spec.cluster_id] = service
        except Exception:
            for service in reversed(started):
                service.close()
            self.services.clear()
            raise

    def get(self, cluster_id: str) -> ServiceProcess:
        try:
            return self.services[cluster_id]
        except KeyError:
            raise LeaseError(
                "E_STATE",
                f"no arbitration service is running for cluster {cluster_id}",
                field="cluster_id",
            ) from None

    def stop(self) -> None:
        for cluster_id in sorted(self.services, reverse=True):
            self.services[cluster_id].close()
        self.services.clear()
