"""Platform-side lifecycle management for node subprocesses and TCP client."""
from __future__ import annotations

import socket
import subprocess
import sys
from typing import Any

from .errors import BusinessError, ErrorCode
from .logging_utils import StructuredLogger
from .models import NodeSpec, Receipt
from .protocol import recv_line, send_line
from .util import now_ms

BOOT_GRACE_MS = 4000


class NodeProcess:
    """Start/stop a node subprocess and perform handshake/dispatch calls."""

    def __init__(self, spec: NodeSpec, logger: StructuredLogger) -> None:
        self.spec = spec
        self.logger = logger
        self.process: subprocess.Popen | None = None

    def _preflight_port_free(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.2)
        try:
            probe.connect((self.spec.host, self.spec.port))
        except OSError:
            probe.close()
            return
        probe.close()
        self.logger.error(ErrorCode.E_PORT.value, "port_already_in_use",
                          node_id=self.spec.node_id, host=self.spec.host,
                          port=self.spec.port)
        raise BusinessError(
            ErrorCode.E_PORT,
            f"port {self.spec.host}:{self.spec.port} is already occupied for node "
            f"{self.spec.node_id}",
            [{"node_id": self.spec.node_id, "host": self.spec.host,
              "port": self.spec.port, "reason": "port_in_use"}],
        )

    def start(self) -> None:
        self._preflight_port_free()
        command = [
            sys.executable,
            "-m",
            "configmesh.protocol",
            "--node-id",
            self.spec.node_id,
            "--host",
            self.spec.host,
            "--port",
            str(self.spec.port),
            "--bias",
            str(self.spec.bias),
        ]
        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
        except OSError as exc:
            self.logger.error(ErrorCode.E_PORT.value, "node_spawn_failed",
                              node_id=self.spec.node_id, detail=str(exc))
            raise BusinessError(ErrorCode.E_PORT,
                                f"unable to spawn node {self.spec.node_id}",
                                [{"node_id": self.spec.node_id, "detail": str(exc)}])
        self._wait_until_listening()

    def _wait_until_listening(self) -> None:
        deadline = now_ms() + BOOT_GRACE_MS
        last_error: Exception | None = None
        while now_ms() < deadline:
            if self.process and self.process.poll() is not None:
                stderr = ""
                if self.process.stderr:
                    stderr = self.process.stderr.read().decode("utf-8", "replace")[:2000]
                self.logger.error(ErrorCode.E_PORT.value, "node_process_exited",
                                  node_id=self.spec.node_id, returncode=self.process.returncode,
                                  stderr=stderr)
                raise BusinessError(
                    ErrorCode.E_PORT,
                    f"node {self.spec.node_id} exited before opening port",
                    [{"node_id": self.spec.node_id, "port": self.spec.port,
                       "returncode": self.process.returncode, "stderr": stderr}],
                )
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.settimeout(0.2)
            try:
                probe.connect((self.spec.host, self.spec.port))
                probe.close()
                return
            except OSError as exc:
                last_error = exc
            finally:
                try:
                    probe.close()
                except OSError:
                    pass
        self.logger.error(ErrorCode.E_PORT.value, "node_port_unreachable",
                          node_id=self.spec.node_id, host=self.spec.host,
                          port=self.spec.port, detail=str(last_error))
        raise BusinessError(
            ErrorCode.E_PORT,
            f"node {self.spec.node_id} did not open {self.spec.host}:{self.spec.port}",
            [{"node_id": self.spec.node_id, "host": self.spec.host,
              "port": self.spec.port, "detail": str(last_error)}],
        )

    def handshake(self) -> dict[str, Any]:
        with self._connect(self.spec.timeout_ms) as sock:
            sock.settimeout(self.spec.timeout_ms / 1000.0)
            try:
                send_line(sock, {"type": "hello", "node_id": self.spec.node_id})
                reply = recv_line(sock)
            except socket.timeout as exc:
                self.logger.error(ErrorCode.E_HANDSHAKE.value, "handshake_timeout",
                                  node_id=self.spec.node_id, timeout_ms=self.spec.timeout_ms)
                raise BusinessError(ErrorCode.E_HANDSHAKE,
                                    f"handshake timeout for {self.spec.node_id}",
                                    [{"node_id": self.spec.node_id}]) from exc
            if reply.get("type") != "hello_ack" or reply.get("node_id") != self.spec.node_id:
                self.logger.error(ErrorCode.E_HANDSHAKE.value, "handshake_bad_ack",
                                  node_id=self.spec.node_id, reply=reply)
                raise BusinessError(ErrorCode.E_HANDSHAKE,
                                    f"invalid handshake ack from {self.spec.node_id}",
                                    [{"node_id": self.spec.node_id, "reply": reply}])
            self.logger.info("handshake_ok", node_id=self.spec.node_id)
            return reply

    def dispatch(self, message: dict[str, Any], attempt: int) -> Receipt:
        with self._connect(self.spec.timeout_ms) as sock:
            sock.settimeout(self.spec.timeout_ms / 1000.0)
            try:
                send_line(sock, message)
                reply = recv_line(sock)
            except socket.timeout as exc:
                self.logger.error(ErrorCode.E_HANDSHAKE.value, "dispatch_timeout",
                                  node_id=self.spec.node_id, attempt=attempt,
                                  timeout_ms=self.spec.timeout_ms)
                raise BusinessError(ErrorCode.E_HANDSHAKE,
                                    f"dispatch timeout for {self.spec.node_id} attempt {attempt}",
                                    [{"node_id": self.spec.node_id, "attempt": attempt}]) from exc
            except ConnectionError as exc:
                self.logger.error(ErrorCode.E_REPLY.value, "dispatch_connection_broken",
                                  node_id=self.spec.node_id, attempt=attempt, detail=str(exc))
                raise BusinessError(ErrorCode.E_REPLY,
                                    f"connection broken waiting receipt on {self.spec.node_id}",
                                    [{"node_id": self.spec.node_id, "attempt": attempt}]) from exc
        if reply.get("type") == "error":
            self.logger.error(ErrorCode.E_REPLY.value, "node_error_envelope",
                              node_id=self.spec.node_id, attempt=attempt, reply=reply)
            raise BusinessError(ErrorCode.E_REPLY,
                                f"node {self.spec.node_id} returned error envelope",
                                [{"node_id": self.spec.node_id, "reply": reply}])
        missing = [field for field in Receipt.required_fields() if field not in reply]
        if missing:
            self.logger.error(ErrorCode.E_REPLY.value, "receipt_missing_fields",
                              node_id=self.spec.node_id, attempt=attempt, missing=missing)
            raise BusinessError(ErrorCode.E_REPLY,
                                f"receipt from {self.spec.node_id} missing fields",
                                [{"node_id": self.spec.node_id, "missing": missing}])
        receipt = Receipt.from_dict(reply)
        if receipt.node_id != self.spec.node_id:
            raise BusinessError(ErrorCode.E_REPLY, "receipt node_id mismatch",
                                [{"expected": self.spec.node_id, "actual": receipt.node_id}])
        self.logger.info("receipt_received", node_id=self.spec.node_id,
                         version=receipt.version, attempt=attempt,
                         shard_index=receipt.shard_index, compute_ms=receipt.compute_ms)
        return receipt

    def _connect(self, timeout_ms: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_ms / 1000.0)
        try:
            sock.connect((self.spec.host, self.spec.port))
        except OSError as exc:
            self.logger.error(ErrorCode.E_PORT.value, "connect_failed",
                              node_id=self.spec.node_id, host=self.spec.host,
                              port=self.spec.port, detail=str(exc))
            sock.close()
            raise BusinessError(ErrorCode.E_PORT,
                                f"cannot connect to node {self.spec.node_id}",
                                [{"node_id": self.spec.node_id,
                                  "host": self.spec.host, "port": self.spec.port,
                                  "detail": str(exc)}]) from exc
        return sock

    def stop(self) -> None:
        if not self.process:
            return
        process = self.process
        try:
            with self._connect(min(self.spec.timeout_ms, 1000)) as sock:
                sock.settimeout(1.0)
                send_line(sock, {"type": "shutdown", "node_id": self.spec.node_id})
                recv_line(sock)
        except (BusinessError, OSError, ConnectionError):
            pass
        try:
            process.wait(timeout=3)
        except (subprocess.TimeoutExpired, OSError):
            process.terminate()
            try:
                process.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                process.kill()
                process.wait(timeout=2)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self.process = None
