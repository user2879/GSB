from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_text
from .errors import DomainError
from .models import ScriptMessage, Spec


MAX_LINE_BYTES = 1_048_576


def _recv_line(sock: socket.socket, buffer: bytearray) -> bytes | None:
    while b"\n" not in buffer:
        chunk = sock.recv(65536)
        if not chunk:
            return None
        buffer.extend(chunk)
        if len(buffer) > MAX_LINE_BYTES:
            raise DomainError("E_REPLY", "reply line exceeds maximum size", field_path="reply.line")
    line, _, rest = buffer.partition(b"\n")
    buffer[:] = rest
    return bytes(line)


class LoopbackService:
    def __init__(
        self,
        spec: Spec,
        cert_path: str | Path,
        handshake_path: str | Path,
        port: int,
        host: str = "127.0.0.1",
        drop_client_seq: int | None = None,
        omit_reply_field: str | None = None,
    ) -> None:
        self.spec = spec
        self.cert_path = str(Path(cert_path).resolve())
        self.handshake_path = str(Path(handshake_path).resolve())
        self.port = port
        self.host = host
        self.drop_client_seq = drop_client_seq
        self.omit_reply_field = omit_reply_field
        self.process: subprocess.Popen[str] | None = None
        self.socket: socket.socket | None = None
        self.buffer = bytearray()

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        package_root = str(Path(__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = package_root if not existing else package_root + os.pathsep + existing
        return env

    def start(self) -> None:
        command = [
            sys.executable,
            "-m",
            "tlschain.service",
            "--certs",
            self.cert_path,
            "--handshake",
            self.handshake_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.drop_client_seq is not None:
            command.extend(["--drop-client-seq", str(self.drop_client_seq)])
        if self.omit_reply_field is not None:
            command.extend(["--omit-reply-field", self.omit_reply_field])
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._environment(),
        )
        assert self.process.stdout is not None
        ready_line = self.process.stdout.readline()
        if not ready_line:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            self.stop()
            raise DomainError(
                "E_PORT",
                f"service failed to start: {stderr.strip() or 'no readiness line'}",
                field_path="socket.port",
            )
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            self.stop()
            raise DomainError("E_PORT", f"invalid service readiness line: {ready_line!r}", field_path="socket.port") from exc
        if ready.get("status") != "READY":
            self.stop()
            code = ready.get("code", "E_PORT")
            raise DomainError(
                code,
                ready.get("message", "service failed to bind"),
                cert_id=ready.get("cert_id"),
                field_path=ready.get("field", "socket.port"),
                message_seq=ready.get("message_seq"),
            )
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(max(0.05, self.spec.socket_timeout_ms / 1000.0))
        try:
            sock.connect((self.host, self.port))
        except OSError as exc:
            sock.close()
            self.stop()
            raise DomainError("E_PORT", f"cannot connect to loopback service: {exc}", field_path="socket.port") from exc
        self.socket = sock

    def exchange(self, request: ScriptMessage) -> list[dict[str, Any]]:
        if self.socket is None:
            raise DomainError("E_REPLY", "service socket is not connected", message_seq=request.seq, field_path="socket")
        body = {
            "op": "client_message",
            "client_seq": request.client_seq,
            "message": request.wire_dict(),
        }
        try:
            self.socket.sendall(canonical_text(body).encode("utf-8") + b"\n")
            line = _recv_line(self.socket, self.buffer)
        except socket.timeout as exc:
            raise DomainError(
                "E_TIMEOUT",
                f"handshake socket timed out after {self.spec.socket_timeout_ms} ms",
                cert_id=request.cert_ids[0] if request.cert_ids else None,
                field_path="socket.timeout",
                message_seq=request.seq,
                details={"client_seq": request.client_seq, "timeout_ms": self.spec.socket_timeout_ms},
            ) from exc
        except OSError as exc:
            raise DomainError(
                "E_REPLY",
                f"handshake socket failed: {exc}",
                cert_id=request.cert_ids[0] if request.cert_ids else None,
                field_path="socket",
                message_seq=request.seq,
            ) from exc
        if line is None:
            raise DomainError(
                "E_REPLY",
                "service closed the connection without a reply",
                cert_id=request.cert_ids[0] if request.cert_ids else None,
                field_path="reply.presence",
                message_seq=request.seq,
            )
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DomainError(
                "E_REPLY",
                "service reply is not valid JSON",
                cert_id=request.cert_ids[0] if request.cert_ids else None,
                field_path="reply.json",
                message_seq=request.seq,
            ) from exc
        if not isinstance(reply, dict):
            raise DomainError(
                "E_REPLY",
                "service reply must be an object",
                message_seq=request.seq,
                field_path="reply",
            )
        if reply.get("status") != "OK":
            raise DomainError(
                reply.get("code", "E_REPLY"),
                reply.get("message", "service returned an error"),
                cert_id=reply.get("cert_id"),
                field_path=reply.get("field", "reply.status"),
                message_seq=reply.get("message_seq", request.seq),
            )
        for field in ("client_seq", "request_seq", "replies"):
            if field not in reply:
                raise DomainError(
                    "E_REPLY",
                    f"service reply is missing field: {field}",
                    cert_id=request.cert_ids[0] if request.cert_ids else None,
                    field_path=f"reply.{field}",
                    message_seq=request.seq,
                )
        replies = reply["replies"]
        if not isinstance(replies, list):
            raise DomainError(
                "E_REPLY",
                "service reply field replies must be an array",
                message_seq=request.seq,
                field_path="reply.replies",
            )
        return replies

    def stop(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            finally:
                self.socket = None
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3.0)
            self.process = None

    def __enter__(self) -> LoopbackService:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


def wait_for_port_release(host: str, port: int, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((host, port))
            return
        except OSError:
            time.sleep(0.05)
        finally:
            probe.close()
