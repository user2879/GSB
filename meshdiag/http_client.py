from __future__ import annotations

import socket
import time
from dataclasses import dataclass

from .models import Instance


@dataclass(frozen=True)
class AttemptResponse:
    outcome: str
    reason: str
    status_code: int | None
    detail: str


class LocalHttpClient:
    """A minimal HTTP/1.1 client whose timeout is enforced by the socket."""

    def __init__(self, timeout_ms: int) -> None:
        self.timeout_ms = timeout_ms

    def attempt(
        self,
        instance: Instance,
        method: str,
        path: str,
        attempt_no: int,
    ) -> AttemptResponse:
        timeout_seconds = self.timeout_ms / 1000.0
        started = time.monotonic_ns()
        sock: socket.socket | None = None
        try:
            sock = socket.create_connection(
                (instance.host, instance.port), timeout=timeout_seconds
            )
            sock.settimeout(timeout_seconds)
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            safe_path = path if path.startswith("/") else "/"
            request = (
                f"{method} {safe_path} HTTP/1.1\r\n"
                f"Host: {instance.host}:{instance.port}\r\n"
                f"X-Mesh-Attempt: {attempt_no}\r\n"
                "X-Mesh-Instance: " + instance.instance_id + "\r\n"
                "Connection: close\r\n"
                "User-Agent: meshdiag/1\r\n"
                "\r\n"
            ).encode("ascii")
            sock.sendall(request)
            chunks = []
            total = 0
            while total < 1_048_576:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            raw = b"".join(chunks)
            if not raw:
                return AttemptResponse("failure", "empty_response", None, "上游关闭连接且无响应")
            header_lines = raw.split(b"\r\n", 1)[0].decode("iso-8859-1", errors="replace")
            parts = header_lines.split(" ", 2)
            if len(parts) < 2 or not parts[0].startswith("HTTP/"):
                return AttemptResponse(
                    "failure", "invalid_response", None, header_lines[:200]
                )
            try:
                status = int(parts[1])
            except ValueError:
                return AttemptResponse(
                    "failure", "invalid_status", None, header_lines[:200]
                )
            elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
            if 200 <= status < 300:
                return AttemptResponse("success", "ok", status, f"{elapsed_ms}ms")
            return AttemptResponse("failure", f"http_{status}", status, header_lines[:200])
        except (TimeoutError, socket.timeout):
            return AttemptResponse(
                "timeout",
                "socket_timeout",
                None,
                f"单次尝试套接字超时 {self.timeout_ms}ms",
            )
        except OSError as exc:
            return AttemptResponse("failure", "socket_error", None, str(exc))
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
