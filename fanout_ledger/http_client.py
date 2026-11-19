"""Minimal loopback HTTP/1.1 client built directly on sockets.

Using a raw socket (instead of ``requests``) makes the timeout semantics
unambiguous:

* the **connect phase** gets a short OS connect timeout; a refused/closed
  loopback port surfaces as a genuine ``ConnectionRefusedError`` / OSError
  and is classified ``unreachable``;
* the **request phase** uses the configured per-attempt socket timeout, so a
  slow handler that never answers in time raises a genuine ``socket.timeout``
  and is classified ``timeout``.

No artificial sleeps synthesize either outcome; both come from the real OS
socket stack against the local mock service.
"""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass

from .config import NodeSpec

OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_UNREACHABLE = "unreachable"

CONNECT_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class AttemptResponse:
    outcome: str
    http_status: int | None
    note: str


def post_attempt(node: NodeSpec, attempt_seq: int, task_id: str, payload_bytes: int,
                 timeout_ms: int) -> AttemptResponse:
    payload = json.dumps(
        {"node_id": node.node_id, "task_id": task_id,
         "attempt_seq": attempt_seq, "payload_bytes": payload_bytes},
        sort_keys=True,
    ).encode("utf-8")
    request = (
        f"POST /deliver HTTP/1.1\r\nHost: {node.host}:{node.port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("utf-8") + payload

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Phase 1: real TCP connect with an OS-level connect timeout.
        sock.settimeout(CONNECT_TIMEOUT_SECONDS)
        try:
            sock.connect((node.host, node.port))
        except ConnectionRefusedError as exc:
            return AttemptResponse(OUTCOME_UNREACHABLE, None,
                                   f"connection refused: {exc}")
        except socket.timeout as exc:
            return AttemptResponse(OUTCOME_UNREACHABLE, None,
                                   f"connect timeout (peer unreachable): {exc}")
        except OSError as exc:
            return AttemptResponse(OUTCOME_UNREACHABLE, None,
                                   f"connection error: {exc}")

        # Phase 2: per-attempt timeout governs the actual request/response.
        sock.settimeout(timeout_ms / 1000.0)
        try:
            sock.sendall(request)
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        except socket.timeout as exc:  # genuine OS socket timeout
            return AttemptResponse(OUTCOME_TIMEOUT, None, f"socket timeout: {exc}")
        except OSError as exc:
            return AttemptResponse(OUTCOME_UNREACHABLE, None,
                                   f"connection error: {exc}")

        raw = b"".join(chunks)
        if not raw:
            return AttemptResponse(OUTCOME_UNREACHABLE, None,
                                   "empty response; peer closed")
        header_blob, _, body_blob = raw.partition(b"\r\n\r\n")
        status_line = header_blob.split(b"\r\n", 1)[0].decode("latin-1")
        parts = status_line.split(" ", 2)
        status_code = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
        try:
            parsed = json.loads(body_blob.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed = {}
        result = str(parsed.get("result", ""))
        if status_code == 200 and result != "failure":
            return AttemptResponse(OUTCOME_SUCCESS, status_code, result)
        return AttemptResponse(OUTCOME_FAILURE, status_code,
                               result or f"http {status_code}")
    finally:
        sock.close()
