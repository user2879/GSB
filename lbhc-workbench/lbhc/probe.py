"""Real-socket TCP/HTTP probes.

Success/failure is decided exclusively by real socket behaviour:
  * timeout        -> socket.settimeout() expiring on connect/recv
  * conn_refused   -> a real connect() error (ECONNREFUSED)
  * digest_mismatch-> response bytes whose sha256 differs from expected
"""

from __future__ import annotations

import hashlib
import socket

from . import health


def _expected_hex(expect_body: str) -> str:
    return hashlib.sha256(expect_body.encode()).hexdigest()


def probe_tcp(host, port, timeout_ms, expect_body):
    """Connect and read a banner; verify its digest."""
    timeout_s = timeout_ms / 1000.0
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as s:
            s.settimeout(timeout_s)
            data = s.recv(256)
    except socket.timeout:
        return ("failure", health.REASON_TIMEOUT,
                f"no banner within {timeout_ms}ms")
    except (ConnectionRefusedError, ConnectionResetError) as exc:
        return ("failure", health.REASON_CONN_REFUSED, str(exc))
    except OSError as exc:
        return ("failure", "io_error", str(exc))
    if not data:
        return ("failure", health.REASON_CONN_REFUSED,
                "peer closed without data")
    if hashlib.sha256(data).hexdigest() != _expected_hex(expect_body):
        return ("failure", health.REASON_DIGEST,
                f"banner digest mismatch: {data!r}")
    return ("success", "", "")


def probe_http(host, port, timeout_ms, expect_body):
    """Send GET /health and verify status + body digest."""
    timeout_s = timeout_ms / 1000.0
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as s:
            s.settimeout(timeout_s)
            s.sendall(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n"
                      b"Connection: close\r\n\r\n")
            chunks = []
            while True:
                try:
                    chunk = s.recv(4096)
                except socket.timeout:
                    return ("failure", health.REASON_TIMEOUT,
                            f"no HTTP response within {timeout_ms}ms")
                if not chunk:
                    break
                chunks.append(chunk)
    except socket.timeout:
        return ("failure", health.REASON_TIMEOUT,
                f"connect exceeded {timeout_ms}ms")
    except (ConnectionRefusedError, ConnectionResetError) as exc:
        return ("failure", health.REASON_CONN_REFUSED, str(exc))
    except OSError as exc:
        return ("failure", "io_error", str(exc))
    raw = b"".join(chunks)
    head, sep, body = raw.partition(b"\r\n\r\n")
    if not sep:
        return ("failure", health.REASON_DIGEST, "malformed HTTP response")
    status_line = head.split(b"\r\n", 1)[0]
    parts = status_line.split(b" ", 2)
    if len(parts) < 2 or parts[1] != b"200":
        return ("failure", health.REASON_DIGEST,
                f"unexpected status: {status_line!r}")
    if hashlib.sha256(body).hexdigest() != _expected_hex(expect_body):
        return ("failure", health.REASON_DIGEST,
                f"body digest mismatch: {body!r}")
    return ("success", "", "")


def run_probe(protocol, host, port, timeout_ms, expect_body):
    if protocol == "tcp":
        return probe_tcp(host, port, timeout_ms, expect_body)
    return probe_http(host, port, timeout_ms, expect_body)
