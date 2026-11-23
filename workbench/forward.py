"""Real loopback forwarding to upstream services over sockets.

Timeouts are produced by real socket timeouts (socket.settimeout /
create_connection timeout), never by artificial sleeps on the gateway side.
"""

import hashlib
import socket
from dataclasses import dataclass

from . import E_RECEIPT, E_TIMEOUT, E_UPSTREAM, OK

MAX_LINE = 4096


@dataclass(frozen=True)
class ForwardResult:
    code: str        # OK | E_UPSTREAM | E_TIMEOUT | E_RECEIPT
    summary: str     # canonical receipt summary, or "none"
    detail: str      # human-readable diagnostic


def build_request_line(seq, service_id, client_key):
    return f"REQ {seq} {service_id} {client_key}\n"


def forward_request(host, port, seq, service_id, client_key,
                    timeout_s=2.0):
    line = build_request_line(seq, service_id, client_key)
    try:
        conn = socket.create_connection((host, port), timeout=timeout_s)
    except OSError as exc:
        return ForwardResult(E_UPSTREAM, "none",
                             f"connect failed: {exc}")
    try:
        conn.settimeout(timeout_s)
        conn.sendall(line.encode("utf-8"))
        fh = conn.makefile("rb")
        try:
            reply = fh.readline(MAX_LINE)
        except (socket.timeout, TimeoutError):
            return ForwardResult(E_TIMEOUT, "none",
                                 f"handshake timeout after {timeout_s}s")
        except OSError as exc:
            return ForwardResult(E_UPSTREAM, "none",
                                 f"read failed: {exc}")
        if not reply:
            return ForwardResult(E_UPSTREAM, "none",
                                 "upstream closed connection")
        try:
            text = reply.decode("utf-8").rstrip("\n")
        except UnicodeDecodeError:
            return ForwardResult(E_RECEIPT, "none",
                                 "receipt is not valid utf-8")
        parts = text.split(" ")
        if len(parts) != 4 or parts[0] != "RCPT":
            return ForwardResult(E_RECEIPT, "none",
                                 f"malformed receipt: {text!r}")
        _, rseq, rsvc, rcksum = parts
        try:
            rseq_int = int(rseq)
        except ValueError:
            return ForwardResult(E_RECEIPT, "none",
                                 f"receipt seq not an integer: {rseq!r}")
        if rseq_int != seq or rsvc != service_id:
            return ForwardResult(
                E_RECEIPT, "none",
                f"receipt mismatch: got seq={rseq_int} svc={rsvc!r}, "
                f"expected seq={seq} svc={service_id!r}")
        expected = hashlib.sha256(
            line.rstrip("\n").encode("utf-8")).hexdigest()[:16]
        if rcksum != expected:
            return ForwardResult(E_RECEIPT, "none",
                                 "receipt checksum mismatch")
        summary = f"RCPT seq={seq} svc={service_id} ck={rcksum}"
        return ForwardResult(OK, summary, "receipt verified")
    finally:
        try:
            conn.close()
        except OSError:
            pass
