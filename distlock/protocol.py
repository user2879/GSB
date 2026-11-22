"""JSON-lines wire protocol over loopback TCP and the arbiter subprocess loop.

Wire messages are a single UTF-8 JSON object terminated by ``\\n``.

Client -> server::

    {"type": "HELLO", "domain_id": "...", "domain_version": 1, "domain_set_version": 3}
    {"type": "REQUEST", "seq": 7, "holder_id": "h1", "action": "ACQUIRE", "priority": 5}
    {"type": "PING"}

Server -> client::

    {"type": "HELLO_ACK", "domain_id": "...", "domain_version": 1, "pid": 1234}
    {"type": "RECEIPT", ...full receipt..., "transitions": [...]}
    {"type": "PONG"}
    {"type": "ERROR", "message": "..."}

Errors observed by the client are normalised to the required codes:
port unavailable -> E_PORT, handshake timeout -> E_HANDSHAKE,
missing/malformed receipt fields -> E_REPLY.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import asdict
from typing import Any

from .errors import HandshakeError, PortError, ReplyError

REQUIRED_RECEIPT_FIELDS = (
    "seq", "domain_id", "holder_id", "verdict", "remaining_ms",
)
HANDSHAKE_TIMEOUT_MS = 3_000
REQUEST_TIMEOUT_MS = 10_000
SO_BACKLOG = 16


def encode(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def send_line(sock: socket.socket, obj: dict[str, Any]) -> None:
    sock.sendall(encode(obj))


def recv_line(sock: socket.socket, timeout_ms: int | None = None) -> dict[str, Any]:
    """Read one newline-terminated JSON object with a real socket timeout."""
    if timeout_ms is not None:
        sock.settimeout(timeout_ms / 1000.0)
    buf = bytearray()
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("connection closed before complete line")
        buf.extend(chunk)
        if b"\n" in buf:
            line, _, _ = buf.partition(b"\n")
            try:
                return json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ReplyError("malformed JSON line from arbiter",
                                 {"field": "<line>", "detail": exc.msg}) from exc


def check_port_free(host: str, port: int) -> None:
    """Raise E_PORT if the loopback port is already in use."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        probe.bind((host, port))
    except OSError as exc:
        raise PortError("loopback port unavailable",
                        {"host": host, "port": port, "detail": str(exc)}) from exc
    finally:
        probe.close()


def wait_for_port(host: str, port: int, timeout_ms: int = HANDSHAKE_TIMEOUT_MS) -> None:
    deadline = time.monotonic() + timeout_ms / 1000.0
    last: OSError | None = None
    while time.monotonic() < deadline:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.25)
        try:
            probe.connect((host, port))
            probe.close()
            return
        except OSError as exc:
            last = exc
            time.sleep(0.02)
        finally:
            try:
                probe.close()
            except OSError:
                pass
    raise PortError("arbiter did not open its port in time",
                    {"host": host, "port": port, "detail": str(last)})


class ArbiterClient:
    """One persistent JSON-lines connection to a single domain arbiter."""

    def __init__(self, host: str, port: int, domain_id: str,
                 domain_version: int, domain_set_version: int,
                 handshake_timeout_ms: int = HANDSHAKE_TIMEOUT_MS):
        self.host = host
        self.port = port
        self.domain_id = domain_id
        self._sock: socket.socket | None = None
        self._io_lock = threading.Lock()
        self._connect(domain_version, domain_set_version, handshake_timeout_ms)

    def _connect(self, domain_version: int, domain_set_version: int,
                 timeout_ms: int) -> None:
        try:
            sock = socket.create_connection((self.host, self.port),
                                            timeout=timeout_ms / 1000.0)
        except socket.timeout as exc:
            raise HandshakeError("TCP connect timed out",
                                 {"host": self.host, "port": self.port}) from exc
        except OSError as exc:
            raise PortError("cannot connect to arbiter port",
                            {"host": self.host, "port": self.port,
                             "detail": str(exc)}) from exc
        sock.settimeout(timeout_ms / 1000.0)
        try:
            send_line(sock, {"type": "HELLO", "domain_id": self.domain_id,
                             "domain_version": domain_version,
                             "domain_set_version": domain_set_version})
            ack = recv_line(sock, timeout_ms)
        except socket.timeout as exc:
            sock.close()
            raise HandshakeError("handshake HELLO_ACK timed out",
                                 {"host": self.host, "port": self.port,
                                  "domain_id": self.domain_id}) from exc
        except (ConnectionError, OSError) as exc:
            sock.close()
            raise HandshakeError("handshake connection failed",
                                 {"host": self.host, "port": self.port,
                                  "detail": str(exc)}) from exc

        if ack.get("type") != "HELLO_ACK" or ack.get("domain_id") != self.domain_id:
            sock.close()
            raise HandshakeError("invalid handshake acknowledgement",
                                 {"host": self.host, "port": self.port,
                                  "domain_id": self.domain_id, "reply": ack})
        sock.settimeout(REQUEST_TIMEOUT_MS / 1000.0)
        self._sock = sock

    def request(self, seq: int, holder_id: str, action: str,
                priority: int) -> dict[str, Any]:
        with self._io_lock:
            return self._request_locked(seq, holder_id, action, priority)

    def _request_locked(self, seq: int, holder_id: str, action: str,
                        priority: int) -> dict[str, Any]:
        if self._sock is None:
            raise ConnectionError("client is closed")
        send_line(self._sock, {"type": "REQUEST", "seq": seq,
                               "holder_id": holder_id, "action": action,
                               "priority": priority})
        try:
            msg = recv_line(self._sock, REQUEST_TIMEOUT_MS)
        except socket.timeout as exc:
            raise ReplyError("arbiter reply timed out",
                             {"seq": seq, "domain_id": self.domain_id}) from exc
        if msg.get("type") == "ERROR":
            raise ReplyError("arbiter returned error envelope",
                             {"seq": seq, "domain_id": self.domain_id,
                              "detail": msg.get("message")})
        if msg.get("type") != "RECEIPT":
            raise ReplyError("expected RECEIPT message",
                             {"seq": seq, "domain_id": self.domain_id,
                              "actual_type": msg.get("type")})
        for field in REQUIRED_RECEIPT_FIELDS:
            if field not in msg:
                raise ReplyError("receipt missing required field",
                                 {"seq": seq, "domain_id": self.domain_id,
                                  "field": field})
        return msg

    def status(self) -> dict[str, Any]:
        with self._io_lock:
            if self._sock is None:
                raise ConnectionError("client is closed")
            send_line(self._sock, {"type": "STATUS"})
            msg = recv_line(self._sock, REQUEST_TIMEOUT_MS)
            if msg.get("type") != "STATE":
                raise ReplyError("expected STATE message",
                                 {"domain_id": self.domain_id,
                                  "actual_type": msg.get("type")})
            return msg.get("state", {})

    def ping(self) -> bool:
        with self._io_lock:
            if self._sock is None:
                return False
            send_line(self._sock, {"type": "PING"})
            return recv_line(self._sock, 2_000).get("type") == "PONG"

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None


def serve_arbiter(arbiter, host: str, port: int, domain_set_version: int,
                  ready_path: str | None = None) -> None:
    """Synchronous, single-threaded service loop executed in the subprocess.

    A fresh arbiter instance is built per process start. After a forced kill and
    restart the in-memory hold table is empty: old ownership is never reused;
    recovery rehydrates only from the committed ledger.
    """
    import os

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        server.bind((host, port))
    except OSError as exc:
        if ready_path:
            _write_ready(ready_path, {"status": "E_PORT", "detail": str(exc),
                                      "port": port})
        raise
    server.listen(SO_BACKLOG)
    server.settimeout(0.5)
    if ready_path:
        _write_ready(ready_path, {"status": "READY", "pid": os.getpid(),
                                  "port": port, "domain_id": arbiter.domain.domain_id})

    while True:
        try:
            conn, _addr = server.accept()
        except socket.timeout:
            # Background real-time expiry sweep even when idle.
            arbiter.reap()
            continue
        except OSError:
            break
        try:
            _serve_conn(conn, arbiter, domain_set_version)
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()


def _serve_conn(conn: socket.socket, arbiter, domain_set_version: int) -> None:
    conn.settimeout(REQUEST_TIMEOUT_MS / 1000.0)
    hello_done = False
    while True:
        try:
            msg = recv_line(conn, REQUEST_TIMEOUT_MS)
        except (ConnectionError, OSError):
            return
        mtype = msg.get("type")
        if mtype == "HELLO":
            if msg.get("domain_id") != arbiter.domain.domain_id:
                send_line(conn, {"type": "ERROR", "message": "domain mismatch"})
                return
            send_line(conn, {"type": "HELLO_ACK",
                             "domain_id": arbiter.domain.domain_id,
                             "domain_version": arbiter.domain.version,
                             "domain_set_version": domain_set_version,
                             "pid": _pid()})
            hello_done = True
            continue
        if not hello_done:
            send_line(conn, {"type": "ERROR", "message": "HELLO required first"})
            return
        if mtype == "PING":
            send_line(conn, {"type": "PONG"})
            continue
        if mtype == "REQUEST":
            try:
                receipt = arbiter.handle(
                    seq=int(msg["seq"]), holder_id=str(msg["holder_id"]),
                    action=str(msg["action"]), priority=int(msg["priority"]))
            except Exception as exc:  # service stays alive; envelope the fault
                send_line(conn, {"type": "ERROR", "message": str(exc)})
                continue
            payload = receipt.as_dict()
            payload["type"] = "RECEIPT"
            payload["transitions"] = [asdict(t) for t in arbiter.transitions
                                      if t.seq == receipt.seq
                                      and t.domain_id == receipt.domain_id]
            if getattr(receipt, "drain_transitions", None):
                payload["drain_transitions"] = [
                    asdict(t) for t in receipt.drain_transitions]
            send_line(conn, payload)
            continue
        if mtype == "STATUS":
            try:
                snapshot = arbiter.snapshot()
                send_line(conn, {"type": "STATE", "state": snapshot})
            except Exception as exc:
                send_line(conn, {"type": "ERROR", "message": f"status: {exc}"})
            continue
        if mtype == "SHUTDOWN":
            send_line(conn, {"type": "BYE"})
            return
        send_line(conn, {"type": "ERROR", "message": f"unknown type {mtype!r}"})


def _pid() -> int:
    import os
    return os.getpid()


def _write_ready(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
