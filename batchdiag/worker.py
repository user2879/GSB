"""Local loopback worker subprocess and socket client.

A worker is a real TCP server bound to 127.0.0.1:<high port>, started as a
subprocess by the demo/test driver. The client performs real sockets with a
real timeout; a worker started in ``stall`` mode never answers and the
client reports E_TIMEOUT instead of pretending.
"""

import json
import socket
import struct
import time

from .canonical import dumps
from .errors import FieldError, TimeoutError_

_PREFIX = struct.Struct(">Q")


def ensure_port_free(host, port, timeout=1.0):
    """Raise if *port* is already accepting connections on *host*."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(timeout)
        probe.connect((host, port))
        in_use = True
    except OSError:
        in_use = False
    finally:
        probe.close()
    if in_use:
        raise FieldError(
            f"loopback port {port} on {host} is already in use",
            field="port",
        )
    return True


def _read_exact(conn, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("worker closed the connection early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def serve_once(host, port, stall_ms=0, exit_after=1, verbose=False):
    """Accept *exit_after* framed requests then return (used by __main__)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(8)
    handled = 0
    stall_seconds = stall_ms / 1000.0
    try:
        while handled < exit_after:
            conn, _addr = server.accept()
            with conn:
                try:
                    length = _PREFIX.unpack(_read_exact(conn, 8))[0]
                    raw = _read_exact(conn, length)
                    request = json.loads(raw.decode("utf-8"))
                    if stall_seconds:
                        time.sleep(stall_seconds)
                    response = {
                        "type": "receipt_ack",
                        "worker_id": request.get("worker_id"),
                        "batch_id": request.get("batch_id"),
                        "output_digest": request.get("output_digest"),
                        "accepted": True,
                    }
                    payload = dumps(response).encode("utf-8")
                    conn.sendall(_PREFIX.pack(len(payload)) + payload)
                    handled += 1
                except (ConnectionError, OSError):
                    continue
    finally:
        server.close()


def send_receipt(host, port, payload, timeout_ms):
    """One real framed round trip with a real socket timeout."""
    body = dumps(payload).encode("utf-8")
    frame = _PREFIX.pack(len(body)) + body
    started = time.monotonic_ns() // 1_000_000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout_ms / 1000.0)
        sock.connect((host, port))
        sock.sendall(frame)
        length = _PREFIX.unpack(_read_exact(sock, 8))[0]
        raw = _read_exact(sock, length)
        response = json.loads(raw.decode("utf-8"))
        if response.get("batch_id") != payload.get("batch_id"):
            raise FieldError(
                "worker ack batch_id does not match the request",
                field="batch_id",
                extra={"sent": payload.get("batch_id"),
                       "received": response.get("batch_id")},
            )
        elapsed = time.monotonic_ns() // 1_000_000 - started
        return response, elapsed
    except socket.timeout:
        elapsed = time.monotonic_ns() // 1_000_000 - started
        raise TimeoutError_(
            f"worker at {host}:{port} did not answer within "
            f"{timeout_ms} ms",
            field="socket",
            extra={"host": host, "port": port,
                   "timeout_ms": timeout_ms, "elapsed_ms": elapsed},
        )
    except OSError as exc:
        raise FieldError(
            f"loopback communication failed: {exc}",
            field="socket",
            extra={"host": host, "port": port},
        )
    finally:
        sock.close()
