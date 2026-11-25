"""Line-oriented JSON protocol for environment arbitration subprocesses."""
from __future__ import annotations

import json
import socket
from typing import Any

from .errors import ErrorContext, PortError, TimeoutErrorW

TERMINATOR = b"\n"
DEFAULT_TIMEOUT_S = 3.0


def send_line(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + TERMINATOR)


def recv_line(sock: socket.socket, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    sock.settimeout(timeout_s)
    buffer = bytearray()
    while True:
        try:
            chunk = sock.recv(4096)
        except socket.timeout as exc:
            raise TimeoutErrorW(
                "socket read timed out waiting for service reply",
                ErrorContext(detail=f"timeout={timeout_s}s"),
            ) from exc
        except OSError as exc:
            raise TimeoutErrorW(
                "socket read failed", ErrorContext(detail=str(exc))
            ) from exc
        if not chunk:
            raise TimeoutErrorW(
                "service closed connection before full reply",
                ErrorContext(),
            )
        buffer.extend(chunk)
        if TERMINATOR in chunk:
            line, _, _ = buffer.partition(TERMINATOR)
            return json.loads(line.decode("utf-8"))


def connect_loopback(host: str, port: int, timeout_s: float = DEFAULT_TIMEOUT_S) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((host, port))
    except socket.timeout as exc:
        sock.close()
        raise TimeoutErrorW(
            "socket handshake timed out",
            ErrorContext(host, str(port), f"timeout={timeout_s}s"),
        ) from exc
    except ConnectionRefusedError as exc:
        sock.close()
        raise PortError(
            "cannot connect to environment arbitration service",
            ErrorContext(host, str(port), str(exc)),
        ) from exc
    except OSError as exc:
        sock.close()
        raise PortError(
            "socket connect failed",
            ErrorContext(host, str(port), str(exc)),
        ) from exc
    return sock


def check_port_available(host: str, port: int) -> None:
    # Deliberately no SO_REUSEADDR: a second bind to a busy loopback port must
    # fail with EADDRINUSE so the E_PORT contract is enforceable.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise PortError(
            "loopback port already in use",
            ErrorContext(host, str(port), str(exc)),
        ) from exc
    finally:
        probe.close()


# Reserved high-port band for this workbench. Ephemeral (port 0) allocation is
# avoided because parallel local workloads can receive the same candidate.
RESERVED_PORT_BAND = range(47100, 47500)


def pick_free_port(host: str = "127.0.0.1",
                   candidates=RESERVED_PORT_BAND) -> int:
    """Return a confirmed-free port from the reserved loopback band.

    Candidates are shuffled per call so rapid successive test runs do not keep
    requesting the same low port while it sits in TIME_WAIT.
    """
    import random

    options = list(candidates)
    random.Random().shuffle(options)
    for port in options:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((host, port))
        except OSError:
            continue
        finally:
            probe.close()
        return port
    raise PortError(
        "no free port left in reserved loopback band",
        ErrorContext(host, f"{min(candidates)}-{max(candidates)}"),
    )
