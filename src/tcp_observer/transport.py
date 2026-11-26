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
from .errors import PortError, TimeoutError as ObsTimeoutError
from .spec import SessionSpec

SRC_DIR = Path(__file__).resolve().parents[1]


def bind_listener(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
        sock.listen(8)
    except OSError as exc:
        sock.close()
        code = getattr(exc, "errno", None)
        if code in {98, 10048} or "Address already in use" in str(exc):
            raise PortError(
                f"TCP port {port} on {host} is already occupied",
                field_name="peer_port",
                details={"host": host, "port": port, "errno": code},
            ) from exc
        raise PortError(
            f"cannot bind TCP {host}:{port}: {exc}",
            field_name="peer_port",
            details={"host": host, "port": port, "errno": code},
        ) from exc
    return sock


def is_port_available(host: str, port: int) -> bool:
    sock = None
    try:
        sock = bind_listener(host, port)
        return True
    except PortError:
        return False
    finally:
        if sock is not None:
            sock.close()


def _socket_timeout_seconds(timeout_ms: int) -> float:
    return max(0.001, timeout_ms / 1000.0)


def read_json_line(reader: Any, timeout_ms: int, *, context: str) -> dict[str, Any]:
    # The timeout is owned by the real socket; callers set it before readline.
    line = reader.readline()
    if not line:
        raise ObsTimeoutError(
            f"socket closed before {context}",
            details={"context": context, "timeout_ms": timeout_ms},
        )
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ObsTimeoutError(
            f"invalid JSON line while {context}: {exc.msg}",
            details={"context": context, "line": line[:200]},
        ) from exc
    if not isinstance(value, dict):
        raise ObsTimeoutError(f"JSON line while {context} must be an object", details={"context": context})
    return value


def connect_handshake(spec: SessionSpec, port: int) -> tuple[socket.socket, Any, dict[str, Any]]:
    timeout_s = _socket_timeout_seconds(spec.handshake_timeout_ms)
    try:
        sock = socket.create_connection((spec.peer_host, port), timeout=timeout_s)
    except (socket.timeout, TimeoutError) as exc:
        raise ObsTimeoutError(
            f"TCP connect/handshake to {spec.peer_host}:{port} timed out",
            field_name="handshake_timeout_ms",
            details={"host": spec.peer_host, "port": port, "timeout_ms": spec.handshake_timeout_ms},
        ) from exc
    except OSError as exc:
        raise PortError(
            f"cannot connect to local peer at {spec.peer_host}:{port}: {exc}",
            field_name="peer_port",
            details={"host": spec.peer_host, "port": port},
        ) from exc

    sock.settimeout(timeout_s)
    reader = sock.makefile("r", encoding="utf-8", newline="\n")
    try:
        hello = read_json_line(reader, spec.handshake_timeout_ms, context="reading peer hello")
    except (socket.timeout, TimeoutError) as exc:
        sock.close()
        raise ObsTimeoutError(
            "peer handshake hello timed out on a real socket timeout",
            field_name="handshake_timeout_ms",
            details={"host": spec.peer_host, "port": port, "timeout_ms": spec.handshake_timeout_ms},
        ) from exc
    if hello.get("type") != "hello":
        sock.close()
        raise ObsTimeoutError("first peer line was not hello", details={"hello": hello})

    client_hello = {
        "type": "hello_ack",
        "session_id": spec.session_id,
        "client": "tcp-observer",
        "format_version": 3,
    }
    sock.sendall((canonical_text(client_hello) + "\n").encode("utf-8"))
    sock.settimeout(_socket_timeout_seconds(spec.read_timeout_ms))
    return sock, reader, hello


def start_peer_subprocess(spec_path: str | Path) -> tuple[subprocess.Popen[bytes], int, dict[str, Any]]:
    spec_path = Path(spec_path).resolve()
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing if existing else "")
    cmd = [sys.executable, "-m", "tcp_observer.peer", "--spec", str(spec_path)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(SRC_DIR.parent),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    ready_line = proc.stdout.readline()
    if not ready_line:
        stderr = b""
        try:
            stdout_rest, stderr = proc.communicate(timeout=2)
            ready_line += stdout_rest
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_rest, stderr = proc.communicate()
            stderr = stderr or b""
        raise PortError(
            "peer subprocess exited before becoming ready",
            details={"cmd": cmd, "stderr": stderr.decode("utf-8", "replace")[:1000]},
        )
    try:
        ready = json.loads(ready_line)
    except json.JSONDecodeError as exc:
        proc.kill()
        raise PortError("peer ready line was invalid JSON", details={"line": ready_line.decode("utf-8", "replace")}) from exc
    if ready.get("code") == "E_PORT":
        proc.wait(timeout=2)
        raise PortError(ready.get("message", "peer port error"), details=ready.get("details", {}))
    port = int(ready["port"])
    return proc, port, ready


def stop_peer_subprocess(proc: subprocess.Popen[bytes], sock: socket.socket | None = None) -> None:
    if sock is not None:
        try:
            shutdown = {"type": "shutdown"}
            sock.sendall((canonical_text(shutdown) + "\n").encode("utf-8"))
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)
    for stream in (proc.stdout, proc.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def run_live_exchange(spec: SessionSpec) -> dict[str, Any]:
    if not spec.source_path:
        raise ValueError("live exchange requires a loaded spec source_path")
    started_ns = time.monotonic_ns()
    proc, port, ready = start_peer_subprocess(spec.source_path)
    sock = None
    reader = None
    try:
        sock, reader, hello = connect_handshake(spec, port)
        responses: list[dict[str, Any]] = []
        for event_index in range(spec.event_count):
            seq = spec.initial_seq + event_index
            segment = {
                "type": "segment",
                "event_seq": seq,
                "tcp_seq": spec.tcp_seq(event_index),
                "mss": spec.mss,
                "sent_monotonic_ms": (time.monotonic_ns() - started_ns) // 1_000_000,
            }
            sock.sendall((canonical_text(segment) + "\n").encode("utf-8"))
            try:
                response = read_json_line(reader, spec.read_timeout_ms, context=f"reading response for event {seq}")
            except (socket.timeout, TimeoutError) as exc:
                raise ObsTimeoutError(
                    f"real socket read timed out waiting for event {seq}",
                    field_name="read_timeout_ms",
                    event_seq=seq,
                    details={"timeout_ms": spec.read_timeout_ms},
                ) from exc
            responses.append(response)
        return {
            "port": port,
            "ready": ready,
            "hello": hello,
            "responses": responses,
            "elapsed_wall_ms": (time.monotonic_ns() - started_ns) // 1_000_000,
        }
    finally:
        if reader is not None:
            try:
                reader.close()
            except OSError:
                pass
        stop_peer_subprocess(proc, sock)
