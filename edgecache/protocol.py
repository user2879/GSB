"""Minimal JSON-lines protocol over loopback TCP."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from typing import Any, Callable

from .errors import ERROR_CODES, EdgeError


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 8 * 1024 * 1024


class JsonLineConnection:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.reader = sock.makefile("rb")
        self.writer = sock.makefile("wb")

    def send(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_LINE_BYTES:
            raise EdgeError(
                code="E_PROTOCOL",
                message="JSONL message exceeds maximum size",
                details={"bytes": len(encoded), "maximum": MAX_LINE_BYTES},
            )
        try:
            self.writer.write(encoded + b"\n")
            self.writer.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise EdgeError(
                code="E_PROTOCOL",
                message="failed to write JSONL message",
                cause=type(exc).__name__,
            ) from exc

    def receive(self) -> dict[str, Any]:
        try:
            line = self.reader.readline(MAX_LINE_BYTES + 1)
        except (socket.timeout, TimeoutError) as exc:
            raise EdgeError(
                code="E_TIMEOUT",
                message="socket timed out while waiting for a JSONL message",
                cause=type(exc).__name__,
            ) from exc
        except (ConnectionResetError, OSError) as exc:
            raise EdgeError(
                code="E_PROTOCOL",
                message="failed to read JSONL message",
                cause=type(exc).__name__,
            ) from exc
        if not line:
            raise EdgeError(code="E_PROTOCOL", message="peer closed the connection before a response")
        if len(line) > MAX_LINE_BYTES:
            raise EdgeError(code="E_PROTOCOL", message="JSONL message exceeds maximum size")
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EdgeError(
                code="E_PROTOCOL",
                message="peer returned invalid JSONL",
                cause=type(exc).__name__,
            ) from exc
        if not isinstance(value, dict):
            raise EdgeError(code="E_PROTOCOL", message="JSONL response must be an object")
        return value

    def close(self) -> None:
        for stream in (self.reader, self.writer):
            try:
                stream.close()
            except OSError:
                pass
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self) -> "JsonLineConnection":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def connect_jsonl(host: str, port: int, timeout: float) -> JsonLineConnection:
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except (socket.timeout, TimeoutError) as exc:
        raise EdgeError(
            code="E_TIMEOUT",
            message="real socket timeout while connecting to a local worker",
            details={"host": host, "port": port, "timeout_seconds": timeout},
            cause=type(exc).__name__,
        ) from exc
    except OSError as exc:
        raise EdgeError(
            code="E_PORT",
            message="unable to connect to local worker",
            details={"host": host, "port": port, "timeout_seconds": timeout, "error": str(exc)},
            cause=type(exc).__name__,
        ) from exc
    sock.settimeout(timeout)
    return JsonLineConnection(sock)


def handshake(
    host: str,
    port: int,
    *,
    client: str,
    timeout: float,
    expected_service: str,
) -> dict[str, Any]:
    with connect_jsonl(host, port, timeout) as connection:
        connection.send(
            {
                "op": "hello",
                "protocol": PROTOCOL_VERSION,
                "client": client,
            }
        )
        response = connection.receive()
    if not response.get("ok"):
        code = str(response.get("code") or "E_PROTOCOL")
        if code not in ERROR_CODES:
            code = "E_PROTOCOL"
        raise EdgeError(
            code=code,
            message=str(response.get("message") or "worker rejected handshake"),
            details={"service": expected_service, "response": response},
        )
    if response.get("service") != expected_service:
        raise EdgeError(
            code="E_PROTOCOL",
            message="worker handshake returned an unexpected service",
            details={"expected": expected_service, "actual": response.get("service")},
        )
    if int(response.get("protocol", 0)) != PROTOCOL_VERSION:
        raise EdgeError(
            code="E_PROTOCOL",
            message="worker protocol version mismatch",
            details={"expected": PROTOCOL_VERSION, "actual": response.get("protocol")},
        )
    return response


def request(host: str, port: int, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    with connect_jsonl(host, port, timeout) as connection:
        connection.send(payload)
        response = connection.receive()
    if not response.get("ok"):
        code = str(response.get("code") or "E_PROTOCOL")
        if code not in {
            "E_SPEC",
            "E_CONTENT",
            "E_CYCLE",
            "E_PORT",
            "E_TIMEOUT",
            "E_PROTOCOL",
            "E_PROCESS",
            "E_STALE",
            "E_SNAPSHOT",
            "E_REPLAY",
            "E_STATE",
            "E_TK",
        }:
            code = "E_PROTOCOL"
        raise EdgeError(
            code=code,
            message=str(response.get("message") or "worker request failed"),
            path=response.get("path"),
            field_path=response.get("field_path"),
            details=response.get("details") if isinstance(response.get("details"), dict) else {},
        )
    return response


@dataclass
class JsonLineServer:
    host: str
    port: int
    service: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    sock: socket.socket | None = None

    def bind(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if os.name == "nt" and hasattr(socket, "SO_REUSEADDR"):
            # Allows a worker to rebind its own high loopback port while old
            # accepted sockets are in TIME_WAIT.  The orchestrator's exclusive
            # preflight still detects a live listener before spawning.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        elif hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except OSError:
                pass
        elif hasattr(socket, "SO_REUSEADDR"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
            sock.listen(16)
        except OSError as exc:
            sock.close()
            raise EdgeError(
                code="E_PORT",
                message="loopback port is already occupied or unavailable",
                details={"host": self.host, "port": self.port, "error": str(exc), "service": self.service},
                cause=type(exc).__name__,
            ) from exc
        sock.settimeout(0.25)
        self.sock = sock

    def serve_forever(self) -> None:
        if self.sock is None:
            self.bind()
        assert self.sock is not None
        while True:
            try:
                client, _address = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(5.0)
            with JsonLineConnection(client) as connection:
                while True:
                    try:
                        message = connection.receive()
                    except EdgeError as exc:
                        if exc.code == "E_PROTOCOL" and exc.message.startswith("peer closed"):
                            break
                        try:
                            connection.send(exc.envelope())
                        except EdgeError:
                            pass
                        break
                    try:
                        if int(message.get("protocol", PROTOCOL_VERSION)) != PROTOCOL_VERSION and message.get("op") != "hello":
                            raise EdgeError(
                                code="E_PROTOCOL",
                                message="unsupported protocol version",
                                details={"expected": PROTOCOL_VERSION, "actual": message.get("protocol")},
                            )
                        response = self.handler(message)
                        response.setdefault("ok", True)
                    except EdgeError as exc:
                        response = exc.envelope()
                    except Exception as exc:  # defensive process boundary
                        response = EdgeError.from_exception(exc).envelope()
                    try:
                        connection.send(response)
                    except EdgeError:
                        break
                    if response.get("shutdown"):
                        return
                    if response.get("close_connection"):
                        break

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
