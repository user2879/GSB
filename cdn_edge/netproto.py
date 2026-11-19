from __future__ import annotations

import json
import os
import socket
import socketserver
import tempfile
from pathlib import Path
from typing import Any

from .errors import PipelineError

MAX_LINE_BYTES = 32 * 1024 * 1024


def encode_message(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def decode_message(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError(
            "E_PROTOCOL",
            "invalid JSON line protocol message",
            path="$",
            field="line",
            details={"reason": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise PipelineError("E_PROTOCOL", "protocol payload must be an object", path="$", field="$")
    return payload


def send_json_line(host: str, port: int, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(encode_message(payload))
            reader = sock.makefile("rb")
            line = reader.readline(MAX_LINE_BYTES + 1)
    except OSError as exc:
        raise PipelineError(
            "E_PROTOCOL",
            "TCP JSON-line endpoint is unavailable",
            path=f"{host}:{port}",
            field="socket",
            details={"reason": str(exc)},
        ) from exc
    if not line:
        raise PipelineError("E_PROTOCOL", "empty response from endpoint", path=f"{host}:{port}")
    if len(line) > MAX_LINE_BYTES:
        raise PipelineError("E_PROTOCOL", "response line exceeds size limit", path=f"{host}:{port}")
    return decode_message(line)


class JsonLineHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw = self.rfile.readline(MAX_LINE_BYTES + 1)
            if not raw:
                return
            if len(raw) > MAX_LINE_BYTES:
                self._write(
                    PipelineError(
                        "E_PROTOCOL", "request line exceeds size limit", path="$", field="line"
                    ).to_dict()
                )
                return
            try:
                payload = decode_message(raw)
                response = self.server.app.handle(payload)  # type: ignore[attr-defined]
            except PipelineError as exc:
                response = exc.to_dict()
            except Exception as exc:  # defensive boundary for worker processes
                response = PipelineError(
                    "E_PROTOCOL",
                    "unhandled worker exception",
                    path="$",
                    field="handler",
                    details={"exception_type": type(exc).__name__, "reason": str(exc)},
                ).to_dict()
            self._write(response)

    def _write(self, payload: dict[str, Any]) -> None:
        self.wfile.write(encode_message(payload))
        self.wfile.flush()


class JsonLineServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: Any) -> None:
        self.app = app
        super().__init__(address, JsonLineHandler)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_json_file(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def port_is_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        if family == socket.AF_INET:
            if os.name == "nt":
                exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
                if exclusive is not None:
                    probe.setsockopt(socket.SOL_SOCKET, exclusive, 1)
            else:
                # Permit a TIME_WAIT rebind, but an active listener still makes
                # bind() fail unless it explicitly opted into port sharing.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()
