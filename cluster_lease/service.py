"""One deterministic lease arbitration service for one cluster."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import sys
import threading
from typing import Any

from .canonical import PROTOCOL_VERSION, canonical_json
from .errors import LeaseError
from .models import ClusterSpec
from .state_machine import ArbitrationEngine

MAX_LINE_BYTES = 1_048_576


def _log(value: dict[str, Any]) -> None:
    sys.stderr.write(canonical_json(value) + "\n")
    sys.stderr.flush()


def _parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class LineConnection:
    def __init__(self, conn: socket.socket) -> None:
        self.conn = conn
        self.buffer = bytearray()

    def read_line(self) -> dict[str, Any] | None:
        while b"\n" not in self.buffer:
            chunk = self.conn.recv(65536)
            if not chunk:
                return None
            self.buffer.extend(chunk)
            if len(self.buffer) > MAX_LINE_BYTES:
                raise LeaseError("E_PROTOCOL", "request line exceeds maximum length")
        raw, _, remainder = self.buffer.partition(b"\n")
        self.buffer = bytearray(remainder)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LeaseError("E_PROTOCOL", f"invalid JSON line: {exc}") from None
        if not isinstance(value, dict):
            raise LeaseError("E_PROTOCOL", "request must be a JSON object")
        return value

    def send(self, value: dict[str, Any]) -> None:
        self.conn.sendall((canonical_json(value) + "\n").encode("utf-8"))


def serve(config_path: str) -> int:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    spec = ClusterSpec(
        cluster_id=config["cluster_id"],
        host=config.get("host", "127.0.0.1"),
        port=int(config["port"]),
        lease_duration_ms=int(config["lease_duration_ms"]),
        renewal_interval_ms=int(config["renewal_interval_ms"]),
        capacity=int(config["capacity"]),
    )
    engine = ArbitrationEngine(
        spec,
        {str(key): int(value) for key, value in config["priorities"].items()},
        int(config.get("parameter_version", 1)),
        clock_mode=config.get("clock_mode", "hybrid"),
    )
    lock = threading.Lock()
    stop_event = threading.Event()
    parent_pid = int(config.get("parent_pid", 0))

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name != "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((spec.host, spec.port))
        listener.listen(16)
        listener.settimeout(0.25)
    except OSError as exc:
        try:
            listener.close()
        except OSError:
            pass
        _log(
            {
                "code": "E_PORT",
                "field": "port",
                "message": f"cannot bind {spec.host}:{spec.port}: {exc}",
                "ok": False,
                "port": spec.port,
            }
        )
        return 2

    def watchdog() -> None:
        while not stop_event.wait(0.2):
            if parent_pid and not _parent_alive(parent_pid):
                stop_event.set()
                return

    if parent_pid:
        threading.Thread(target=watchdog, name="parent-watchdog", daemon=True).start()

    _log(
        {
            "cluster_id": spec.cluster_id,
            "code": "SERVICE_LISTENING",
            "ok": True,
            "port": spec.port,
            "protocol": PROTOCOL_VERSION,
        }
    )
    try:
        while not stop_event.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(60.0)
            connection = LineConnection(conn)
            try:
                hello = connection.read_line()
                if hello is None:
                    continue
                if hello.get("type") != "hello":
                    connection.send(
                        {
                            "code": "E_PROTOCOL",
                            "field": "type",
                            "message": "first frame must be hello",
                            "ok": False,
                        }
                    )
                    continue
                connection.send(
                    {
                        "cluster_id": spec.cluster_id,
                        "ok": True,
                        "parameter_version": engine.parameter_version,
                        "protocol": PROTOCOL_VERSION,
                        "type": "hello",
                    }
                )
                while not stop_event.is_set():
                    try:
                        request = connection.read_line()
                    except socket.timeout:
                        continue
                    if request is None:
                        break
                    request_type = request.get("type")
                    with lock:
                        if request_type == "shutdown":
                            connection.send({"ok": True, "type": "shutdown"})
                            # Wait briefly for the client to close first so the
                            # listening port is not left in server-side TIME_WAIT.
                            try:
                                conn.settimeout(1.0)
                                connection.read_line()
                            except (OSError, socket.timeout, LeaseError):
                                pass
                            stop_event.set()
                            break
                        if request_type == "status":
                            timestamp = request.get("timestamp_ms")
                            if timestamp is not None and (
                                not isinstance(timestamp, int) or isinstance(timestamp, bool)
                            ):
                                raise LeaseError(
                                    "E_PROTOCOL",
                                    "timestamp_ms must be an integer",
                                    field="timestamp_ms",
                                )
                            connection.send(engine.status(timestamp))
                            continue
                        if request_type == "restore":
                            last_seq = request.get("last_event_seq", 0)
                            if not isinstance(last_seq, int) or isinstance(last_seq, bool):
                                raise LeaseError(
                                    "E_PROTOCOL",
                                    "last_event_seq must be an integer",
                                    field="last_event_seq",
                                )
                            leases = request.get("leases", [])
                            if not isinstance(leases, list):
                                raise LeaseError(
                                    "E_PROTOCOL", "leases must be a list", field="leases"
                                )
                            last_timestamp = request.get("last_timestamp_ms", 0)
                            if not isinstance(last_timestamp, int) or isinstance(last_timestamp, bool):
                                raise LeaseError(
                                    "E_PROTOCOL",
                                    "last_timestamp_ms must be an integer",
                                    field="last_timestamp_ms",
                                )
                            connection.send(engine.restore(leases, last_seq, last_timestamp))
                            continue
                        if request_type != "lease":
                            raise LeaseError(
                                "E_PROTOCOL",
                                f"unsupported request type: {request_type}",
                                field="type",
                            )
                        connection.send(engine.process(request))
            except LeaseError as exc:
                try:
                    connection.send(exc.to_dict())
                except OSError:
                    pass
            except (ConnectionError, OSError, socket.timeout):
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
    finally:
        listener.close()
        _log({"cluster_id": spec.cluster_id, "code": "SERVICE_STOPPED", "ok": True})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one cluster lease arbitration service")
    parser.add_argument("--config", required=True, help="path to service JSON config")
    args = parser.parse_args(argv)
    try:
        return serve(args.config)
    except LeaseError as exc:
        _log(exc.to_dict())
        return 2
    except Exception as exc:  # pragma: no cover - startup guard for child process
        _log({"code": "E_SERVICE", "message": str(exc), "ok": False})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
