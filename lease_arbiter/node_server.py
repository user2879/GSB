from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import socket
import sys
import threading
import time
from typing import Any

from .canonical import canonical_json
from .errors import LeaseError, StaleError


MAX_LINE_BYTES = 1_048_576


def _failure(request: dict[str, Any], code: str, message: str, **details: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "ok": False,
        "code": code,
        "message": message,
        "request_id": request.get("request_id"),
    }
    payload.update(details)
    return payload


class NodeLeaseStore:
    def __init__(self) -> None:
        self._leases: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        with self._lock:
            if op == "grant":
                return self._grant(request)
            if op == "renew":
                return self._renew(request)
            if op == "release":
                return self._release(request)
            if op == "snapshot":
                return self._snapshot(request)
            return _failure(request, "E_PROTOCOL", "unsupported node operation", op=op)

    def _grant(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        expected_label = request.get("expected_label")
        lease_type = request.get("lease_type")
        if not isinstance(lease_id, str) or not lease_id:
            return _failure(request, "E_PROTOCOL", "lease_id is required")
        if not isinstance(expected_label, int) or isinstance(expected_label, bool) or expected_label < 0:
            return _failure(request, "E_PROTOCOL", "expected_label must be a non-negative integer")
        if not isinstance(lease_type, str) or not lease_type:
            return _failure(request, "E_PROTOCOL", "lease_type is required")
        record = self._leases.setdefault(
            lease_id,
            {"label": 0, "state": "new", "lease_type": lease_type},
        )
        if record["state"] in {"granted", "renewed"}:
            return _failure(
                request,
                "E_LEASE",
                "lease is already active",
                lease_id=lease_id,
                current_label=record["label"],
                state=record["state"],
            )
        if expected_label != record["label"]:
            return _failure(
                request,
                "E_STALE",
                "request label is stale",
                lease_id=lease_id,
                expected_label=expected_label,
                current_label=record["label"],
                state=record["state"],
            )
        record["label"] += 1
        record["state"] = "granted"
        record["lease_type"] = lease_type
        return {
            "format_version": 1,
            "ok": True,
            "op": "grant",
            "lease_id": lease_id,
            "lease_type": lease_type,
            "label": record["label"],
            "state": record["state"],
            "request_id": request.get("request_id"),
        }

    def _renew(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._active_mutation(request, "renew")

    def _release(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._active_mutation(request, "release")

    def _active_mutation(self, request: dict[str, Any], op: str) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        expected_label = request.get("expected_label")
        if not isinstance(lease_id, str) or not lease_id:
            return _failure(request, "E_PROTOCOL", "lease_id is required")
        if not isinstance(expected_label, int) or isinstance(expected_label, bool) or expected_label < 0:
            return _failure(request, "E_PROTOCOL", "expected_label must be a non-negative integer")
        record = self._leases.get(lease_id)
        if record is None:
            return _failure(
                request,
                "E_STALE",
                "request label is stale for an unknown lease",
                lease_id=lease_id,
                expected_label=expected_label,
                current_label=0,
                state="unknown",
            )
        if expected_label != record["label"]:
            return _failure(
                request,
                "E_STALE",
                "request label is stale",
                lease_id=lease_id,
                expected_label=expected_label,
                current_label=record["label"],
                state=record["state"],
            )
        if record["state"] not in {"granted", "renewed"}:
            return _failure(
                request,
                "E_LEASE",
                "lease is not active",
                lease_id=lease_id,
                current_label=record["label"],
                state=record["state"],
            )
        record["label"] += 1
        record["state"] = "renewed" if op == "renew" else "released"
        return {
            "format_version": 1,
            "ok": True,
            "op": op,
            "lease_id": lease_id,
            "lease_type": record["lease_type"],
            "label": record["label"],
            "state": record["state"],
            "request_id": request.get("request_id"),
        }

    def _snapshot(self, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "format_version": 1,
            "ok": True,
            "op": "snapshot",
            "request_id": request.get("request_id"),
            "leases": [
                {
                    "lease_id": lease_id,
                    "label": self._leases[lease_id]["label"],
                    "state": self._leases[lease_id]["state"],
                    "lease_type": self._leases[lease_id]["lease_type"],
                }
                for lease_id in sorted(self._leases)
            ],
        }


class NodeServer:
    def __init__(self, node_id: str, host: str, port: int, read_timeout_ms: int) -> None:
        self.node_id = node_id
        self.host = host
        self.port = port
        self.read_timeout_ms = read_timeout_ms
        self.store = NodeLeaseStore()
        self.stop_event = threading.Event()
        self.listener: socket.socket | None = None
        self._threads: list[threading.Thread] = []

    def serve(self) -> int:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self.host, self.port))
            listener.listen(64)
            listener.settimeout(0.2)
        except OSError as exc:
            print(
                canonical_json(
                    {
                        "ok": False,
                        "code": "E_PORT",
                        "message": f"cannot bind {self.host}:{self.port}: {exc}",
                        "node_id": self.node_id,
                        "host": self.host,
                        "port": self.port,
                    }
                ),
                flush=True,
            )
            listener.close()
            return 20
        self.listener = listener
        print(
            canonical_json(
                {
                    "ok": True,
                    "code": "READY",
                    "node_id": self.node_id,
                    "host": self.host,
                    "port": self.port,
                    "pid": __import__("os").getpid(),
                }
            ),
            flush=True,
        )
        try:
            while not self.stop_event.is_set():
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop_event.is_set():
                        break
                    raise
                thread = threading.Thread(target=self._handle_connection, args=(conn,), daemon=True)
                thread.start()
                self._threads.append(thread)
        finally:
            self.stop_event.set()
            try:
                listener.close()
            except OSError:
                pass
            for thread in self._threads:
                thread.join(timeout=0.5)
        return 0

    def stop(self) -> None:
        self.stop_event.set()

    def _handle_connection(self, conn: socket.socket) -> None:
        conn.settimeout(self.read_timeout_ms / 1000.0)
        buffer = bytearray()
        try:
            hello_line = self._read_line(conn, buffer)
            if hello_line is None:
                return
            hello = self._decode(hello_line)
            if hello.get("op") != "hello" or hello.get("protocol") != 1:
                self._send(conn, _failure(hello, "E_PROTOCOL", "first frame must be hello protocol 1"))
                return
            self._send(
                conn,
                {
                    "format_version": 1,
                    "ok": True,
                    "op": "hello",
                    "protocol": 1,
                    "node_id": self.node_id,
                    "request_id": hello.get("request_id"),
                },
            )
            request_line = self._read_line(conn, buffer)
            if request_line is None:
                return
            request = self._decode(request_line)
            response = self.store.process(request)
            response["node_id"] = self.node_id
            self._send(conn, response)
        except socket.timeout:
            try:
                self._send(
                    conn,
                    {
                        "format_version": 1,
                        "ok": False,
                        "code": "E_TIMEOUT",
                        "message": "node socket read timed out",
                        "node_id": self.node_id,
                    },
                )
            except OSError:
                pass
        except (OSError, ValueError, json.JSONDecodeError):
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _read_line(self, conn: socket.socket, buffer: bytearray) -> bytes | None:
        while True:
            newline = buffer.find(b"\n")
            if newline >= 0:
                line = bytes(buffer[:newline])
                del buffer[: newline + 1]
                if len(line) > MAX_LINE_BYTES:
                    return None
                return line
            chunk = conn.recv(65536)
            if not chunk:
                return None
            buffer.extend(chunk)
            if len(buffer) > MAX_LINE_BYTES:
                return None

    @staticmethod
    def _decode(line: bytes) -> dict[str, Any]:
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("protocol frame must be a JSON object")
        return value

    @staticmethod
    def _send(conn: socket.socket, payload: dict[str, Any]) -> None:
        conn.sendall((canonical_json(payload) + "\n").encode("utf-8"))


def _install_parent_death_signal() -> None:
    """Ensure a SIGKILLed central session does not leave loopback nodes behind."""

    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None)
        # PR_SET_PDEATHSIG = 1
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            return
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        return


def run_node(node_id: str, host: str, port: int, read_timeout_ms: int) -> int:
    _install_parent_death_signal()
    server = NodeServer(node_id, host, port, read_timeout_ms)

    def _stop(_signum: int, _frame: Any) -> None:
        server.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    return server.serve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="loopback JSONL lease node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--read-timeout-ms", type=int, default=1000)
    args = parser.parse_args(argv)
    return run_node(args.node_id, args.host, args.port, args.read_timeout_ms)


if __name__ == "__main__":
    raise SystemExit(main())
