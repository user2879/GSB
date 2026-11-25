"""Loopback worker subprocess.

Run as ``python -m speaker_diag.worker --spec FILE --worker-id ID``.  The
worker owns no global state except the loaded declaration; each connection is
handled independently so bounded concurrent batches are possible.
"""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes
from .engine import infer_batch, resolve_payload_features
from .errors import DiagnosticError
from .model import ParameterCache
from .protocol import LineReader, PROTOCOL_VERSION, require_fields, send_json_line
from .spec import load_workload


class WorkerServer:
    def __init__(self, spec_path: str | Path, worker_id: str):
        self.spec = load_workload(spec_path)
        self.worker = self.spec.worker_by_id(worker_id)
        self.cache = ParameterCache()
        self.parameters = self.cache.get(self.worker, self.spec.audio.speaker_count)
        self.stop_event = threading.Event()
        self.listener: socket.socket | None = None
        self.threads: list[threading.Thread] = []

    def _send_error(self, sock: socket.socket, exc: DiagnosticError) -> None:
        try:
            send_json_line(sock, {"type": "error", "error": exc.as_dict()})
        except OSError:
            pass

    def _handle_connection(self, connection: socket.socket, address: tuple[str, Any]) -> None:
        try:
            if address[0] != "127.0.0.1":
                return
            reader = LineReader(connection)
            hello = reader.receive(self.spec.limits.io_timeout_ms, phase="worker_hello")
            require_fields(hello, ("type", "protocol_version", "worker_id"), phase="worker_hello")
            if hello["type"] != "hello" or hello["worker_id"] != self.worker.worker_id:
                raise DiagnosticError("E_FIELD", "worker hello does not match this subprocess", field="worker_id")
            if hello["protocol_version"] != PROTOCOL_VERSION:
                raise DiagnosticError("E_FIELD", "unsupported protocol version", field="protocol_version")
            send_json_line(
                connection,
                {
                    "type": "ready",
                    "protocol_version": PROTOCOL_VERSION,
                    "worker_id": self.worker.worker_id,
                    "param_version": self.worker.parameter_version,
                    "param_digest": self.parameters.digest,
                },
            )
            while not self.stop_event.is_set():
                try:
                    message = reader.receive(self.spec.limits.io_timeout_ms, phase="worker_batch")
                except DiagnosticError as exc:
                    if exc.field == "socket.recv" or exc.message.startswith("worker closed"):
                        break
                    raise
                message_type = message.get("type")
                if message_type == "shutdown":
                    send_json_line(connection, {"type": "bye", "worker_id": self.worker.worker_id})
                    self.stop_event.set()
                    break
                if message_type != "batch":
                    raise DiagnosticError("E_FIELD", "worker only accepts batch messages after hello", field="type")
                require_fields(
                    message,
                    (
                        "batch_id",
                        "worker_id",
                        "param_version",
                        "param_digest",
                        "windows",
                    ),
                    phase="worker_batch",
                )
                if message["worker_id"] != self.worker.worker_id:
                    raise DiagnosticError("E_FIELD", "batch was routed to the wrong worker", field="worker_id")
                if message["param_version"] != self.worker.parameter_version:
                    raise DiagnosticError("E_FIELD", "batch parameter version mismatch", field="param_version")
                if message["param_digest"] != self.parameters.digest:
                    raise DiagnosticError("E_FIELD", "batch parameter digest mismatch", field="param_digest")
                if not isinstance(message["windows"], list):
                    raise DiagnosticError("E_FIELD", "windows must be an array", field="windows")
                window_ids, feature_rows = resolve_payload_features(self.worker, message["windows"])
                if "participant_window_ids" in message:
                    if not isinstance(message["participant_window_ids"], list) or list(
                        message["participant_window_ids"]
                    ) != list(window_ids):
                        raise DiagnosticError(
                            "E_FIELD",
                            "participant_window_ids must match the delivered window order",
                            field="participant_window_ids",
                        )
                features = {window_id: row for window_id, row in zip(window_ids, feature_rows)}
                start = time.perf_counter_ns()
                result = infer_batch(
                    self.spec,
                    self.worker,
                    str(message["batch_id"]),
                    window_ids,
                    features,
                    parameter_cache=self.cache,
                    elapsed_ms=0,
                )
                elapsed_ms = max(0, (time.perf_counter_ns() - start) // 1_000_000)
                receipt = dict(result.receipt)
                receipt["elapsed_ms"] = int(elapsed_ms)
                send_json_line(connection, receipt)
        except DiagnosticError as exc:
            self._send_error(connection, exc)
        except OSError:
            pass
        finally:
            try:
                connection.close()
            except OSError:
                pass

    def serve_forever(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener = listener
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self.worker.host, self.worker.port))
            listener.listen(64)
            listener.settimeout(0.2)
            while not self.stop_event.is_set():
                try:
                    connection, address = listener.accept()
                except socket.timeout:
                    continue
                connection.settimeout(self.spec.limits.io_timeout_ms / 1000.0)
                thread = threading.Thread(
                    target=self._handle_connection,
                    args=(connection, address),
                    daemon=True,
                    name=f"connection-{address[1]}",
                )
                thread.start()
                self.threads.append(thread)
        finally:
            try:
                listener.close()
            except OSError:
                pass
            for thread in self.threads:
                thread.join(timeout=1.0)

    def stop(self) -> None:
        self.stop_event.set()
        if self.listener is not None:
            try:
                self.listener.close()
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback inference worker")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args(argv)
    try:
        server = WorkerServer(args.spec, args.worker_id)
    except DiagnosticError as exc:
        print(canonical_bytes(exc.as_dict()).decode("utf-8"), file=sys.stderr)
        return 2

    def stop_handler(signum: int, frame: Any) -> None:
        server.stop()

    for signal_name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        if hasattr(signal, signal_name):
            try:
                signal.signal(getattr(signal, signal_name), stop_handler)
            except (ValueError, OSError):
                pass
    try:
        server.serve_forever()
    except OSError as exc:
        print(canonical_bytes({"code": "E_PORT", "message": str(exc)}).decode("utf-8"), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
