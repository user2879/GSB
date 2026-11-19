"""Node simulator entry point: runs inside a local child subprocess.

Protocol (one JSON object per line over a loopback TCP socket):

* The child binds 127.0.0.1:0 (or a fixed port) and prints
  ``{"event":"ready","port":P,"node_id":...}`` as the *first* stdout line.
* The platform connects, sends a ``hello`` line, and the child answers
  ``hello_ack`` (handshake).
* Each ``config`` line carries a configuration slice; the child applies the
  values, computes the canonical derived-state summary and answers ``receipt``.
* Re-delivery of the same ``request_id`` returns the cached receipt unchanged
  (single-delivery semantics across retries).

Each accepted connection is handled on its own thread so a request that runs
long never blocks a retry arriving on a fresh connection.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..config.manifest import NodeSpec
from ..util.clock import now_ms
from .summary import node_summary, summary_bytes


def _emit_stdout(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def _child_log(log_path: str | None, event: str, **fields: Any) -> None:
    if not log_path:
        return
    record = {"ts_ms": now_ms(), "event": event, "source": "node-worker"}
    record.update(fields)
    with Path(log_path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _readline(reader: socket.socket) -> dict[str, Any]:
    buffer = bytearray()
    while True:
        chunk = reader.recv(4096)
        if not chunk:
            raise ConnectionError("peer closed connection")
        buffer.extend(chunk)
        if b"\n" in buffer:
            line, _, _ = buffer.partition(b"\n")
            return json.loads(line.decode("utf-8"))


def _send(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n")


class NodeServer:
    def __init__(self, spec: NodeSpec, log_path: str | None, inject: set[str]) -> None:
        self.spec = spec
        self.log_path = log_path
        self.inject = inject
        self.sequence = 0
        self.lock = threading.Lock()
        self.cached_receipts: dict[str, dict[str, Any]] = {}
        self.slow_consumed = False
        self._stop = threading.Event()

    def _next_sequence(self) -> int:
        with self.lock:
            self.sequence += 1
            return self.sequence

    def _handle_config(self, conn: socket.socket, message: dict[str, Any]) -> None:
        request_id = message["request_id"]
        with self.lock:
            cached = self.cached_receipts.get(request_id)
        if cached is not None:
            receipt = dict(cached)
            receipt["duplicate"] = True
            _send(conn, receipt)
            _child_log(self.log_path, "receipt_duplicate", key=self.spec.node_id,
                       request_id=request_id)
            return

        if "slow_first" in self.inject:
            with self.lock:
                should_sleep = not self.slow_consumed
                self.slow_consumed = True
            if should_sleep:
                _child_log(self.log_path, "injected_slow_first", key=self.spec.node_id)
                time.sleep(5.0)
        if "always_slow" in self.inject:
            time.sleep(5.0)

        started = time.perf_counter_ns()
        slice_data = message["slice"]
        version = slice_data["version"]
        grid_q = slice_data["grid_q"]
        item_reals: dict[str, Fraction] = {}
        ordered_keys: list[str] = []
        for entry in slice_data["items"]:
            item_reals[entry["key"]] = Fraction(entry["mantissa"], 10**grid_q)
            ordered_keys.append(entry["key"])
        ordered_keys.sort()

        summary = node_summary(
            node_id=self.spec.node_id,
            version=version,
            aggregate=self.spec.aggregate,
            sample=list(self.spec.sample),
            item_reals=item_reals,
            ordered_keys=ordered_keys,
        )
        elapsed_ms = max(1, (time.perf_counter_ns() - started) // 1_000_000)
        sequence = self._next_sequence()
        receipt = {
            "kind": "receipt",
            "node_id": self.spec.node_id,
            "request_id": request_id,
            "version": version,
            "sequence": sequence,
            "elapsed_ms": int(elapsed_ms),
            "summary": summary,
            "summary_bytes": summary_bytes(summary).decode("utf-8"),
            "duplicate": False,
        }
        with self.lock:
            # Another thread may have computed the same request concurrently;
            # keep the first canonical receipt for idempotent redelivery.
            if request_id not in self.cached_receipts:
                self.cached_receipts[request_id] = receipt
            stored = dict(self.cached_receipts[request_id])
        _send(conn, stored)
        _child_log(self.log_path, "receipt", key=self.spec.node_id,
                   request_id=request_id, sequence=stored["sequence"])

    def _serve_connection(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(30.0)
            try:
                message = _readline(conn)
            except (OSError, json.JSONDecodeError, ConnectionError):
                return
            kind = message.get("kind")
            if kind == "hello":
                if "handshake_fail" in self.inject:
                    _child_log(self.log_path, "injected_handshake_fail", key=self.spec.node_id)
                    time.sleep(8.0)
                    return
                _send(conn, {"kind": "hello_ack", "node_id": self.spec.node_id})
                return
            if kind == "shutdown":
                _send(conn, {"kind": "shutdown_ack", "node_id": self.spec.node_id})
                self._stop.set()
                return
            if kind != "config":
                _send(conn, {"kind": "error", "reason": f"unknown kind {kind}"})
                return
            try:
                self._handle_config(conn, message)
            except (OSError, ConnectionError) as exc:
                _child_log(self.log_path, "config_connection_error",
                           key=self.spec.node_id, reason=str(exc))

    def serve(self, server: socket.socket) -> None:
        server.settimeout(1.0)
        threads: list[threading.Thread] = []
        idle_deadline = time.monotonic() + 60.0
        while not self._stop.is_set() and time.monotonic() < idle_deadline:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            worker = threading.Thread(target=self._serve_connection, args=(conn,), daemon=True)
            worker.start()
            threads.append(worker)
        for worker in threads:
            worker.join(timeout=2.0)


def run(spec: NodeSpec, log_path: str | None, inject: list[str]) -> int:
    inject_set = set(inject)
    if "node_fail" in inject_set:
        _child_log(log_path, "injected_node_fail", key=spec.node_id)
        return 3

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("127.0.0.1", spec.port or 0))
    except OSError as exc:
        _child_log(log_path, "bind_failed", key=spec.node_id, reason=str(exc))
        _emit_stdout({"event": "bind_error", "node_id": spec.node_id, "reason": str(exc)})
        return 4
    server.listen(16)
    port = server.getsockname()[1]
    _emit_stdout({"event": "ready", "port": port, "node_id": spec.node_id})

    try:
        NodeServer(spec, log_path, inject_set).serve(server)
    finally:
        server.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="cfgdist simulated node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--sample", required=True, help="JSON list of floats")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--inject", action="append", default=[])
    args = parser.parse_args(argv)

    spec = NodeSpec(
        node_id=args.node_id,
        aggregate=args.aggregate,
        sample=json.loads(args.sample),
        port=args.port,
    )
    try:
        return run(spec, args.log_path, args.inject)
    except Exception as exc:  # pragma: no cover - defensive child guard
        _emit_stdout({"event": "fatal", "node_id": args.node_id, "reason": str(exc)})
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
