from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .errors import DiagnosticError, LeaseConflictError, PortInUseError, SpecError
from .samples import LossAccumulator
from .spec import load_spec_bundle


class NodeServer:
    def __init__(self, node: dict[str, Any]) -> None:
        self.node = node
        self.node_id = node["node_id"]
        self.host = node["host"]
        self.port = int(node["port"])
        self.timeout_seconds = int(node["socket_timeout_ms"]) / 1000.0
        self.server: socket.socket | None = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.lease_versions: dict[str, int] = {}
        self.lease_deadlines_ns: dict[str, int] = {}
        self.active_jobs = 0
        concurrency = max(0, int(node["concurrency"]))
        self.execution_slots = threading.BoundedSemaphore(concurrency) if concurrency > 0 else None

    def _bind(self) -> socket.socket:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
        except OSError as exc:
            server.close()
            raise PortInUseError(
                f"节点 {self.node_id} 绑定 {self.host}:{self.port} 失败",
                entry=self.node_id,
                field="port",
                details={"errno": getattr(exc, "errno", None)},
            ) from exc
        server.listen(16)
        server.settimeout(0.2)
        self.server = server
        return server

    def stop(self) -> None:
        self.stop_event.set()
        if self.server is not None:
            try:
                self.server.close()
            except OSError:
                pass

    def serve_forever(self) -> None:
        server = self._bind()
        ready = {"type": "node_ready", "node_id": self.node_id, "host": self.host, "port": self.port}
        print(json.dumps(ready, sort_keys=True), flush=True)
        while not self.stop_event.is_set():
            try:
                conn, address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                if self.stop_event.is_set():
                    break
                raise
            thread = threading.Thread(target=self._safe_handle, args=(conn,), daemon=True)
            thread.start()

    def _send(self, conn: socket.socket, message: dict[str, Any]) -> None:
        conn.sendall(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")

    def _read(self, conn: socket.socket, buffer: Any) -> dict[str, Any]:
        conn.settimeout(self.timeout_seconds)
        line = buffer.readline()
        if not line:
            raise ConnectionError("connection closed")
        message = json.loads(line.decode("utf-8"))
        if not isinstance(message, dict):
            raise ValueError("message must be object")
        return message

    def _safe_handle(self, conn: socket.socket) -> None:
        try:
            self._handle(conn)
        except Exception as exc:  # node must report deterministic JSON error when possible
            try:
                code = getattr(exc, "code", "E_NODE")
                self._send(conn, {"type": "error", "code": code, "message": str(exc)})
            except OSError:
                pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _handle(self, conn: socket.socket) -> None:
        buffer = conn.makefile("rb")
        hello = self._read(conn, buffer)
        if hello.get("type") != "hello" or hello.get("node_id") != self.node_id:
            self._send(conn, {"type": "error", "code": "E_SPEC", "message": "node_id handshake mismatch"})
            return
        self._send(conn, {"type": "hello_ack", "node_id": self.node_id, "protocol_version": 1})

        message = self._read(conn, buffer)
        if message.get("type") == "renew":
            self._handle_renew(conn, message)
            return
        if message.get("type") != "start":
            self._send(conn, {"type": "error", "code": "E_PROTO", "message": "expected start"})
            return

        shard_id = str(message["shard_id"])
        quota_version = int(message["quota_version"])
        lease_version = int(message.get("lease_version", 1))
        delay_ms = int(message.get("delay_ms", self.node.get("processing_delay_ms", 0)))
        if self.execution_slots is None:
            self._send(conn, {"type": "error", "code": "E_LEASE", "message": "node concurrency quota is zero"})
            return
        self._send(conn, {"type": "start_ack", "shard_id": shard_id})
        with self.execution_slots:
          with self.lock:
            self.active_jobs += 1
            self.lease_versions[shard_id] = lease_version
            self.lease_deadlines_ns[shard_id] = time.monotonic_ns() + int(message.get("ttl_ms", 30000)) * 1_000_000

          started_ns = time.monotonic_ns()
          accumulator = LossAccumulator(self.node, shard_id, quota_version)
          expected_rows = int(message.get("row_count", -1))
          accepted_rows = 0
          batch_count = int(message.get("batch_count", 0))
          delayed = False

          try:
            while True:
                batch = self._read(conn, buffer)
                kind = batch.get("type")
                if kind == "finish":
                    break
                if kind != "batch":
                    self._send(conn, {"type": "error", "code": "E_PROTO", "message": "expected batch or finish"})
                    return
                if delay_ms and not delayed:
                    time.sleep(delay_ms / 1000.0)
                    delayed = True
                rows = batch.get("rows", [])
                if not isinstance(rows, list):
                    raise SpecError("batch.rows 必须是数组", entry=shard_id, field="rows")
                accepted_rows += accumulator.add_rows(rows)
                self._send(
                    conn,
                    {
                        "type": "batch_ack",
                        "shard_id": shard_id,
                        "batch_index": int(batch.get("batch_index", -1)),
                        "accepted_rows": len(rows),
                    },
                )

            finish = message
            if finish.get("type") != "finish":
                finish = batch  # last read was finish in normal flow? Actually loop reads and breaks on finish.
            if expected_rows >= 0 and accepted_rows != expected_rows:
                raise SpecError(
                    f"接收到的行数 {accepted_rows} 与 start.row_count {expected_rows} 不一致",
                    entry=shard_id,
                    field="row_count",
                )
            summary = accumulator.summary()
            elapsed_ns = time.monotonic_ns() - started_ns
            receipt = {
                "type": "receipt",
                "node_id": self.node_id,
                "shard_id": shard_id,
                "quota_version": quota_version,
                "lease_version": lease_version,
                "loss_summary": summary,
                "monotonic_ms": elapsed_ns // 1_000_000,
                "monotonic_ns": elapsed_ns,
                "clock": "time.monotonic_ns",
            }
            self._send(conn, receipt)
          finally:
              with self.lock:
                  self.active_jobs = max(0, self.active_jobs - 1)

    def _handle_renew(self, conn: socket.socket, message: dict[str, Any]) -> None:
        shard_id = str(message.get("shard_id", ""))
        expected_version = int(message.get("expected_version", -1))
        ttl_ms = int(message.get("ttl_ms", 30000))
        with self.lock:
            current = self.lease_versions.get(shard_id)
            if current is None or current != expected_version:
                raise LeaseConflictError(
                    "租约版本乐观校验失败，不延长到期",
                    entry=shard_id,
                    field="lease_version",
                    details={"expected": expected_version, "current": current},
                )
            new_version = current + 1
            self.lease_versions[shard_id] = new_version
            self.lease_deadlines_ns[shard_id] = time.monotonic_ns() + ttl_ms * 1_000_000
            expires_ns = self.lease_deadlines_ns[shard_id]
        self._send(
            conn,
            {
                "type": "renew_ack",
                "shard_id": shard_id,
                "lease_version": new_version,
                "expires_monotonic_ns": expires_ns,
            },
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shard inference loopback node")
    parser.add_argument("--root", default=".", help="project root containing declarations/")
    parser.add_argument("--node-id", required=True)
    args = parser.parse_args(argv)
    try:
        bundle = load_spec_bundle(Path(args.root))
        if args.node_id not in bundle.node_by_id:
            raise SpecError("node_id 不存在", entry=args.node_id, field="node_id")
        server = NodeServer(bundle.node_by_id[args.node_id])
        server.serve_forever()
        return 0
    except DiagnosticError as exc:
        print(json.dumps({"type": "fatal", **exc.to_dict()}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
