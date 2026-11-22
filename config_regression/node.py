"""Local simulated component node.

Each node runs as an independent OS subprocess, binds a 127.0.0.1 loopback
TCP socket and speaks line-delimited JSON:

request (one JSON object + ``\\n``)::

    {"request_id": "...", "node_id": "node-a", "manifest_version": 3,
     "base_seed": 1, "sample_count": 2000, "group_count": 8,
     "aggregations": ["sum", "mean"],
     "items": [{"key": "k", "value": 1.25, "scale_value": 1.25,
                "quantized_value": 1250000}]}

response (one JSON object + ``\\n``)::

    {"request_id": "...", "node_id": "node-a", "manifest_version": 3,
     "results": [{"key": "k", "summary": {...}, "elapsed_ms": 3}],
     "status": "ok"}

Fault modes (declared in the registry) let tests exercise retries without any
network simulation shim:

* ``drop_first``  – silently close the first N attempts (fault_arg), then work
* ``drop_every``  – close every fault_arg-th attempt (fault_arg > 1)
* ``drop_all``    – never answer (used for E_EXHAUST tests)
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass

from .sampling import deterministic_elapsed_ms, run_aggregation


@dataclass
class NodeConfig:
    node_id: str
    host: str
    port: int
    work_ms_min: int
    work_ms_max: int
    fault_mode: str = "none"
    fault_arg: int = 0


class NodeServer:
    def __init__(self, config: NodeConfig):
        self.config = config
        self._attempts = 0
        self._sock: socket.socket | None = None
        self.bound_port = 0

    def serve(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.config.host, self.config.port))
        sock.listen(16)
        self._sock = sock
        self.bound_port = sock.getsockname()[1]
        # Readiness line consumed by the parent process.
        sys.stdout.write(json.dumps({
            "event": "ready",
            "node_id": self.config.node_id,
            "host": self.config.host,
            "port": self.bound_port,
        }) + "\n")
        sys.stdout.flush()
        try:
            while True:
                conn, _ = sock.accept()
                self._handle(conn)
        except OSError:
            return

    def _should_drop(self) -> bool:
        self._attempts += 1
        mode = self.config.fault_mode
        if mode == "drop_all":
            return True
        if mode == "drop_first":
            return self._attempts <= max(1, self.config.fault_arg)
        if mode == "drop_every" and self.config.fault_arg > 1:
            return self._attempts % self.config.fault_arg == 0
        return False

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            buffer = b""
            while b"\n" not in buffer:
                chunk = conn.recv(65536)
                if not chunk:
                    # Client vanished or fault injection closes silently.
                    return
                buffer += chunk
            if self._should_drop():
                return
            line, _, _ = buffer.partition(b"\n")
            try:
                request = json.loads(line.decode("utf-8"))
                response = self._process(request)
            except (ValueError, KeyError, TypeError) as exc:
                response = {"status": "error",
                            "node_id": self.config.node_id,
                            "error": f"bad request: {exc}"}
            conn.sendall((json.dumps(response, sort_keys=True) + "\n")
                         .encode("utf-8"))

    def _process(self, request: dict) -> dict:
        results = []
        for item in request["items"]:
            scale_value = float(item.get("scale_value", item["value"]))
            summary = run_aggregation(
                base_seed=int(request["base_seed"]),
                node_id=self.config.node_id,
                item_key=item["key"],
                sample_count=int(request["sample_count"]),
                group_count=int(request["group_count"]),
                aggregations=tuple(request["aggregations"]),
                scale_value=scale_value,
            )
            elapsed = deterministic_elapsed_ms(
                self.config.node_id, item["key"],
                int(request["manifest_version"]),
                str(request["request_id"]),
                self.config.work_ms_min, self.config.work_ms_max)
            self._busy_wait(min(elapsed, 8))
            results.append({
                "key": item["key"],
                "summary": summary,
                "elapsed_ms": elapsed,
                "quantized_value": item.get("quantized_value"),
            })
        return {
            "status": "ok",
            "request_id": request["request_id"],
            "node_id": self.config.node_id,
            "manifest_version": request["manifest_version"],
            "results": results,
        }

    @staticmethod
    def _busy_wait(ms: int) -> None:
        if ms <= 0:
            return
        deadline = time.perf_counter() + ms / 1000.0
        while time.perf_counter() < deadline:
            pass

    def shutdown(self) -> None:
        if self._sock is not None:
            self._sock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="simulated component node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--work-ms-min", type=int, default=1)
    parser.add_argument("--work-ms-max", type=int, default=4)
    parser.add_argument("--fault-mode", default="none")
    parser.add_argument("--fault-arg", type=int, default=0)
    args = parser.parse_args(argv)
    server = NodeServer(NodeConfig(
        node_id=args.node_id, host=args.host, port=args.port,
        work_ms_min=args.work_ms_min, work_ms_max=args.work_ms_max,
        fault_mode=args.fault_mode, fault_arg=args.fault_arg))
    try:
        server.serve()
    except KeyboardInterrupt:  # pragma: no cover - subprocess control path
        server.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
