from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
import time
from pathlib import Path
from typing import Any

from .compute import forward_batch
from .models import Batch
from .net import call_worker
from .receipts import attach_encoded_receipts
from .spec import load_bundle


class WorkerServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type, bundle: Any, worker_id: str) -> None:
        super().__init__(address, handler)
        self.bundle = bundle
        self.worker_id = worker_id


class WorkerHandler(socketserver.StreamRequestHandler):
    server: WorkerServer

    def handle(self) -> None:
        line = self.rfile.readline()
        if not line:
            return
        try:
            payload = json.loads(line.decode("utf-8"))
            operation = payload.get("op")
            if operation == "health":
                response = {
                    "ok": True,
                    "worker_id": self.server.worker_id,
                    "pid": os.getpid(),
                    "host": "127.0.0.1",
                }
            elif operation == "forward":
                response = {"ok": True, "result": self._forward(payload["batch"])}
            else:
                response = {"ok": False, "error": f"unknown op: {operation}"}
        except Exception as exc:  # boundary must return a structured line
            response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        data = (json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()

    def _forward(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        bundle = self.server.bundle
        worker = bundle.worker_map()[self.server.worker_id]
        lookup = bundle.request_map()
        batch = Batch(
            batch_id=batch_data["batch_id"],
            worker_id=batch_data["worker_id"],
            batch_index=int(batch_data["batch_index"]),
            arrival_seqs=tuple(int(value) for value in batch_data["arrival_seqs"]),
            request_ids=tuple(str(value) for value in batch_data["request_ids"]),
            stable_ranks={str(key): int(value) for key, value in batch_data.get("stable_ranks", {}).items()},
        )
        started_ns = time.perf_counter_ns()
        result = forward_batch(worker, batch, lookup)
        result["elapsed_ms"] = max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
        result["worker_pid"] = os.getpid()
        return attach_encoded_receipts(result)


def serve(spec_dir: str | Path, worker_id: str) -> None:
    bundle = load_bundle(spec_dir)
    worker = bundle.worker_map()[worker_id]
    with WorkerServer((worker.host, worker.port), WorkerHandler, bundle, worker_id) as server:
        print(
            json.dumps({"ready": True, "worker_id": worker_id, "host": worker.host, "port": worker.port, "pid": os.getpid()}),
            flush=True,
        )
        server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback inference worker")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args(argv)
    try:
        serve(args.spec, args.worker_id)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
