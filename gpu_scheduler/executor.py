from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import digest_obj


CALIBER = {
    "name": "numpy-int64-nonnegative-floor-dot",
    "version": 1,
    "fixed_scale": 1000,
    "dtype": "int64",
    "truncation": "floor_down",
    "determinism": "seeded_pcg64_and_fixed_final_result",
}


def integer_fixed_compute(seed: int, compute_ms: int, integer_units: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    length = int(max(128, min(8192, integer_units * 128)))
    left = rng.integers(0, 10, size=length, dtype=np.int64)
    right = rng.integers(0, 10, size=length, dtype=np.int64)

    start_ns = time.monotonic_ns()
    deadline_ns = start_ns + max(0, int(compute_ms)) * 1_000_000
    warmup_sum = 0
    busy_iterations = 0
    while time.monotonic_ns() < deadline_ns:
        warmup_sum += int(np.dot(left, right))
        if busy_iterations & 31 == 31:
            left = np.roll(left, 1)
        busy_iterations += 1

    dot = int(np.dot(left, right))
    scale = CALIBER["fixed_scale"]
    fixed_mean = int(np.floor_divide(np.int64(dot * scale), np.int64(length)))
    checksum = int(digest_obj([dot, length, fixed_mean, seed])[:16], 16)
    return {
        "vector_length": length,
        "dot": dot,
        "fixed_mean": fixed_mean,
        "checksum": checksum,
        "warmup_sum": int(warmup_sum),
        "busy_iterations": busy_iterations,
        "numpy_version": np.__version__,
    }


def execute_shard(request: dict[str, Any]) -> dict[str, Any]:
    shard_id = str(request["shard_id"])
    started = time.monotonic_ns()
    resource_usage = {
        "gpu_count": int(request.get("gpu_count", 1)),
        "memory_mib": int(request.get("memory_mib", 0)),
        "integer_units": int(request.get("integer_units", 0)),
        "executor_pid": __import__("os").getpid(),
        "host": "127.0.0.1",
    }
    if request.get("force_failure"):
        elapsed_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
        deterministic_payload = {
            "shard_id": shard_id,
            "status": "failed",
            "error_code": "E_SHARD_INJECTED",
            "caliber": CALIBER,
            "resource_usage": resource_usage,
            "compute": None,
        }
        outcome_digest = digest_obj(deterministic_payload)
        receipt = {
            **deterministic_payload,
            "elapsed_monotonic_ms": int(elapsed_ms),
            "outcome_digest": outcome_digest,
        }
        receipt["receipt_digest"] = digest_obj({k: v for k, v in receipt.items() if k != "receipt_digest"})
        return receipt

    compute = integer_fixed_compute(
        int(request.get("seed", 0)),
        int(request.get("compute_ms", 0)),
        int(request.get("integer_units", 1)),
    )
    elapsed_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
    deterministic_payload = {
        "shard_id": shard_id,
        "status": "success",
        "error_code": None,
        "caliber": CALIBER,
        "resource_usage": resource_usage,
        "compute": compute,
    }
    outcome_digest = digest_obj(deterministic_payload)
    receipt = {
        **deterministic_payload,
        "elapsed_monotonic_ms": int(elapsed_ms),
        "outcome_digest": outcome_digest,
    }
    receipt["receipt_digest"] = digest_obj({k: v for k, v in receipt.items() if k != "receipt_digest"})
    return receipt


def execute_batch(batch: dict[str, Any]) -> dict[str, Any]:
    results = [execute_shard(item) for item in batch.get("shards", [])]
    return {
        "batch_id": batch.get("batch_id"),
        "executor_pid": __import__("os").getpid(),
        "caliber": CALIBER,
        "results": results,
    }


class ExecutorHandler(BaseHTTPRequestHandler):
    server_version = "GpuSchedulerExecutor/1.0"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "pid": __import__("os").getpid(), "caliber": CALIBER})
        else:
            self._send_json(404, {"error_code": "E_NOT_FOUND", "path": self.path})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/execute":
            self._send_json(404, {"error_code": "E_NOT_FOUND", "path": self.path})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            batch = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(batch, dict) or not isinstance(batch.get("shards"), list):
                raise ValueError("batch must contain a shards array")
            self._send_json(200, execute_batch(batch))
        except Exception as exc:  # deterministic boundary for malformed batches
            self._send_json(400, {"error_code": "E_EXECUTOR_BATCH", "message": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        record = {"event": "executor_http", "message": fmt % args}
        print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)


def watch_stdin_and_shutdown(server: ThreadingHTTPServer) -> None:
    try:
        while sys.stdin.buffer.read(4096):
            pass
    except Exception:
        pass
    finally:
        threading.Thread(target=server.shutdown, daemon=True).start()


def serve(host: str, port: int) -> None:
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((host, port), ExecutorHandler)
    server.daemon_threads = True
    watcher = threading.Thread(target=watch_stdin_and_shutdown, args=(server,), daemon=True)
    watcher.start()
    print(json.dumps({"event": "executor_listening", "host": host, "port": port, "pid": __import__("os").getpid()}), flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback shard executor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args(argv)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
