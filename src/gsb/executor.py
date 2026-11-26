from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading
import time
from pathlib import Path
from typing import Any

from .canonical import digest_json
from .execution import evaluate_plan_summary
from .receipts import COST_VECTOR_WIDTH, Receipt, encode_receipt, make_receipt

PROTOCOL = "gsb-exec-v1"
MAX_LINE_BYTES = 64 * 1024 * 1024


class ExecutorState:
    def __init__(self, executor_id: str, max_concurrency: int) -> None:
        self.executor_id = executor_id
        self.max_concurrency = max_concurrency
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._accepted = 0
        self._lock = threading.Lock()

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._slots.acquire(blocking=False):
            return {"type": "error", "code": "E_BUSY", "message": "executor concurrency quota exhausted"}
        try:
            with self._lock:
                self._accepted += 1
            plan_summary = request.get("plan_summary")
            batch = request.get("batch")
            if not isinstance(plan_summary, dict) or not isinstance(batch, dict):
                return {"type": "error", "code": "E_FORMAT", "message": "plan_summary and batch are required objects"}
            plan_id = plan_summary.get("plan_id")
            if not isinstance(plan_id, str) or len(plan_id) < 16:
                return {"type": "error", "code": "E_FORMAT", "message": "plan_summary.plan_id is too narrow", "field": "plan_summary.plan_id"}
            cost_vector = plan_summary.get("cost_vector")
            if (
                not isinstance(cost_vector, list)
                or len(cost_vector) != COST_VECTOR_WIDTH
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in cost_vector)
            ):
                return {"type": "error", "code": "E_FORMAT", "message": "plan_summary.cost_vector must be a non-negative integer vector", "field": "plan_summary.cost_vector"}
            shard_index = request.get("shard_index")
            total_shards = request.get("total_shards")
            shard_id = request.get("shard_id")
            if isinstance(shard_index, bool) or not isinstance(shard_index, int):
                return {"type": "error", "code": "E_FORMAT", "message": "shard_index must be an integer", "field": "shard_index"}
            if isinstance(total_shards, bool) or not isinstance(total_shards, int) or total_shards < 1:
                return {"type": "error", "code": "E_FORMAT", "message": "total_shards must be a positive integer", "field": "total_shards"}
            if not 0 <= shard_index < total_shards:
                return {"type": "error", "code": "E_FORMAT", "message": "shard_index is outside [0,total_shards)", "field": "shard_index"}
            if not isinstance(shard_id, str) or len(shard_id) < 3:
                return {"type": "error", "code": "E_FORMAT", "message": "shard_id is too narrow", "field": "shard_id"}
            lease = request.get("lease")
            if not isinstance(lease, dict):
                return {"type": "error", "code": "E_FORMAT", "message": "lease object is required", "field": "lease"}
            lease_id = lease.get("lease_id")
            if not isinstance(lease_id, str) or not lease_id:
                return {"type": "error", "code": "E_FORMAT", "message": "lease_id is required", "field": "lease.lease_id"}
            deadline = lease.get("deadline_mono_ns")
            if isinstance(deadline, bool) or not isinstance(deadline, int):
                return {"type": "error", "code": "E_FORMAT", "message": "lease deadline must be integer monotonic nanoseconds", "field": "lease.deadline_mono_ns"}

            evaluated = evaluate_plan_summary(plan_summary, batch)
            completed_mono_ns = time.monotonic_ns()
            receipt_id = "rcpt-" + digest_json(
                {
                    "plan_id": plan_id,
                    "shard_id": shard_id,
                    "result_digest": evaluated["result_digest"],
                    "batch_digest": digest_json(batch),
                }
            )[:56]
            receipt = make_receipt(
                receipt_id=receipt_id,
                plan_id=plan_id,
                shard_id=shard_id,
                shard_index=shard_index,
                total_shards=total_shards,
                status="ok",
                row_count=int(evaluated["row_count"]),
                result_digest=evaluated["result_digest"],
                executor_id=self.executor_id,
                cost_vector=tuple(int(value) for value in cost_vector),
                mono_ns=completed_mono_ns,
                lease_id=lease_id,
                lease_state="completed_before_deadline" if completed_mono_ns <= deadline else "completed_after_deadline",
                result=evaluated["result"],
            )
            return {
                "type": "executor_receipt",
                "protocol": PROTOCOL,
                "receipt": encode_receipt(receipt, "q3"),
            }
        finally:
            self._slots.release()


class _ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        state: ExecutorState = self.server.state  # type: ignore[attr-defined]
        self.request.settimeout(30.0)
        while True:
            try:
                raw = self.rfile.readline(MAX_LINE_BYTES + 1)
            except (TimeoutError, OSError):
                return
            if not raw:
                return
            if len(raw) > MAX_LINE_BYTES:
                self._send({"type": "error", "code": "E_FORMAT", "message": "JSON line exceeds maximum width"})
                return
            try:
                request = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send({"type": "error", "code": "E_FORMAT", "message": f"invalid JSON line: {exc}"})
                continue
            if not isinstance(request, dict):
                self._send({"type": "error", "code": "E_FORMAT", "message": "request must be a JSON object"})
                continue
            request_type = request.get("type")
            if request_type == "hello":
                if request.get("protocol") != PROTOCOL:
                    self._send({"type": "error", "code": "E_FORMAT", "message": f"protocol must be {PROTOCOL}", "field": "protocol"})
                    continue
                self._send(
                    {
                        "type": "hello",
                        "protocol": PROTOCOL,
                        "executor_id": state.executor_id,
                        "max_concurrency": state.max_concurrency,
                        "mono_ns": time.monotonic_ns(),
                    }
                )
                continue
            if request_type == "execute_shard":
                if request.get("protocol") != PROTOCOL:
                    self._send({"type": "error", "code": "E_FORMAT", "message": f"protocol must be {PROTOCOL}", "field": "protocol"})
                    continue
                self._send(state.execute(request))
                continue
            self._send({"type": "error", "code": "E_FORMAT", "message": "unknown request type", "field": "type"})

    def _send(self, payload: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8"))
        self.wfile.flush()


def create_server(host: str, port: int, executor_id: str, max_concurrency: int) -> _ThreadingTCPServer:
    if host != "127.0.0.1":
        raise ValueError("executor host must be 127.0.0.1")
    server = _ThreadingTCPServer((host, port), _Handler)
    server.state = ExecutorState(executor_id, max_concurrency)  # type: ignore[attr-defined]
    return server


def run_executor(
    *,
    executor_id: str,
    port: int,
    max_concurrency: int,
    ready_file: str | Path,
    host: str = "127.0.0.1",
) -> None:
    server = create_server(host, port, executor_id, max_concurrency)
    ready_path = Path(ready_file)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    actual_host, actual_port = server.server_address[:2]
    temporary = ready_path.with_suffix(ready_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "executor_id": executor_id,
                "host": actual_host,
                "port": int(actual_port),
                "pid": os.getpid(),
                "max_concurrency": max_concurrency,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, ready_path)
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GSB local TCP executor")
    parser.add_argument("--executor-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--ready-file", required=True)
    args = parser.parse_args(argv)
    run_executor(
        executor_id=args.executor_id,
        host=args.host,
        port=args.port,
        max_concurrency=args.max_concurrency,
        ready_file=args.ready_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
