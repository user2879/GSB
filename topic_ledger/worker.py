from __future__ import annotations

import argparse
import json
import socketserver
import threading
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .errors import BusinessError, E_DUPLICATE_DELIVERY, E_STALE_DIGEST, E_WORKER_PROTOCOL
from .sampling import compute_shard_receipt


class NodeState:
    def __init__(self, node_id: str, concurrency_quota: int, lease_limit: int) -> None:
        self.node_id = node_id
        self.concurrency_quota = concurrency_quota
        self.lease_limit = lease_limit
        self.semaphore = threading.BoundedSemaphore(concurrency_quota)
        self.delivered_digests: dict[str, str] = {}
        self.rejections: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def reject(self, batch_id: str | None, shard_id: str | None, code: str, reason: str, **details: Any) -> dict[str, Any]:
        record = {
            "batch_id": batch_id,
            "shard_id": shard_id,
            "code": code,
            "reason": reason,
            "monotonic_order": len(self.rejections),
        }
        if details:
            record["details"] = details
        self.rejections.append(record)
        error = BusinessError(code, reason, batch_id=batch_id, details=details or None)
        return error.to_dict() | {"recorded_rejection": record}


class JsonLineHandler(socketserver.StreamRequestHandler):
    server: "WorkerServer"

    def _send(self, response: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        self.wfile.flush()

    def handle(self) -> None:
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                request = json.loads(line.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._send({"ok": False, "code": E_WORKER_PROTOCOL, "message": str(exc)})
                continue
            self._send(self.server.state.dispatch(request))


class WorkerServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[JsonLineHandler], state: NodeState) -> None:
        super().__init__(address, handler)
        self.state = state


def _dispatch_sample(state: NodeState, request: dict[str, Any]) -> dict[str, Any]:
    batch_id = request.get("batch_id") if isinstance(request.get("batch_id"), str) else None
    shard_id = request.get("shard_id") if isinstance(request.get("shard_id"), str) else None
    if not batch_id:
        return BusinessError(
            "E_INVALID_INPUT", "sample request requires batch_id", field="batch_id"
        ).to_dict()
    if not shard_id:
        return BusinessError(
            "E_INVALID_INPUT", "sample request requires shard_id", field="shard_id", batch_id=batch_id
        ).to_dict()
    if "config" not in request or "vocab_size" not in request:
        return BusinessError(
            "E_INVALID_INPUT",
            "sample request requires config and vocab_size",
            field="config",
            batch_id=batch_id,
        ).to_dict()

    prior_digest = request.get("prior_receipt_digest")
    with state.lock:
        current_digest = state.delivered_digests.get(shard_id)
        if current_digest is not None:
            if prior_digest is None:
                return state.reject(
                    batch_id,
                    shard_id,
                    E_DUPLICATE_DELIVERY,
                    "duplicate shard delivery requires prior_receipt_digest",
                    current_receipt_digest=current_digest,
                )
            if prior_digest != current_digest:
                return state.reject(
                    batch_id,
                    shard_id,
                    E_STALE_DIGEST,
                    "prior_receipt_digest does not match current receipt",
                    current_receipt_digest=current_digest,
                    supplied_digest=prior_digest,
                )

    request_with_node = {**request, "node_id": state.node_id}
    with state.semaphore:
        start_ns = time.monotonic_ns()
        receipt = compute_shard_receipt(request_with_node)
        elapsed_ms = max(0, (time.monotonic_ns() - start_ns) // 1_000_000)
        receipt["elapsed_monotonic_ms"] = int(elapsed_ms)
        with state.lock:
            state.delivered_digests[shard_id] = receipt["receipt_digest"]

    return {"ok": True, "batch_id": batch_id, "receipt": receipt}


def dispatch(state: NodeState, request: dict[str, Any]) -> dict[str, Any]:
    op = request.get("op", "sample")
    if op == "hello":
        return {
            "ok": True,
            "node_id": state.node_id,
            "concurrency_quota": state.concurrency_quota,
            "lease_limit": state.lease_limit,
        }
    if op == "rejections":
        return {"ok": True, "node_id": state.node_id, "rejections": list(state.rejections)}
    if op == "sample":
        return _dispatch_sample(state, request)
    if op == "shutdown":
        threading.Thread(target=request.get("_shutdown_callable", lambda: None), daemon=True).start()
        return {"ok": True, "node_id": state.node_id, "shutdown": True}
    return BusinessError(E_WORKER_PROTOCOL, f"unknown op: {op}", batch_id=request.get("batch_id")).to_dict()


# Attach dispatch as a method-like attribute without changing the pure function signature.
NodeState.dispatch = lambda self, request: dispatch(self, request)  # type: ignore[attr-defined]


def serve(
    *,
    node_id: str,
    host: str,
    port: int,
    concurrency_quota: int,
    lease_limit: int,
    ready_file: str | Path | None = None,
) -> None:
    state = NodeState(node_id, concurrency_quota, lease_limit)
    server = WorkerServer((host, port), JsonLineHandler, state)

    def shutdown_soon() -> None:
        time.sleep(0.05)
        server.shutdown()

    original_dispatch = state.dispatch

    def dispatch_with_shutdown(request: dict[str, Any]) -> dict[str, Any]:
        if request.get("op") == "shutdown":
            response = original_dispatch(request)
            threading.Thread(target=shutdown_soon, daemon=True).start()
            return response
        return original_dispatch(request)

    state.dispatch = dispatch_with_shutdown  # type: ignore[method-assign]
    actual_host, actual_port = server.server_address[:2]
    ready = {
        "event": "ready",
        "node_id": node_id,
        "host": actual_host,
        "port": actual_port,
        "pid": __import__("os").getpid(),
    }
    print(json.dumps(ready, sort_keys=True), flush=True)
    if ready_file is not None:
        Path(ready_file).write_text(canonical_json(ready), encoding="utf-8")
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Topic ledger JSONL worker")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--concurrency-quota", type=int, required=True)
    parser.add_argument("--lease-limit", type=int, required=True)
    parser.add_argument("--ready-file")
    args = parser.parse_args(argv)
    serve(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        concurrency_quota=args.concurrency_quota,
        lease_limit=args.lease_limit,
        ready_file=args.ready_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
