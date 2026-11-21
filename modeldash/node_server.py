"""A deterministic loopback inference node process."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from .inference import forward_summary, request_semantic_digest
from .protocol import PROTOCOL_VERSION, ProtocolError, receive_message, send_message
from .util import canonical_json


@dataclass
class FaultState:
    shard_index: int
    remaining: int
    mode: str


class InferenceNode:
    def __init__(self, node_id: str, port: int, seed: int, delay_ms: int, faults: list[dict[str, Any]]) -> None:
        self.node_id = node_id
        self.port = port
        self.seed = seed
        self.delay_ms = delay_ms
        self.faults = [FaultState(int(item["shard_index"]), int(item["remaining"]), str(item["mode"])) for item in faults]
        self.cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self.lock = threading.Lock()

    def _take_fault(self, shard_index: int, op: str) -> str | None:
        with self.lock:
            for fault in self.faults:
                if fault.remaining <= 0:
                    continue
                if fault.mode == "drop_before_handshake" and op == "hello":
                    fault.remaining -= 1
                    return fault.mode
                if fault.mode in {"drop_before_reply", "error_reply"} and op == "infer" and fault.shard_index == shard_index:
                    fault.remaining -= 1
                    return fault.mode
        return None

    def handle_connection(self, conn: socket.socket) -> None:
        conn.settimeout(120.0)
        try:
            with conn:
                while True:
                    try:
                        request = receive_message(conn)
                    except ProtocolError as exc:
                        send_message(conn, {"op": "error", "code": "E_PROTOCOL", "detail": str(exc)})
                        return
                    op = request.get("op")
                    if op == "hello":
                        fault = self._take_fault(-1, "hello")
                        if fault:
                            return
                        send_message(
                            conn,
                            {
                                "op": "hello_ok",
                                "protocol": PROTOCOL_VERSION,
                                "node_id": self.node_id,
                                "pid": os.getpid(),
                                "seed": self.seed,
                            },
                        )
                        continue
                    if op == "infer":
                        self._handle_infer(conn, request)
                        continue
                    send_message(conn, {"op": "error", "code": "E_PROTOCOL", "detail": f"unknown op {op!r}"})
        except (ConnectionError, OSError, ProtocolError, ProtocolEOF):
            return

    def _handle_infer(self, conn: socket.socket, request: dict[str, Any]) -> None:
        if request.get("protocol") != PROTOCOL_VERSION:
            send_message(conn, {"op": "error", "code": "E_PROTOCOL", "detail": "protocol version mismatch"})
            return
        if request.get("node_id") != self.node_id:
            send_message(conn, {"op": "error", "code": "E_PROTOCOL", "detail": "node id mismatch"})
            return
        shard_index = request.get("shard_index")
        if not isinstance(shard_index, int) or isinstance(shard_index, bool):
            send_message(conn, {"op": "error", "code": "E_PROTOCOL", "detail": "shard_index must be an integer"})
            return
        semantic_digest = request_semantic_digest(request)
        request_id = request.get("request_id")
        if not isinstance(request_id, str):
            send_message(conn, {"op": "error", "code": "E_PROTOCOL", "detail": "request_id must be a string"})
            return
        with self.lock:
            cached = self.cache.get(request_id)
        if cached is not None:
            cached_digest, cached_response = cached
            if cached_digest != semantic_digest:
                send_message(conn, {"op": "error", "code": "E_IDEMPOTENCY", "detail": "request id reused with different semantics"})
                return
            send_message(conn, cached_response)
            return

        fault = self._take_fault(shard_index, "infer")
        if fault == "drop_before_reply":
            return
        if fault == "error_reply":
            send_message(conn, {"op": "error", "code": "E_INJECTED", "detail": "injected inference failure"})
            return

        started = time.perf_counter_ns()
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        items = request.get("items")
        input_obj = request.get("input")
        if not isinstance(items, dict) or not isinstance(input_obj, dict):
            send_message(conn, {"op": "error", "code": "E_PROTOCOL", "detail": "items and input must be objects"})
            return
        try:
            output_summary, keys = forward_summary(
                seed=int(input_obj["seed"]),
                config_version=int(request["config_version"]),
                shard_index=shard_index,
                shard_count=int(request["shard_count"]),
                input_size=int(input_obj["size"]),
                input_scale=int(input_obj["scale"]),
                items=items,
            )
        except (KeyError, TypeError, ValueError) as exc:
            send_message(conn, {"op": "error", "code": "E_PROTOCOL", "detail": f"invalid inference payload: {exc}"})
            return
        compute_ms = max(1, (time.perf_counter_ns() - started) // 1_000_000)
        response = {
            "op": "result",
            "protocol": PROTOCOL_VERSION,
            "request_id": request_id,
            "semantic_digest": semantic_digest,
            "node_id": self.node_id,
            "task_id": request.get("task_id"),
            "config_version": request.get("config_version"),
            "shard_index": shard_index,
            "shard_count": request.get("shard_count"),
            "input_digest": self._input_digest(request),
            "output_summary": output_summary,
            "items_used": keys,
            "compute_ms": int(compute_ms),
        }
        with self.lock:
            self.cache[request_id] = (semantic_digest, response)
        send_message(conn, response)

    @staticmethod
    def _input_digest(request: dict[str, Any]) -> str:
        from .util import digest_json

        return digest_json({"input": request.get("input"), "shard_index": request.get("shard_index")})


def _serve(node: InferenceNode) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE") and os.name == "nt":
        server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", node.port))
    server.listen(64)
    print(canonical_json({"event": "ready", "node_id": node.node_id, "port": node.port, "pid": os.getpid()}), flush=True)
    try:
        while True:
            conn, _ = server.accept()
            thread = threading.Thread(target=node.handle_connection, args=(conn,), daemon=True)
            thread.start()
    finally:
        server.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="deterministic local inference node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--faults-json", default="[]")
    parser.add_argument("--watch-parent", action="store_true")
    args = parser.parse_args(argv)
    try:
        faults = json.loads(args.faults_json)
    except json.JSONDecodeError as exc:
        print(canonical_json({"event": "fatal", "code": "E_NODE_ARGS", "detail": str(exc)}), file=sys.stderr, flush=True)
        return 2
    node = InferenceNode(args.node_id, args.port, args.seed, args.delay_ms, faults)
    if args.watch_parent:
        def watch_parent() -> None:
            try:
                while True:
                    if not sys.stdin.buffer.read(1):
                        os._exit(0)
            except Exception:
                os._exit(0)

        threading.Thread(target=watch_parent, name="parent-watch", daemon=True).start()
    try:
        _serve(node)
    except OSError as exc:
        print(canonical_json({"event": "fatal", "code": "E_PORT_IN_USE", "detail": str(exc)}), file=sys.stderr, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
