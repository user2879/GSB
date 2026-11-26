"""Standalone loopback JSON-lines aggregation worker."""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import threading
import time
from typing import Any

from .errors import GSBError, ProtocolError
from .protocol import JsonLineChannel, PROTOCOL_VERSION, require_fields
from .samples import aggregate_groups, canonical_json_bytes, sha256_hex

_STOP = threading.Event()


def _elapsed_ms(start_ns: int) -> int:
    elapsed = (time.monotonic_ns() - start_ns) // 1_000_000
    return max(0, int(elapsed))


def _handle_hello(message: dict[str, Any], node_id: str, fixed_scale: int) -> dict[str, Any]:
    require_fields(message, ("op", "request_id"), stage="hello_request")
    requested_node = message.get("node_id")
    if requested_node is not None and requested_node != node_id:
        raise ProtocolError(
            "hello requested the wrong node",
            details={"expected": node_id, "actual": requested_node},
        )
    return {
        "ok": True,
        "op": "hello_ack",
        "protocol_version": PROTOCOL_VERSION,
        "node_id": node_id,
        "fixed_scale": fixed_scale,
        "request_id": message["request_id"],
    }


def _handle_aggregate(
    message: dict[str, Any],
    node_id: str,
    fixed_scale: int,
) -> dict[str, Any]:
    required = (
        "op",
        "request_id",
        "node_id",
        "shard_id",
        "capacity_version",
        "fixed_scale",
        "batch_index",
        "batch_count",
        "samples",
        "group_count",
    )
    require_fields(message, required, stage="aggregate_request")
    if message["node_id"] != node_id:
        raise ProtocolError(
            "request node_id does not match this worker",
            details={"expected": node_id, "actual": message["node_id"]},
        )
    if message["fixed_scale"] != fixed_scale:
        raise ProtocolError(
            "request fixed_scale does not match this worker declaration",
            details={"expected": fixed_scale, "actual": message["fixed_scale"]},
        )
    if not isinstance(message["samples"], list):
        raise ProtocolError("samples must be an array")
    if not isinstance(message["group_count"], int) or isinstance(message["group_count"], bool):
        raise ProtocolError("group_count must be an integer")
    try:
        aggregate = aggregate_groups(message["samples"], message["group_count"], fixed_scale)
    except (ValueError, OverflowError) as exc:
        raise ProtocolError("invalid sample batch", details={"error": str(exc)}) from exc
    stage_payload = {
        "node_id": node_id,
        "shard_id": message["shard_id"],
        "capacity_version": message["capacity_version"],
        "batch_index": message["batch_index"],
        "group_count": message["group_count"],
        **aggregate,
    }
    return {
        "ok": True,
        "op": "aggregate_ack",
        "node_id": node_id,
        "shard_id": message["shard_id"],
        "capacity_version": message["capacity_version"],
        "fixed_scale": fixed_scale,
        "request_id": message["request_id"],
        "batch_index": message["batch_index"],
        "batch_count": message["batch_count"],
        "participating_rows": aggregate["participating_rows"],
        "groups": aggregate["groups"],
        "aggregate_digest": sha256_hex(canonical_json_bytes(stage_payload)),
        "sample_batch_digest": sha256_hex(canonical_json_bytes(stage_payload)),
    }


def _serve_connection(
    connection: socket.socket,
    node_id: str,
    fixed_scale: int,
    idle_timeout_s: float,
) -> None:
    channel = JsonLineChannel(connection)
    try:
        while not _STOP.is_set():
            try:
                message = channel.receive(idle_timeout_s, stage="node_read")
            except Exception:
                return
            started_ns = time.monotonic_ns()
            try:
                op = message.get("op")
                if op == "hello":
                    response = _handle_hello(message, node_id, fixed_scale)
                elif op == "aggregate":
                    response = _handle_aggregate(message, node_id, fixed_scale)
                else:
                    raise ProtocolError("unknown operation", details={"op": op})
                response["elapsed_ms"] = _elapsed_ms(started_ns)
            except GSBError as exc:
                response = {"ok": False, **exc.as_dict(), "elapsed_ms": _elapsed_ms(started_ns)}
            except Exception as exc:  # defensive boundary for a child process
                response = {
                    "ok": False,
                    "code": "E_PROTO",
                    "message": "unhandled worker request error",
                    "details": {"error": str(exc)},
                    "elapsed_ms": _elapsed_ms(started_ns),
                }
            try:
                channel.send(response)
            except GSBError:
                return
    finally:
        channel.close()


def run_worker(node_id: str, host: str, port: int, fixed_scale: int, idle_timeout_s: float) -> int:
    if host != "127.0.0.1":
        print(
            "GSB_ERROR " + json.dumps({"code": "E_SPEC", "message": "host must be 127.0.0.1"}),
            flush=True,
        )
        return 2
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((host, port))
        listener.listen(128)
    except OSError as exc:
        print(
            "GSB_ERROR "
            + json.dumps(
                {
                    "code": "E_PORT",
                    "message": "cannot bind declared loopback port",
                    "details": {"host": host, "port": port, "error": str(exc)},
                },
                sort_keys=True,
            ),
            flush=True,
        )
        listener.close()
        return 3
    listener.settimeout(0.2)
    print(
        "GSB_READY "
        + json.dumps(
            {"node_id": node_id, "host": host, "port": port, "fixed_scale": fixed_scale},
            sort_keys=True,
        ),
        flush=True,
    )
    threads: list[threading.Thread] = []
    try:
        while not _STOP.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(
                target=_serve_connection,
                args=(connection, node_id, fixed_scale, idle_timeout_s),
                daemon=True,
            )
            thread.start()
            threads.append(thread)
            threads = [candidate for candidate in threads if candidate.is_alive()]
    finally:
        listener.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GSB loopback aggregation node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--fixed-scale", required=True, type=int)
    parser.add_argument("--idle-timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    def stop(_signum: int, _frame: Any) -> None:
        _STOP.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return run_worker(args.node_id, args.host, args.port, args.fixed_scale, args.idle_timeout)


if __name__ == "__main__":
    sys.exit(main())
