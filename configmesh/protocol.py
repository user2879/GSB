"""Loopback TCP JSON-lines framing and the node subprocess entry point.

Wire protocol (one JSON object per line, LF terminated):

* platform -> node: {"type": "hello", ...} / {"type": "dispatch", ...} /
  {"type": "shutdown", ...}
* node -> platform: {"type": "hello_ack", ...} / {"type": "receipt", ...} /
  {"type": "error", ...}

All sockets use real ``settimeout`` values; connect/read failures are mapped
by the platform layer to E_PORT / E_HANDSHAKE / E_REPLY.
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from typing import Any

from .errors import BusinessError, ErrorCode
from .fixedpoint import derive_qvalue
from .sample import aggregate_shards
from .util import now_ms


def send_line(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def recv_line(sock: socket.socket, max_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("peer closed connection before line terminator")
        chunks.append(chunk)
        total += len(chunk)
        if b"\n" in chunk:
            break
        if total > max_bytes:
            raise BusinessError(ErrorCode.E_REPLY, "receipt line exceeds max size",
                                [{"max_bytes": max_bytes}])
    raw = b"".join(chunks)
    line, _, _ = raw.partition(b"\n")
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BusinessError(ErrorCode.E_REPLY, "invalid JSON line from node",
                            [{"detail": str(exc)}])
    if not isinstance(message, dict):
        raise BusinessError(ErrorCode.E_REPLY, "protocol message must be a JSON object")
    return message


def _compute_dispatch(message: dict[str, Any], node_spec_bias: int) -> dict[str, Any]:
    started = now_ms()
    node_id = message["node_id"]
    version = message["version"]
    shard_index = int(message["shard_index"])
    shard_count = int(message["shard_count"])
    items = message["items"]
    qvalues_by_key = {key: int(payload["qvalue"]) for key, payload in items.items()}
    derived_state: dict[str, int] = {}
    for key in sorted(items):
        payload = items[key]
        if not payload.get("is_derived"):
            continue
        coeffs = {dep: int(q) for dep, q in payload["derived_from_q"].items()}
        qvalue = derive_qvalue(coeffs, qvalues_by_key, _GridFromScale(message["grid_scale"]))
        derived_state[key] = qvalue + node_spec_bias
    aggregate = aggregate_shards(
        [qvalues_by_key[key] for key in sorted(qvalues_by_key)],
        node_id, shard_count,
    )
    # shard-specific aggregate digest: include shard index for partition identity
    aggregate["shard_index"] = shard_index
    digest_payload = json.dumps(aggregate, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib
    receipt = {
        "type": "receipt",
        "node_id": node_id,
        "version": version,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "derived_state": dict(sorted(derived_state.items())),
        "aggregate_digest": hashlib.sha256(digest_payload).hexdigest(),
        "compute_ms": max(0, now_ms() - started),
    }
    if message.get("timeout_once"):
        time.sleep(max(5.0, message.get("timeout_sleep", 5.0)))
    return receipt


class _GridFromScale:
    def __init__(self, scale: int) -> None:
        self.scale = int(scale)


class NodeServer:
    """Single-process loopback node serving sequential JSON-lines connections."""

    def __init__(self, node_id: str, host: str, port: int, bias: int = 0) -> None:
        self.node_id = node_id
        self.host = host
        self.port = port
        self.bias = bias
        self._stop = threading.Event()
        self._server: socket.socket | None = None

    def serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
        except OSError as exc:
            raise BusinessError(ErrorCode.E_PORT,
                                f"node {self.node_id} failed to bind {self.host}:{self.port}",
                                [{"node_id": self.node_id, "host": self.host,
                                  "port": self.port, "detail": str(exc)}])
        server.listen(8)
        server.settimeout(0.25)
        self._server = server
        while not self._stop.is_set():
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        server.close()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(5.0)
            try:
                message = recv_line(conn)
                kind = message.get("type")
                if kind == "hello":
                    send_line(conn, {"type": "hello_ack", "node_id": self.node_id,
                                     "status": "ready"})
                elif kind == "dispatch":
                    if message.get("node_id") != self.node_id:
                        send_line(conn, {"type": "error", "reason": "node_id_mismatch"})
                        return
                    receipt = _compute_dispatch(message, self.bias)
                    send_line(conn, receipt)
                elif kind == "shutdown":
                    send_line(conn, {"type": "bye", "node_id": self.node_id})
                    self._stop.set()
                else:
                    send_line(conn, {"type": "error", "reason": f"unknown_type:{kind}"})
            except (BusinessError, ConnectionError, OSError, KeyError, ValueError) as exc:
                try:
                    send_line(conn, {"type": "error", "reason": str(exc)})
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()


def node_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="configmesh simulated component node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--bias", type=int, default=0)
    args = parser.parse_args(argv)
    server = NodeServer(args.node_id, args.host, args.port, bias=args.bias)
    try:
        server.serve()
    except BusinessError as exc:
        print(json.dumps(exc.to_dict()), flush=True)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(node_main())
