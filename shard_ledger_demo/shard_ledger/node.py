"""Node worker process.

Each node is an independent OS process serving JSON lines over a loopback TCP
socket. One connection carries one optional handshake line and one batch
request; the node replies with one receipt line and closes the connection.
"""

import json
import socket
import sys
import time

from .clock import now_ms
from .compute import aggregate
from .spec import NodeSpec, load_manifest

REQUIRED_REPLY_FIELDS = ("node_id", "shard_id", "config_version", "summary", "duration_ms")


def _read_line(sock: socket.socket) -> dict:
    buffer = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buffer.extend(chunk)
        if b"\n" in chunk:
            break
    line, _, _ = buffer.partition(b"\n")
    if not line:
        return {}
    return json.loads(line.decode("utf-8"))


def serve(node: NodeSpec, revisions: dict, stop_after: int = 0, idle_timeout_s: float = 8.0) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((node.host, node.port))
    server.listen(node.concurrency + 4)
    print(f"NODE_READY {node.node_id} {node.host}:{node.port}", flush=True)

    handled = 0
    last_activity = time.monotonic()
    try:
        while True:
            if stop_after and handled >= stop_after:
                break
            try:
                server.settimeout(0.5)
                conn, _ = server.accept()
                last_activity = time.monotonic()
            except socket.timeout:
                if time.monotonic() - last_activity > idle_timeout_s:
                    break
                continue
            handled += 1
            with conn:
                conn.settimeout(5.0)
                first = _read_line(conn)
                if first.get("op") == "handshake":
                    conn.sendall((json.dumps({"op": "handshake_ack", "node_id": node.node_id}) + "\n").encode())
                    request = _read_line(conn)
                else:
                    request = first
                if request.get("op") != "batch":
                    raise ValueError("expected batch request")
                if node.work_ms:
                    time.sleep(node.work_ms / 1000.0)
                started = now_ms()
                summary = aggregate(
                    node.node_id,
                    request["shard_id"],
                    node.seed,
                    int(request["config_version"]),
                    revisions,
                )
                duration_ms = max(1, now_ms() - started)
                receipt = {
                    "node_id": node.node_id,
                    "shard_id": request["shard_id"],
                    "config_version": int(request["config_version"]),
                    "summary": summary,
                    "duration_ms": duration_ms,
                }
                conn.sendall((json.dumps(receipt) + "\n").encode())
                if node.hold_ms:
                    time.sleep(node.hold_ms / 1000.0)
    finally:
        server.close()


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        print("usage: python -m shard_ledger.node <manifest> <node_id> [stop_after]", file=sys.stderr)
        return 2
    manifest = load_manifest(argv[0])
    node = manifest.nodes[argv[1]]
    stop_after = int(argv[2]) if len(argv) > 2 else 0
    serve(node, manifest.revisions, stop_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
