from __future__ import annotations

import argparse
import json
import os
import signal
import socketserver
import threading
import time
from pathlib import Path
from typing import Any

from .formats import encode_v3
from .integrity import compute_shard_proof
from .protocol import PROTOCOL_NAME
from .spec import NodeSpec, SpecBundle, load_spec_dir


class NodeState:
    def __init__(self, spec: SpecBundle, node: NodeSpec):
        self.spec = spec
        self.node = node
        self.lock = threading.Lock()
        self.lease_versions: dict[str, int] = {}

    def grant(self, lease_id: str, version: int) -> int:
        with self.lock:
            self.lease_versions[lease_id] = int(version)
            return self.lease_versions[lease_id]

    def renew(self, lease_id: str, expected_version: int) -> int:
        with self.lock:
            current = self.lease_versions.get(lease_id)
            if current is None:
                self.lease_versions[lease_id] = int(expected_version)
                current = int(expected_version)
            if current != expected_version:
                raise PermissionError(f"E_LEASE version mismatch expected={expected_version} actual={current}")
            self.lease_versions[lease_id] = current + 1
            return self.lease_versions[lease_id]


class JSONLineHandler(socketserver.StreamRequestHandler):
    server: "NodeJSONServer"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.timeout_s)

    def _send(self, obj: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(obj, sort_keys=True) + "\n").encode("utf-8"))
        self.wfile.flush()

    def handle(self) -> None:
        authed_node: str | None = None
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                message = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send({"ok": False, "error": {"code": "E_FORMAT", "message": str(exc)}})
                return
            mtype = message.get("type")
            try:
                if mtype == "hello":
                    if message.get("protocol") != PROTOCOL_NAME:
                        raise ValueError("unsupported protocol")
                    authed_node = message.get("node_id")
                    if authed_node != self.server.state.node.node_id:
                        raise PermissionError("node id does not match worker")
                    self._send(
                        {
                            "ok": True,
                            "type": "hello_ack",
                            "protocol": PROTOCOL_NAME,
                            "node_id": authed_node,
                            "quota_version": self.server.state.spec.quota_version,
                        }
                    )
                    continue
                if authed_node is None:
                    raise PermissionError("hello handshake required")

                if mtype == "ping":
                    self._send({"ok": True, "type": "pong", "node_id": authed_node})
                elif mtype == "grant":
                    version = self.server.state.grant(message["lease_id"], int(message["lease_version"]))
                    self._send({"ok": True, "type": "grant_ack", "lease_id": message["lease_id"], "lease_version": version})
                elif mtype == "renew":
                    try:
                        version = self.server.state.renew(message["lease_id"], int(message["expected_version"]))
                    except PermissionError as exc:
                        self._send({"ok": False, "error": {"code": "E_LEASE", "message": str(exc)}})
                        continue
                    self._send({"ok": True, "type": "renew_ack", "lease_id": message["lease_id"], "lease_version": version})
                elif mtype == "verify":
                    response = self._verify(message)
                    if message.get("crash_before_reply"):
                        os.kill(os.getpid(), signal.SIGKILL)
                    self._send(response)
                else:
                    self._send({"ok": False, "error": {"code": "E_PROTOCOL", "message": f"unknown type {mtype}"}})
            except Exception as exc:  # keep worker alive for ordinary request errors
                code = "E_LEASE" if isinstance(exc, PermissionError) else "E_NODE"
                self._send({"ok": False, "error": {"code": code, "message": str(exc)}})

    def _verify(self, message: dict[str, Any]) -> dict[str, Any]:
        state = self.server.state
        content = state.spec.content_for(str(message["content_key"]))
        seq = int(message["shard_seq"])
        started_ns = time.monotonic_ns()
        proof = compute_shard_proof(state.node, content, seq, elapsed_ms=0, exchange_format="v3")
        elapsed_ns = time.monotonic_ns() - started_ns
        elapsed_ms = elapsed_ns // 1_000_000
        object.__setattr__(proof, "elapsed_ms", elapsed_ms)
        invalid_ref = next((ref for ref in content.invalid_refs if ref.shard_id == proof.shard_id), None)
        invalid_obj = None if invalid_ref is None else {"ref_id": invalid_ref.ref_id, "shard_id": invalid_ref.shard_id}
        receipt = encode_v3(
            proof,
            lease_version=int(message.get("lease_version", 0)),
            quota_version=int(message.get("quota_version", state.spec.quota_version)),
            invalid_ref=invalid_obj,
        )
        return {
            "ok": True,
            "type": "verify_ack",
            "node_id": state.node.node_id,
            "request_id": message.get("request_id"),
            "shard_id": proof.shard_id,
            "digest": proof.digest,
            "content_bytes_digest": proof.content_bytes_digest,
            "exchange_format_version": "v3",
            "receipt_v3": receipt,
            "proof": proof.to_canonical(),
            "monotonic_clock": "time.monotonic_ns//1_000_000",
            "elapsed_ms": elapsed_ms,
            "elapsed_ns": elapsed_ns,
        }


class NodeJSONServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: NodeState, timeout_ms: int):
        self.state = state
        self.timeout_s = max(0.1, timeout_ms / 1000.0)
        super().__init__(address, JSONLineHandler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edge cache shard verification node worker")
    parser.add_argument("--spec-dir", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--timeout-ms", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_spec_dir(Path(args.spec_dir))
    node = spec.node(args.node_id)
    timeout_ms = args.timeout_ms or spec.policy.socket_timeout_ms
    server = NodeJSONServer((node.host, node.port), NodeState(spec, node), timeout_ms)
    ready = {
        "event": "ready",
        "node_id": node.node_id,
        "host": node.host,
        "port": node.port,
        "pid": os.getpid(),
        "protocol": PROTOCOL_NAME,
    }
    print(json.dumps(ready, sort_keys=True), flush=True)
    try:
        server.serve_forever(poll_interval=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
