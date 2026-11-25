from __future__ import annotations

import argparse
import json
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any

from .aggregate import timed_aggregate
from .errors import CloudShardError, SpecError
from .spec import NodeSpec, load_spec


class NodeState:
    def __init__(self, node: NodeSpec) -> None:
        self.node = node
        self.leases: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()

    def grant(self, message: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(message["lease_id"])
        with self.lock:
            lease = {
                "lease_id": lease_id,
                "shard_id": message.get("shard_id"),
                "lease_version": int(message.get("lease_version", 1)),
                "renewal_count": 0,
                "expires_at_ms": int(message.get("expires_at_ms", 0)),
            }
            self.leases[lease_id] = lease
            return {"type": "lease_ack", "ok": True, **lease}

    def renew(self, message: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(message["lease_id"])
        expected = int(message["expected_version"])
        with self.lock:
            lease = self.leases.get(lease_id)
            if lease is None or int(lease["lease_version"]) != expected:
                actual = None if lease is None else int(lease["lease_version"])
                return {
                    "type": "error",
                    "ok": False,
                    "code": "E_LEASE",
                    "field": "lease_version",
                    "error": "node optimistic lease version check failed",
                    "basis": {"expected_version": expected, "actual_version": actual},
                }
            renewal_limit = int(self.node.renewal_limit)
            if int(lease.get("renewal_count", 0)) >= renewal_limit:
                return {
                    "type": "error",
                    "ok": False,
                    "code": "E_LEASE",
                    "field": "renewal_count",
                    "error": "node declared renewal limit exceeded",
                    "basis": {
                        "lease_id": lease_id,
                        "renewal_count": lease.get("renewal_count", 0),
                        "renewal_limit": renewal_limit,
                    },
                }
            lease["lease_version"] = expected + 1
            lease["renewal_count"] = int(lease.get("renewal_count", 0)) + 1
            lease["expires_at_ms"] = int(message.get("new_expires_at_ms", lease["expires_at_ms"]))
            return {"type": "lease_ack", "ok": True, **lease}

    def release(self, message: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(message["lease_id"])
        with self.lock:
            self.leases.pop(lease_id, None)
            return {"type": "lease_ack", "ok": True, "lease_id": lease_id, "released": True}


class ThreadingNodeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[socketserver.BaseRequestHandler], state: NodeState, timeout_ms: int):
        self.node_state = state
        self.socket_timeout_s = max(0.05, timeout_ms / 1000.0)
        super().__init__(address, handler)


class NodeHandler(socketserver.StreamRequestHandler):
    server: ThreadingNodeServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.socket_timeout_s)

    def _send(self, payload: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        self.wfile.flush()

    def _error(self, code: str, message: str, *, field: str | None = None, status: int = 400) -> None:
        payload: dict[str, Any] = {"type": "error", "ok": False, "code": code, "error": message}
        if field:
            payload["field"] = field
        self._send(payload)

    def handle(self) -> None:
        try:
            hello_line = self.rfile.readline()
            if not hello_line:
                return
            hello = json.loads(hello_line.decode("utf-8"))
            if not isinstance(hello, dict) or hello.get("type") != "hello" or hello.get("protocol") != 1:
                self._error("E_FIELD", "first line must be protocol hello", field="hello")
                return
            node = self.server.node_state.node
            self._send(
                {
                    "type": "hello_ack",
                    "protocol": 1,
                    "node_id": node.node_id,
                    "host": node.host,
                    "port": node.port,
                }
            )
            while True:
                line = self.rfile.readline()
                if not line:
                    return
                message = json.loads(line.decode("utf-8"))
                if not isinstance(message, dict):
                    self._error("E_FIELD", "message must be a JSON object", field="message")
                    return
                self._dispatch(message)
        except (TimeoutError, ConnectionError):
            return
        except json.JSONDecodeError as exc:
            self._error("E_FIELD", f"invalid JSON line: {exc}", field="json")
        except Exception as exc:  # pragma: no cover - subprocess boundary guard
            self._error("E_INTERNAL", f"node internal error: {type(exc).__name__}: {exc}")

    def _dispatch(self, message: dict[str, Any]) -> None:
        state = self.server.node_state
        kind = message.get("type")
        if kind == "compute":
            for field_name in ("shard_id", "revision", "samples"):
                if field_name not in message:
                    self._error("E_FIELD", f"compute message missing {field_name!r}", field=field_name)
                    return
            try:
                summary, digest, elapsed_ms = timed_aggregate(
                    message["samples"], state.node, message.get("config") or {}
                )
            except (KeyError, TypeError, ValueError) as exc:
                self._error("E_FIELD", f"invalid samples: {exc}", field="samples")
                return
            self._send(
                {
                    "type": "receipt",
                    "ok": True,
                    "node_id": state.node.node_id,
                    "shard_id": message["shard_id"],
                    "revision": int(message["revision"]),
                    "summary": summary,
                    "summary_digest": digest,
                    "elapsed_ms": elapsed_ms,
                }
            )
            return
        if kind == "lease_grant":
            for field_name in ("lease_id", "lease_version", "expires_at_ms"):
                if field_name not in message:
                    self._error("E_FIELD", f"lease_grant missing {field_name!r}", field=field_name)
                    return
            self._send(state.grant(message))
            return
        if kind == "lease_renew":
            if "lease_id" not in message or "expected_version" not in message:
                self._error("E_FIELD", "lease_renew requires lease_id and expected_version", field="expected_version")
                return
            response = state.renew(message)
            self._send(response)
            return
        if kind == "lease_release":
            if "lease_id" not in message:
                self._error("E_FIELD", "lease_release requires lease_id", field="lease_id")
                return
            self._send(state.release(message))
            return
        self._error("E_FIELD", f"unknown message type {kind!r}", field="type")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CloudShard loopback NumPy node subprocess")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--node-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_spec(args.spec)
        node = spec.node_by_id[args.node_id]
    except (SpecError, KeyError) as exc:
        payload = exc.to_payload() if isinstance(exc, CloudShardError) else {"ok": False, "code": "E_SPEC", "error": str(exc)}
        print(json.dumps({"type": "startup_error", **payload}), flush=True)
        return 2

    state = NodeState(node)
    try:
        server = ThreadingNodeServer((node.host, node.port), NodeHandler, state, spec.socket_timeout_ms)
    except OSError as exc:
        print(
            json.dumps(
                {
                    "type": "startup_error",
                    "ok": False,
                    "code": "E_PORT",
                    "field": "port",
                    "path": f"nodes[node_id={node.node_id}].port",
                    "error": f"cannot bind {node.host}:{node.port}: {exc}",
                }
            ),
            flush=True,
        )
        return 3

    print(
        json.dumps(
            {
                "type": "ready",
                "ok": True,
                "node_id": node.node_id,
                "host": node.host,
                "port": node.port,
                "pid": threading.get_native_id(),
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
