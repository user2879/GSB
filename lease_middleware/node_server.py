"""Version-tag authority running as a real JSON-line TCP subprocess."""

from __future__ import annotations

import argparse
import json
import socketserver
import sys
import threading
import time
from typing import Any

from .timeutil import monotonic_ms


class NodeTagState:
    def __init__(self, node_id: str, process_delay_ms: int = 0) -> None:
        self.node_id = node_id
        self.process_delay_ms = process_delay_ms
        self.lock = threading.RLock()
        self.leases: dict[str, dict[str, Any]] = {}

    def reply(self, op: str, **kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": True, "op": op, "node_id": self.node_id}
        payload.update(kwargs)
        return payload

    def error(
        self, op: str, code: str, message: str, *, current_tag: int = 0, **extra: Any
    ) -> dict[str, Any]:
        payload = {
            "ok": False,
            "op": op,
            "node_id": self.node_id,
            "code": code,
            "message": message,
            "current_tag": current_tag,
        }
        payload.update(extra)
        return payload

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op == "handshake":
            return self.reply(
                "handshake",
                protocol_version=1,
                status="ready",
                current_tags=self.current_tags(),
                server_monotonic_ms=monotonic_ms(),
            )
        if self.process_delay_ms:
            time.sleep(self.process_delay_ms / 1000.0)
        if op == "grant":
            return self._grant(request)
        if op in {"renew", "release", "expire"}:
            return self._tagged(str(op), request)
        return self.error(str(op), "E_REPLY", "unsupported op", current_tag=0)

    def current_tags(self) -> dict[str, int]:
        with self.lock:
            return {lease_id: int(item["tag"]) for lease_id, item in self.leases.items()}

    def _grant(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        if not isinstance(lease_id, str) or not lease_id:
            return self.error("grant", "E_REPLY", "missing lease_id", current_tag=0)
        with self.lock:
            existing = self.leases.get(lease_id)
            if existing is not None:
                return self.error(
                    "grant", "E_LEASE", "lease already exists", current_tag=int(existing["tag"])
                )
            self.leases[lease_id] = {"tag": 1, "state": "granted"}
            return self.reply("grant", lease_id=lease_id, current_tag=1, state="granted")

    def _tagged(self, op: str, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        requested_tag = request.get("version_tag")
        if not isinstance(lease_id, str) or not lease_id:
            return self.error(op, "E_REPLY", "missing lease_id", current_tag=0)
        if not isinstance(requested_tag, int) or isinstance(requested_tag, bool):
            with self.lock:
                current = self.leases.get(lease_id, {}).get("tag", 0)
            return self.error(op, "E_REPLY", "missing integer version_tag", current_tag=current)

        with self.lock:
            lease = self.leases.get(lease_id)
            current_tag = int(lease["tag"]) if lease else 0
            if requested_tag != current_tag:
                return self.error(
                    op,
                    "E_LEASE",
                    "version tag does not match node current tag",
                    current_tag=current_tag,
                    requested_tag=requested_tag,
                )
            if lease is None:
                return self.error(op, "E_LEASE", "unknown lease", current_tag=0)

            state = lease["state"]
            valid = {
                "renew": {"granted", "renewed"},
                "release": {"granted", "renewed"},
                "expire": {"granted", "renewed"},
            }[op]
            if state not in valid:
                return self.error(
                    op,
                    "E_LEASE",
                    f"cannot {op} lease in state {state}",
                    current_tag=current_tag,
                )
            if op == "renew":
                lease["state"] = "renewed"
            elif op == "release":
                lease["state"] = "released"
            else:
                lease["state"] = "expired"
            lease["tag"] = current_tag + 1
            return self.reply(
                op,
                lease_id=lease_id,
                current_tag=int(lease["tag"]),
                previous_tag=current_tag,
                state=lease["state"],
            )


class JSONLineHandler(socketserver.StreamRequestHandler):
    server: "JSONLineServer"

    def handle(self) -> None:
        request: dict[str, Any] = {}
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                parsed = json.loads(line.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise ValueError("request is not an object")
                request = parsed
                reply = self.server.state.dispatch(request)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                reply = self.server.state.error(
                    "unknown", "E_REPLY", f"invalid JSON-line request: {exc}", current_tag=0
                )
            self.wfile.write((json.dumps(reply, separators=(",", ":")) + "\n").encode())
            self.wfile.flush()


class JSONLineServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type,
        state: NodeTagState,
    ) -> None:
        self.state = state
        super().__init__(server_address, handler)


def serve(node_id: str, host: str, port: int, process_delay_ms: int = 0) -> None:
    state = NodeTagState(node_id, process_delay_ms=process_delay_ms)
    try:
        server = JSONLineServer((host, port), JSONLineHandler, state)
    except OSError as exc:
        payload = {
            "ok": False,
            "code": "E_PORT",
            "field": "port",
            "message": f"cannot bind {host}:{port}: {exc.strerror or exc}",
            "details": {"host": host, "port": port, "errno": exc.errno},
        }
        print(json.dumps(payload), file=sys.stderr, flush=True)
        raise SystemExit(2) from exc

    print(
        json.dumps(
            {
                "event": "listening",
                "ok": True,
                "node_id": node_id,
                "host": host,
                "port": port,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Versioned lease cache node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--process-delay-ms", type=int, default=0)
    args = parser.parse_args(argv)
    serve(args.node_id, args.host, args.port, args.process_delay_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
