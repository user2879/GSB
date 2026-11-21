"""Configuration consumer node.

The node runs as an independent OS child process.  It binds one 127.0.0.1 TCP
server, speaks the JSON-lines protocol, keeps an in-memory integer-key view and
computes NumPy aggregates over the deterministic revision samples.

The node never persists anything: durability is the platform's job.  After a
crash the platform rebuilds a node through a full ``replace_view``.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

from .clock import is_expired, now_ms
from .errors import (
    KeyNotCoveredError,
    LeaseExpiredError,
    RevisionOutOfRangeError,
)
from .protocol import read_message, write_message
from .sample import summary_for_revision


class NodeState:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.view: dict[str, int] = {}
        self.applied_rev = 0
        self.leases: dict[str, dict[str, Any]] = {}

    def handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        mtype = msg.get("type")
        try:
            if mtype == "ping":
                return {"type": "pong", "node_id": self.node_id, "now_ms": now_ms()}
            if mtype == "lease_grant":
                return self._grant(msg)
            if mtype == "apply":
                return self._apply(msg)
            if mtype == "replace_view":
                return self._replace_view(msg)
            return _error("E_UNKNOWN", f"unknown message type {mtype!r}")
        except Exception as exc:  # noqa: BLE001 - surfaced as protocol error.
            code = getattr(exc, "code", "E_UNKNOWN")
            return _error(code, str(exc), **getattr(exc, "context", {}))

    def _grant(self, msg: dict[str, Any]) -> dict[str, Any]:
        lease = msg["lease"]
        self.leases[lease["lease_id"]] = {
            "lease_id": lease["lease_id"],
            "low_seq": int(lease["low_seq"]),
            "high_seq": int(lease["high_seq"]),
            "keys": set(lease["keys"]),
            "deadline_ms": int(lease["deadline_ms"]),
        }
        return {"type": "lease_ack", "node_id": self.node_id,
                "lease_id": lease["lease_id"], "accepted": True}

    def _apply(self, msg: dict[str, Any]) -> dict[str, Any]:
        rev_seq = int(msg["rev_seq"])
        changes = {str(k): int(v) for k, v in msg["changes"].items()}
        lease_id = msg.get("lease_id")
        lease = self.leases.get(lease_id) if lease_id else None
        if lease is None:
            raise LeaseExpiredError("node holds no such lease",
                                    node_id=self.node_id, lease_id=lease_id)
        if is_expired(lease["deadline_ms"]):
            raise LeaseExpiredError("lease already expired", lease_id=lease["lease_id"])
        if not lease["low_seq"] <= rev_seq <= lease["high_seq"]:
            raise RevisionOutOfRangeError(
                "rev_seq outside lease window",
                rev_seq=rev_seq, low_seq=lease["low_seq"], high_seq=lease["high_seq"],
            )
        uncovered = sorted(set(changes) - lease["keys"])
        if uncovered:
            raise KeyNotCoveredError(
                "keys not covered by lease", rev_seq=rev_seq, uncovered=uncovered,
            )
        # Apply in deterministic key order; rejected applies never reach here,
        # so the cursor only moves on success.
        for key in sorted(changes):
            self.view[key] = changes[key]
        self.applied_rev = rev_seq
        summary = summary_for_revision(rev_seq)
        return {
            "type": "apply_result",
            "node_id": self.node_id,
            "lease_id": lease["lease_id"],
            "rev_seq": rev_seq,
            "applied_rev": self.applied_rev,
            "view": dict(sorted(self.view.items())),
            "summary": summary,
        }

    def _replace_view(self, msg: dict[str, Any]) -> dict[str, Any]:
        # Whole-state replacement, never an incremental merge.
        self.view = {str(k): int(v) for k, v in msg["view"].items()}
        self.applied_rev = int(msg["applied_rev"])
        if msg.get("clear_lease"):
            self.leases.clear()
        return {
            "type": "replace_ack",
            "node_id": self.node_id,
            "applied_rev": self.applied_rev,
            "view": dict(sorted(self.view.items())),
        }


def _error(code: str, message: str, **context: Any) -> dict[str, Any]:
    payload = {"type": "error", "code": code, "message": message}
    payload.update(context)
    return payload


async def _serve(node_id: str, host: str, port: int) -> None:
    state = NodeState(node_id)

    async def on_connect(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter) -> None:
        await write_message(writer, {
            "type": "ready",
            "node_id": node_id,
            "pid": os.getpid(),
            "now_ms": now_ms(),
        })
        try:
            async for raw in _reader(reader):
                reply = state.handle(raw)
                await write_message(writer, reply)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _reader(reader: asyncio.StreamReader):
        from .protocol import stream_messages
        async for message in stream_messages(reader):
            yield message

    server = await asyncio.start_server(on_connect, host, port)
    actual_port = server.sockets[0].getsockname()[1]
    # stdout line lets the platform confirm the bound port; server already owns
    # the socket at this point, so a bind failure would have raised earlier.
    print(f"LISTENING {actual_port}", flush=True)
    try:
        await server.serve_forever()
    finally:
        server.close()
        await server.wait_closed()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="configlease consumer node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        asyncio.run(_serve(args.node_id, args.host, args.port))
    except OSError as exc:
        # Bind failure (e.g. EADDRINUSE) is reported as a structured stdout line.
        print(f"BIND_ERROR {exc.__class__.__name__}: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
