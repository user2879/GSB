"""Cache node subprocess.

A node binds its declared loopback port, speaks the JSON-line protocol, owns
the monotonic generation (lease algebra) tags for every lease it grants, and
performs the NumPy fixed-point verification described in :mod:`compute`.

Generation rule (per ``lease_id``):

* ``acquire`` creates the lease at tag 0 and records algebra event 0
* ``renew`` / ``release`` must present the current tag; on match the node
  accepts, increments the tag and appends an algebra event; on mismatch it
  rejects with ``E_LEASE_TAG_MISMATCH`` and the tag is untouched

The centre owns wall/monotonic expiry; the node owns generation state.  A
``restore`` message rebuilds node algebra state from a committed snapshot so
that a recovered run keeps the same lease history.
"""
from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import dataclass, field
from typing import Any

from .clock import DeterministicClock, node_clock
from .compute import compute_shard
from .config import NodeConfig, ShardConfig
from .errors import (
    E_LEASE_TAG_MISMATCH,
    E_LEASE_LIMIT,
    E_PROTOCOL,
    E_SHARD_UNAVAILABLE,
    Failure,
)
from .protocol import decode_message, encode_message


@dataclass
class LeaseState:
    lease_id: str
    shard_id: str
    tag: int = 0
    generation_events: list[dict[str, int]] = field(default_factory=list)


class CacheNode:
    """In-process implementation; :func:`run_node` wraps it with a TCP server."""

    def __init__(self, config: NodeConfig, shards: dict[str, ShardConfig]) -> None:
        self.config = config
        self.shards = dict(shards)
        self._leases: dict[str, LeaseState] = {}
        self._lock = threading.Lock()
        self.algebra_log: list[dict[str, Any]] = []
        self.advance_count = 0
        self._clock = node_clock()

    # -- lease algebra -----------------------------------------------------
    def acquire(self, lease_id: str, shard_id: str, now_ms: int) -> dict[str, Any]:
        with self._lock:
            if lease_id in self._leases:
                return self._reject(
                    E_LEASE_LIMIT, "lease_id already exists", lease_id=lease_id
                )
            if len(self._leases) >= self.config.lease_limit:
                return self._reject(
                    E_LEASE_LIMIT, "per-node lease limit reached",
                    lease_id=lease_id, node_id=self.config.node_id,
                )
            state = LeaseState(lease_id=lease_id, shard_id=shard_id)
            event = {"lease_id": lease_id, "shard_id": shard_id, "old_tag": -1,
                     "new_tag": 0, "action": "acquire", "at_ms": now_ms}
            state.generation_events.append(event)
            self._leases[lease_id] = state
            self.algebra_log.append(event)
            return {"ok": True, "lease_id": lease_id, "tag": 0,
                    "node_id": self.config.node_id, "at_ms": now_ms}

    def renew(self, lease_id: str, tag: int, now_ms: int) -> dict[str, Any]:
        with self._lock:
            state = self._leases.get(lease_id)
            if state is None:
                return self._reject(E_LEASE_LIMIT, "unknown lease", lease_id=lease_id)
            if state.tag != tag:
                return self._reject(
                    E_LEASE_TAG_MISMATCH, "generation tag mismatch",
                    lease_id=lease_id, expected=state.tag, presented=tag,
                )
            old = state.tag
            state.tag += 1
            self.advance_count += 1
            event = {"lease_id": lease_id, "shard_id": state.shard_id,
                     "old_tag": old, "new_tag": state.tag,
                     "action": "renew", "at_ms": now_ms}
            state.generation_events.append(event)
            self.algebra_log.append(event)
            return {"ok": True, "lease_id": lease_id, "tag": state.tag,
                    "node_id": self.config.node_id, "at_ms": now_ms}

    def release(self, lease_id: str, tag: int, now_ms: int) -> dict[str, Any]:
        with self._lock:
            state = self._leases.get(lease_id)
            if state is None:
                return self._reject(E_LEASE_LIMIT, "unknown lease", lease_id=lease_id)
            if state.tag != tag:
                return self._reject(
                    E_LEASE_TAG_MISMATCH, "generation tag mismatch",
                    lease_id=lease_id, expected=state.tag, presented=tag,
                )
            old = state.tag
            state.tag += 1
            self.advance_count += 1
            event = {"lease_id": lease_id, "shard_id": state.shard_id,
                     "old_tag": old, "new_tag": state.tag,
                     "action": "release", "at_ms": now_ms}
            state.generation_events.append(event)
            self.algebra_log.append(event)
            del self._leases[lease_id]
            return {"ok": True, "lease_id": lease_id, "tag": state.tag,
                    "node_id": self.config.node_id, "at_ms": now_ms}

    def compute(self, lease_id: str, tag: int, shard_id: str, now_ms: int) -> dict[str, Any]:
        with self._lock:
            state = self._leases.get(lease_id)
            if state is None:
                return self._reject(E_LEASE_LIMIT, "unknown lease", lease_id=lease_id)
            if state.tag != tag:
                return self._reject(
                    E_LEASE_TAG_MISMATCH, "generation tag mismatch",
                    lease_id=lease_id, expected=state.tag, presented=tag,
                )
            shard = self.shards.get(shard_id)
            if shard is None:
                return self._reject(E_SHARD_UNAVAILABLE, "unknown shard",
                                    lease_id=lease_id, shard_id=shard_id)
            started = self._clock.now_ms()
            result = compute_shard(
                shard,
                node_id=self.config.node_id,
                quota_version=self.config.quota_version,
                fixed_point=self.config.fixed_point,
                quant_level=self.config.quant_level,
            )
            finished = self._clock.now_ms()
            duration_ms = max(0, finished - started)
            payload = result.to_dict()
            payload.update({
                "ok": True,
                "lease_id": lease_id,
                "tag": state.tag,
                "at_ms": now_ms,
                "duration_ms": duration_ms,
            })
            return payload

    def restore(self, leases: list[dict[str, Any]], algebra_log: list[dict[str, Any]],
                reset_clock: bool = False) -> dict[str, Any]:
        if reset_clock:
            path = os.environ.get("EDGE_CLOCK_SCHEDULE")
            if path:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self._clock = DeterministicClock(payload["schedule"])
        with self._lock:
            self._leases = {}
            for item in leases:
                state = LeaseState(
                    lease_id=item["lease_id"],
                    shard_id=item["shard_id"],
                    tag=int(item["tag"]),
                    generation_events=list(item.get("generation_events", [])),
                )
                self._leases[state.lease_id] = state
            self.algebra_log = list(algebra_log)
            self.advance_count = sum(
                1 for event in self.algebra_log if event["action"] in ("renew", "release")
            )
            return {"ok": True, "node_id": self.config.node_id,
                    "leases": len(self._leases), "events": len(self.algebra_log)}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "node_id": self.config.node_id,
                "leases": [
                    {"lease_id": state.lease_id, "shard_id": state.shard_id,
                     "tag": state.tag, "generation_events": list(state.generation_events)}
                    for state in self._leases.values()
                ],
                "algebra_log": list(self.algebra_log),
            }

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        mtype = message.get("type")
        now_ms = int(message.get("at_ms", self._clock.now_ms()))
        if mtype == "ping":
            return {"ok": True, "node_id": self.config.node_id, "at_ms": now_ms}
        if mtype == "acquire":
            return self.acquire(message["lease_id"], message["shard_id"], now_ms)
        if mtype == "renew":
            return self.renew(message["lease_id"], int(message["tag"]), now_ms)
        if mtype == "release":
            return self.release(message["lease_id"], int(message["tag"]), now_ms)
        if mtype == "compute":
            return self.compute(message["lease_id"], int(message["tag"]),
                                message["shard_id"], now_ms)
        if mtype == "restore":
            return self.restore(
                message["leases"], message["algebra_log"],
                reset_clock=bool(message.get("reset_clock", False)),
            )
        if mtype == "snapshot":
            payload = self.snapshot()
            payload["ok"] = True
            return payload
        return self._reject(E_PROTOCOL, "unknown message type", mtype=mtype)

    def _reject(self, code: str, message: str, **context: Any) -> dict[str, Any]:
        return {"ok": False, "error": Failure(code, message, context).to_dict(),
                "node_id": self.config.node_id}


class _ThreadedTcpServer:
    """One-connection-per-thread JSON-line loopback server."""

    def __init__(self, node: CacheNode, host: str = "127.0.0.1") -> None:
        self.node = node
        self.host = host
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, node.config.port))
        self._server.listen(16)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def serve_forever(self) -> None:
        self._server.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            thread = threading.Thread(target=self._serve_conn, args=(conn,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def _serve_conn(self, conn: socket.socket) -> None:
        buffer = bytearray()
        with conn:
            conn.settimeout(5.0)
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                buffer.extend(chunk)
                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer = bytearray(rest)
                    message = decode_message(bytes(line))
                    if message.get("type") == "shutdown":
                        conn.sendall(encode_message(
                            {"ok": True, "node_id": self.node.config.node_id}))
                        self.stop()
                        return
                    response = self.node.handle(message)
                    conn.sendall(encode_message(response))

    def stop(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass


def run_node(config: NodeConfig, shards: dict[str, ShardConfig]) -> None:
    node = CacheNode(config, shards)
    server = _ThreadedTcpServer(node)
    server.serve_forever()


def _subprocess_main() -> int:
    import sys

    spec_path = sys.argv[1]
    with open(spec_path, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    node_config = NodeConfig(**spec["node"])
    shards = {
        item["shard_id"]: ShardConfig(**item)
        for item in spec["shards"]
    }
    run_node(node_config, shards)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_subprocess_main())
