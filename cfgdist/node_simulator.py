"""Simulated component node: a standalone loopback TCP subprocess.

The node computes the same derived digest as the platform, so the happy path
converges. Fault knobs are supplied through the node declaration and affect
only runtime behaviour, never the platform implementation:

* ``timeout_once``: ignore the first dispatch of a config version once.
* ``digest_mismatch``: corrupt the reported digest (drives E_CONVERGE).
* ``always_fail``: reply with an error to every dispatch.
"""

from __future__ import annotations

import json
import socket
import socketserver
import sys
import threading

from .derivation import node_digest
from .manifest import canonical_json


class NodeState:
    def __init__(self, node_id: str, faults: dict):
        self.node_id = node_id
        self.faults = faults
        self.lock = threading.Lock()
        self.timed_out_versions: set[int] = set()
        self.applied_versions: dict[int, dict] = {}


class NodeHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        state: NodeState = self.server.node_state  # type: ignore[attr-defined]
        for raw_line in self.rfile:
            line = raw_line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._reply({"ok": False, "error": "invalid json"})
                continue
            kind = message.get("kind")
            if kind == "health":
                self._reply({"ok": True, "node": state.node_id, "kind": "health_ack"})
            elif kind == "dispatch":
                self._handle_dispatch(state, message)
            elif kind == "status":
                with state.lock:
                    applied = state.applied_versions.get(message.get("config_version"))
                self._reply(
                    {
                        "ok": True,
                        "node": state.node_id,
                        "kind": "status_ack",
                        "applied": applied,
                    }
                )
            else:
                self._reply({"ok": False, "error": f"unknown kind {kind!r}"})

    def _handle_dispatch(self, state: NodeState, message: dict) -> None:
        config_version = message["config_version"]
        attempt = message.get("attempt", 1)
        faults = state.faults

        if faults.get("always_fail"):
            self._reply(
                {
                    "ok": False,
                    "node": state.node_id,
                    "config_version": config_version,
                    "attempt": attempt,
                    "error": "injected permanent node failure",
                }
            )
            return

        if faults.get("timeout_once") and config_version not in state.timed_out_versions:
            state.timed_out_versions.add(config_version)
            # Simulate a timeout by staying silent and dropping the request.
            return

        entries = message["entries"]
        derivations = message.get("derivations", [])
        digest = node_digest(state.node_id, entries, derivations)
        if faults.get("digest_mismatch"):
            digest = "0" * len(digest)

        with state.lock:
            already = config_version in state.applied_versions
            state.applied_versions[config_version] = {
                "entries": entries,
                "digest": digest,
                "last_attempt": attempt,
            }
        self._reply(
            {
                "ok": True,
                "node": state.node_id,
                "config_version": config_version,
                "attempt": attempt,
                "receipt_id": f"{state.node_id}:{config_version}:{attempt}",
                "retry": attempt > 1,
                "duplicate": already,
                "digest": digest,
                "entry_count": len(entries),
            }
        )

    def _reply(self, payload: dict) -> None:
        self.wfile.write((canonical_json(payload) + "\n").encode("utf-8"))
        self.wfile.flush()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_class, node_state: NodeState):
        super().__init__(server_address, handler_class)
        self.node_state = node_state


def run_node(node_id: str, host: str, port: int, faults: dict) -> None:
    state = NodeState(node_id, faults)
    server = ThreadedTCPServer((host, port), NodeHandler, state)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 3:
        print("usage: python -m cfgdist.node_simulator NODE_ID HOST PORT [faults-json]")
        return 2
    node_id, host, port = argv[0], argv[1], int(argv[2])
    faults = json.loads(argv[3]) if len(argv) > 3 else {}
    run_node(node_id, host, port, faults)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
