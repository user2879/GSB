from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
import threading
import time
from typing import Any

from .canonical import canonical_json
from .simulation import compute_aggregate


class NodeState:
    def __init__(
        self,
        cluster_id: str,
        revision_version: int,
        config: dict[str, Any],
        simulation_seed: int,
    ) -> None:
        self.cluster_id = cluster_id
        self.revision_version = revision_version
        self.config = config
        self.simulation_seed = simulation_seed
        self.shutdown = threading.Event()


class NodeHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        state: NodeState = self.server.state  # type: ignore[attr-defined]
        while not state.shutdown.is_set():
            line = self.rfile.readline()
            if not line:
                return
            try:
                request = json.loads(line.decode("utf-8"))
                response = self._process(request, state)
            except Exception as exc:  # Node protocol must always answer.
                response = {"ok": False, "code": "E_PROTOCOL", "message": str(exc)}
            try:
                self.wfile.write(canonical_json(response).encode("utf-8") + b"\n")
                self.wfile.flush()
            except OSError:
                return

    def _process(self, request: dict[str, Any], state: NodeState) -> dict[str, Any]:
        op = request.get("op")
        if op == "handshake":
            if request.get("cluster_id") != state.cluster_id:
                return {"ok": False, "code": "E_HANDSHAKE", "message": "cluster id mismatch"}
            return {
                "ok": True,
                "event": "handshake",
                "cluster_id": state.cluster_id,
                "revision_version": state.revision_version,
                "pid": os.getpid(),
            }
        if op == "dispatch":
            if request.get("revision_version") != state.revision_version:
                return {"ok": False, "code": "E_REVISION", "message": "revision mismatch"}
            started = time.perf_counter_ns()
            aggregate = compute_aggregate(
                simulation_seed=state.simulation_seed,
                cluster_id=state.cluster_id,
                revision_version=state.revision_version,
                task_id=request["task_id"],
                shard_id=request["shard_id"],
                config_keys=request["config_keys"],
                config_values=state.config,
            )
            elapsed_ms = int((time.perf_counter_ns() - started) // 1_000_000)
            return {
                "ok": True,
                "event": "aggregate",
                "cluster_id": state.cluster_id,
                "shard_id": request["shard_id"],
                "revision_version": state.revision_version,
                "aggregate": aggregate,
                "elapsed_ms": elapsed_ms,
            }
        if op == "stop":
            # The orchestrator closes stdin after receiving this reply. Closing
            # stdin drives shutdown after the client has closed this connection,
            # avoiding a server-side TIME_WAIT that would block an immediate
            # restart on the same declared port.
            return {"ok": True, "event": "stopping"}
        return {"ok": False, "code": "E_PROTOCOL", "message": f"unknown op: {op!r}"}


class NodeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _watch_stdin(server: NodeServer, state: NodeState) -> None:
    """Exit when the orchestrating parent closes the pipe.

    This prevents orphaned loopback nodes when the coordinating process is
    force-terminated during crash-recovery validation.
    """

    try:
        while sys.stdin.buffer.readline():
            pass
    except OSError:
        pass
    state.shutdown.set()
    server.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback distribution node")
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--revision-version", required=True, type=int)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--simulation-seed", required=True, type=int)
    args = parser.parse_args(argv)

    try:
        config = json.loads(args.config_json)
        server = NodeServer((args.host, args.port), NodeHandler)
    except OSError as exc:
        sys.stderr.write(
            canonical_json({"code": "E_PORT", "message": str(exc), "port": args.port}) + "\n"
        )
        sys.stderr.flush()
        return 2
    except Exception as exc:
        sys.stderr.write(canonical_json({"code": "E_HANDSHAKE", "message": str(exc)}) + "\n")
        sys.stderr.flush()
        return 3

    state = NodeState(args.cluster_id, args.revision_version, config, args.simulation_seed)
    server.state = state  # type: ignore[attr-defined]
    monitor = threading.Thread(target=_watch_stdin, args=(server, state), daemon=True)
    monitor.start()
    if os.environ.get("CONFIGDIST_TEST_STALL_HANDSHAKE") == "1":
        # Acceptance hook: the socket is bound and accepts connections, but the
        # JSON-line handshake never completes. This exercises E_HANDSHAKE as a
        # real protocol timeout, not as a fabricated connection failure.
        while True:
            time.sleep(0.25)
    try:
        server.serve_forever(poll_interval=0.02)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
