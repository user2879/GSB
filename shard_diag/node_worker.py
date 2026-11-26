from __future__ import annotations

import argparse
import json
import socketserver
import time
from typing import Any

from .canonical import digest_bytes
from .determinism import aggregate_chunk, generate_samples


class NodeState:
    def __init__(self, node_id: str, seed: int, capacity_version: int) -> None:
        self.node_id = node_id
        self.seed = seed
        self.capacity_version = capacity_version
        self.replayed_revision = 0
        self.leases: dict[str, dict[str, Any]] = {}


class ThreadingJSONLineServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type, state: NodeState) -> None:
        super().__init__(address, handler)
        self.state = state


class JSONLineHandler(socketserver.StreamRequestHandler):
    server: ThreadingJSONLineServer

    def _send(self, response: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(response, sort_keys=True) + "\n").encode("utf-8"))
        self.wfile.flush()

    def _error(self, code: str, message: str, **details: Any) -> None:
        self._send({"ok": False, "error": {"code": code, "message": message, "details": details}})

    def handle(self) -> None:
        line = self.rfile.readline()
        if not line:
            return
        try:
            request = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._error("E_SPEC", f"invalid JSON line: {exc}")
            return

        state = self.server.state
        request_type = request.get("type")
        try:
            if request_type == "ping":
                self._send(
                    {
                        "ok": True,
                        "node_id": state.node_id,
                        "capacity_version": state.capacity_version,
                        "replayed_revision": state.replayed_revision,
                        "monotonic_ms": time.monotonic_ns() // 1_000_000,
                    }
                )
                return

            if request_type == "grant_lease":
                lease_id = str(request["lease_id"])
                version = int(request["lease_version"])
                capacity_version = int(request["capacity_version"])
                if capacity_version != state.capacity_version:
                    self._error(
                        "E_CAPACITY",
                        "node capacity version does not match grant",
                        current=state.capacity_version,
                        received=capacity_version,
                    )
                    return
                state.leases[lease_id] = {
                    "lease_version": version,
                    "capacity_version": capacity_version,
                    "granted_monotonic_ns": time.monotonic_ns(),
                    "ttl_ms": int(request.get("ttl_ms", 0)),
                }
                self._send({"ok": True, "lease_id": lease_id, "lease_version": version})
                return

            if request_type == "renew_lease":
                lease_id = str(request["lease_id"])
                expected = int(request["expected_version"])
                lease = state.leases.get(lease_id)
                if lease is None:
                    self._error("E_LEASE", "lease is not known to node", reason="unknown")
                    return
                ttl_ms = int(lease.get("ttl_ms", 0))
                age_ms = (time.monotonic_ns() - int(lease["granted_monotonic_ns"])) // 1_000_000
                if ttl_ms > 0 and age_ms > ttl_ms:
                    state.leases.pop(lease_id, None)
                    self._error("E_LEASE", "lease has expired on node clock", reason="expired")
                    return
                current = int(lease["lease_version"])
                if expected != current:
                    self._error(
                        "E_LEASE",
                        "optimistic lease version mismatch",
                        expected=expected,
                        current_version=current,
                    )
                    return
                new_version = current + 1
                lease["lease_version"] = new_version
                lease["granted_monotonic_ns"] = time.monotonic_ns()
                self._send({"ok": True, "lease_id": lease_id, "lease_version": new_version})
                return

            if request_type == "release_lease":
                state.leases.pop(str(request["lease_id"]), None)
                self._send({"ok": True, "lease_id": request["lease_id"]})
                return

            if request_type == "fence_capacity":
                new_version = int(request["capacity_version"])
                revoked = len(state.leases)
                state.leases.clear()
                state.capacity_version = new_version
                self._send(
                    {
                        "ok": True,
                        "capacity_version": new_version,
                        "revoked_leases": revoked,
                    }
                )
                return

            if request_type == "aggregate":
                capacity_version = int(request["capacity_version"])
                if capacity_version != state.capacity_version:
                    self._error(
                        "E_CAPACITY",
                        "stale capacity version",
                        current=state.capacity_version,
                        received=capacity_version,
                    )
                    return

                lease_id = request.get("lease_id")
                if lease_id is not None:
                    lease = state.leases.get(str(lease_id))
                    if lease is None:
                        self._error("E_LEASE", "aggregate lease is not known", reason="unknown")
                        return
                    if int(request.get("lease_version", -1)) != int(lease["lease_version"]):
                        self._error(
                            "E_LEASE",
                            "aggregate lease version mismatch",
                            expected=int(lease["lease_version"]),
                            received=int(request.get("lease_version", -1)),
                        )
                        return

                delay_ms = int(request.get("processing_delay_ms", 0))
                if delay_ms > 0:
                    time.sleep(min(delay_ms, 30_000) / 1000.0)

                start_ns = time.monotonic_ns()
                shard_id = str(request["shard_id"])
                sample_rows = int(request["sample_rows"])
                offset = int(request.get("row_offset", 0))
                row_count_value = request.get("row_count")
                row_count = None if row_count_value is None else int(row_count_value)
                samples = generate_samples(state.seed, shard_id, sample_rows)
                chunk = aggregate_chunk(samples, offset, row_count)
                sample_digest = digest_bytes(samples.tobytes())

                primary_revision = int(request["primary_revision"])
                state.replayed_revision = primary_revision
                elapsed_ms = (time.monotonic_ns() - start_ns) // 1_000_000
                self._send(
                    {
                        "ok": True,
                        "node_id": state.node_id,
                        "shard_id": shard_id,
                        "capacity_version": state.capacity_version,
                        "primary_revision": primary_revision,
                        "replayed_revision": state.replayed_revision,
                        "elapsed_monotonic_ms": elapsed_ms,
                        "sample_digest": sample_digest,
                        "chunk": chunk,
                    }
                )
                return

            self._error("E_SPEC", f"unknown request type {request_type!r}")
        except (KeyError, TypeError, ValueError) as exc:
            self._error("E_SPEC", f"invalid request fields: {exc}")
        except Exception as exc:  # pragma: no cover - defensive node boundary
            self._error("E_NODE", f"node failure: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--capacity-version", required=True, type=int)
    args = parser.parse_args()

    state = NodeState(args.node_id, args.seed, args.capacity_version)
    server = ThreadingJSONLineServer((args.host, args.port), JSONLineHandler, state)
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
