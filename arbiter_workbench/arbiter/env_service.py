"""Deterministic per-environment arbitration service (subprocess entry point).

The process binds a real loopback TCP socket and speaks one JSON object per
line. It keeps an in-process effective chain of observations ordered by
revision (then timestamp/instance for stability) and answers arbitration
requests with an authority digest plus a monotonic millisecond cost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from typing import Any

from .errors import ErrorContext, PortError
from .protocol import TERMINATOR


def _digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class EnvironmentServer:
    def __init__(self, env_id: str, host: str, port: int, lease_capacity: int):
        self.env_id = env_id
        self.host = host
        self.port = port
        self.lease_capacity = lease_capacity
        # key -> effective chain (list of obs dicts), revision ordered
        self.chains: dict[str, list[dict[str, Any]]] = {}
        self._server: socket.socket | None = None

    def bind(self, inherited_fd: int | None = None) -> socket.socket:
        if inherited_fd is not None:
            # Listener already bound and listening by the parent; inherit it so
            # there is no window for another process to grab the port.
            server = socket.socket(fileno=inherited_fd)
        else:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server.bind((self.host, self.port))
            except OSError as exc:
                server.close()
                raise PortError(
                    "environment service failed to bind loopback port",
                    ErrorContext(self.host, str(self.port), str(exc)),
                ) from exc
            server.listen(8)
        server.settimeout(30.0)
        self._server = server
        return server

    def _ingest(self, batch: list[dict[str, Any]]) -> int:
        accepted = 0
        for obs in batch:
            if obs.get("env_id") != self.env_id:
                continue
            key = obs["key"]
            chain = self.chains.setdefault(key, [])
            # replace same revision/instance observation, else append
            replaced = False
            for idx, existing in enumerate(chain):
                if (
                    existing["revision"] == obs["revision"]
                    and existing["instance_id"] == obs["instance_id"]
                    and existing["key"] == obs["key"]
                ):
                    chain[idx] = obs
                    replaced = True
                    break
            if not replaced:
                chain.append(obs)
            chain.sort(key=lambda o: (o["revision"], o["ts_ms"], o["instance_id"]))
            # lease capacity bounds the effective chain length per key
            if len(chain) > self.lease_capacity:
                del chain[: len(chain) - self.lease_capacity]
            accepted += 1
        return accepted

    def _arbitrate(self) -> dict[str, Any]:
        per_key: dict[str, dict[str, Any]] = {}
        revisions: list[int] = []
        for key in sorted(self.chains):
            chain = self.chains[key]
            if not chain:
                continue
            head = chain[-1]
            revisions.extend((o["revision"] for o in chain))
            per_key[key] = {
                "key": key,
                "head_revision": head["revision"],
                "head_ts_ms": head["ts_ms"],
                "head_value": head["raw_value"],
                "chain_len": len(chain),
                "digest": _digest(chain),
            }
        return {
            "env_id": self.env_id,
            "revision_min": min(revisions) if revisions else 0,
            "revision_max": max(revisions) if revisions else 0,
            "keys": per_key,
            "authority_digest": _digest(per_key),
        }

    def handle(self, request: dict[str, Any], started_monotonic_ms: int) -> dict[str, Any]:
        command = request.get("command")
        if command == "HELLO":
            return {"status": "OK", "env_id": self.env_id, "lease_capacity": self.lease_capacity}
        if command == "BATCH":
            accepted = self._ingest(request.get("observations", []))
            return {
                "status": "OK",
                "env_id": self.env_id,
                "accepted": accepted,
                "elapsed_ms": _mono_ms() - started_monotonic_ms,
            }
        if command == "ARBITRATE":
            summary = self._arbitrate()
            summary["status"] = "OK"
            summary["elapsed_ms"] = _mono_ms() - started_monotonic_ms
            return summary
        if command == "SHUTDOWN":
            return {"status": "BYE", "env_id": self.env_id}
        return {"status": "ERROR", "env_id": self.env_id, "detail": f"unknown command {command!r}"}

    def serve_forever(self, inherited_fd: int | None = None) -> None:
        server = self.bind(inherited_fd)
        # Announce readiness on a single line of stdout for the parent process.
        sys.stdout.write(f"READY {self.env_id} {self.port}\n")
        sys.stdout.flush()
        while True:
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(5.0)
                buffer = bytearray()
                while True:
                    try:
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        break
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    while TERMINATOR in buffer:
                        line, _, rest = buffer.partition(TERMINATOR)
                        buffer = bytearray(rest)
                        request = json.loads(line.decode("utf-8"))
                        started = _mono_ms()
                        reply = self.handle(request, started)
                        conn.sendall(
                            json.dumps(reply, sort_keys=True, separators=(",", ":")).encode("utf-8")
                            + TERMINATOR
                        )
                        if request.get("command") == "SHUTDOWN":
                            server.close()
                            return


def _mono_ms() -> int:
    return int(time.monotonic() * 1000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Environment arbitration subprocess")
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--lease-capacity", type=int, required=True)
    parser.add_argument("--fd", type=int, default=None,
                        help="inherited listening socket file descriptor")
    args = parser.parse_args(argv)
    server = EnvironmentServer(args.env_id, args.host, args.port, args.lease_capacity)
    try:
        server.serve_forever(inherited_fd=args.fd)
    except PortError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
