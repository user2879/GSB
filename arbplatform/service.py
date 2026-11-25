"""Deterministic per-environment arbitration service run as a subprocess.

The service owns a real listening TCP socket on the declared loopback port and
speaks one NDJSON message per line. It accumulates observation batches, builds
per-key revision chains and answers arbitration requests with an authority
digest, revision range and monotonic elapsed time. No artificial delays are
introduced anywhere; timeout behavior comes solely from real socket timeouts.
"""
from __future__ import annotations

import json
import socket
import sys
from typing import Any

from .determinism import digest_of, now_ms
from .protocol import recv_line, send_line


class EnvironmentService:
    def __init__(self, env_id: str, host: str, port: int) -> None:
        self.env_id = env_id
        self.host = host
        self.port = port
        # obs_id -> observation; replaced observations overwrite deterministically.
        self.observations: dict[str, dict[str, Any]] = {}

    def handle_batch(self, message: dict[str, Any]) -> dict[str, Any]:
        for obs in message.get("observations", []):
            self.observations[obs["obs_id"]] = obs
        keys = sorted({o["key"] for o in self.observations.values()})
        return {
            "type": "batch_ack",
            "env_id": self.env_id,
            "accepted": len(message.get("observations", [])),
            "stored": len(self.observations),
            "keys": keys,
        }

    def _chain_summary(self, key: str) -> dict[str, Any]:
        items = [o for o in self.observations.values() if o["key"] == key]
        if not items:
            return {"seq_low": None, "seq_high": None, "authority_digest": digest_of([])}
        latest: dict[str, dict[str, Any]] = {}
        for obs in items:
            cur = latest.get(obs["instance_id"])
            if cur is None or int(obs["seq"]) >= int(cur["seq"]):
                latest[obs["instance_id"]] = obs
        reps = sorted(
            latest.values(),
            key=lambda o: (-int(o["seq"]), str(o["instance_id"])),
        )
        seqs = [int(o["seq"]) for o in items]
        canonical_reps = sorted(
            (o["instance_id"], o["seq"], o["value"]) for o in reps
        )
        return {
            "seq_low": min(seqs),
            "seq_high": max(seqs),
            "authority_digest": digest_of(canonical_reps),
        }

    def handle_arbitrate(self, message: dict[str, Any]) -> dict[str, Any]:
        start = now_ms()
        keys = sorted(message.get("keys")
                      or {o["key"] for o in self.observations.values()})
        summaries = {key: self._chain_summary(key) for key in keys}
        seq_lows = [s["seq_low"] for s in summaries.values()
                    if s["seq_low"] is not None]
        seq_highs = [s["seq_high"] for s in summaries.values()
                     if s["seq_high"] is not None]
        return {
            "type": "arbitration_reply",
            "env_id": self.env_id,
            "seq_low": min(seq_lows) if seq_lows else None,
            "seq_high": max(seq_highs) if seq_highs else None,
            "authority_digest": digest_of(summaries),
            "summaries": summaries,
            "elapsed_ms": now_ms() - start,
        }

    def serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
        except OSError as exc:
            # Distinct exit marker so the parent maps this to E_PORT.
            sys.stderr.write(f"BIND_ERROR {self.env_id} {self.port} {exc}\n")
            sys.exit(2)
        server.listen(8)
        # Announce readiness on stdout (single line, flushed).
        sys.stdout.write(f"READY {self.env_id} {self.port}\n")
        sys.stdout.flush()
        try:
            while True:
                conn, _ = server.accept()
                with conn:
                    try:
                        message = recv_line(conn)
                    except (ConnectionError, json.JSONDecodeError):
                        continue
                    mtype = message.get("type")
                    if mtype == "shutdown":
                        send_line(conn, {"type": "shutdown_ack",
                                         "env_id": self.env_id})
                        break
                    if mtype == "batch":
                        reply = self.handle_batch(message)
                    elif mtype == "arbitrate":
                        reply = self.handle_arbitrate(message)
                    else:
                        reply = {"type": "error", "env_id": self.env_id,
                                 "error": f"unknown message type {mtype!r}"}
                    send_line(conn, reply)
        finally:
            server.close()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 3:
        sys.stderr.write("usage: python -m arbplatform.service ENV_ID HOST PORT\n")
        return 64
    env_id, host, port = argv[0], argv[1], int(argv[2])
    EnvironmentService(env_id, host, port).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
