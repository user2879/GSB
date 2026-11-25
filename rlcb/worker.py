"""Loopback service subprocess with a real TCP line protocol.

Each service is launched as its own subprocess that binds the declared
127.0.0.1 port and serves newline-delimited JSON. The main process connects
over a real socket with genuine connect/read timeouts and validates every
required reply field, mapping failures to E_PORT / E_TIMEOUT / E_REPLY.

Message kinds (one JSON object per line):

* ``hello``     -> readiness handshake;
* ``execute``   -> deterministic NumPy compute for one request batch;
* ``lease_check`` / ``lease_bump`` -> authoritative optimistic lease version.
"""

from __future__ import annotations

import argparse
import json
import socketserver
import sys
import time
from typing import Any

from .fixedpoint import deterministic_execute
from .lease import LeaseRegistry

REQUIRED_REPLY_FIELDS = (
    "ok", "service_id", "batch_id", "summary", "config_version", "elapsed_ms",
)


class ServiceBackend:
    def __init__(self, service_id: str, config_version: int, seed: int,
                 ttl_ms: int):
        self.service_id = service_id
        self.config_version = config_version
        self.seed = seed
        self.registry = LeaseRegistry(service_id, ttl_ms)

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter_ns()
        kind = message.get("kind")
        if kind == "hello":
            return self._wrap(message, {"ready": True,
                                        "service_id": self.service_id})
        if kind == "execute":
            return self._execute(message, started)
        if kind == "lease_check":
            ok = self.registry.check_version(
                message["lease_id"], int(message["expected_version"]))
            return self._wrap(message, {"lease_ok": ok,
                                        "lease_id": message["lease_id"]})
        if kind == "lease_bump":
            lease_id = message["lease_id"]
            self.registry.record_grant(
                lease_id, int(message.get("initial_version", 1)))
            new_version = self.registry.bump(lease_id)
            return self._wrap(message, {"lease_id": lease_id,
                                        "version": new_version})
        return {"ok": False, "error": f"unknown kind {kind!r}"}

    def _execute(self, message: dict[str, Any], started_ns: int) -> dict[str, Any]:
        batch_id = message["batch_id"]
        results = []
        mac_total = 0
        for item in message["requests"]:
            result = deterministic_execute(
                seed=self.seed,
                service_id=self.service_id,
                batch_id=batch_id,
                arrival_seq=int(item["arrival_seq"]),
                sample_size=int(message.get("sample_size", 16)),
            )
            results.append({
                "request_id": item["request_id"],
                "arrival_seq": int(item["arrival_seq"]),
                "digest": result.digest,
                "vector": list(result.vector),
                "failed": int(item.get("failed", 0)),
            })
            mac_total += result.mac_count
        # Byte-stable summary excludes the wall-clock elapsed time.
        summary = {
            "service_id": self.service_id,
            "batch_id": batch_id,
            "config_version": self.config_version,
            "result_count": len(results),
            "mac_count": mac_total,
            "results": results,
        }
        elapsed_ms = max(1, int((time.perf_counter_ns() - started_ns)
                                // 1_000_000))
        return {
            "ok": True,
            "service_id": self.service_id,
            "batch_id": batch_id,
            "summary": summary,
            "config_version": self.config_version,
            "elapsed_ms": elapsed_ms,
            "compute_backend": "cpu-int-fixed-point-numpy",
        }

    def _wrap(self, message: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        payload.setdefault("ok", True)
        payload.setdefault("service_id", self.service_id)
        payload.setdefault("batch_id", message.get("batch_id", ""))
        payload.setdefault("summary", None)
        payload.setdefault("config_version", self.config_version)
        payload.setdefault("elapsed_ms", 0)
        return payload


class LineTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str, port: int, backend: ServiceBackend):
        self.backend = backend
        super().__init__((host, port), LineHandler)


class LineHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                message = json.loads(line.decode("utf-8"))
                response = self.server.backend.handle(message)  # type: ignore[attr-defined]
            except Exception as exc:  # keep the process alive on bad input
                response = {"ok": False, "error": type(exc).__name__,
                            "message": str(exc)}
            self.wfile.write(
                (json.dumps(response, sort_keys=True) + "\n").encode("utf-8"))
            self.wfile.flush()


def run_service(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rlcb loopback service")
    parser.add_argument("--service-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config-version", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ttl-ms", type=int, default=250)
    args = parser.parse_args(argv)

    backend = ServiceBackend(
        args.service_id, args.config_version, args.seed, args.ttl_ms)
    try:
        server = LineTCPServer(args.host, args.port, backend)
    except OSError as exc:
        print(json.dumps({
            "event": "bind_failed", "service_id": args.service_id,
            "host": args.host, "port": args.port,
            "error": type(exc).__name__, "message": str(exc),
        }), file=sys.stderr, flush=True)
        return 2
    print(json.dumps({
        "event": "listening", "service_id": args.service_id,
        "host": args.host, "port": args.port,
    }), flush=True)
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_service())
