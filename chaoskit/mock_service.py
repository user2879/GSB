"""Deterministic loopback target service, run as a real child process.

The service never reads or modifies source code. Its only runtime inputs are
command-line configuration plus text commands sent over the loopback socket.
The error-rate injection is therefore a genuine runtime parameter change:
the orchestrator tells the service the seed/sequence/ratio to apply.

Line protocol (one JSON object per connection, terminated by ``\\n``)::

    PING\\n                                   -> READY service-id
    IDENT\\n                                  -> IDENT service-id
    REQUEST\\n                                -> OK   <latency-ms>
                                               ERR <latency-ms>
    INJECT {"seq":n,"error_bps":b,"seed":s}\\n -> APPLIED
    CLEAR\\n                                  -> CLEARED
    STATE\\n                                  -> STATE {...json...}
    SHUTDOWN\\n                               -> BYE

Every REQUEST sleeps the configured fixed cost first, so both success and
error paths exhibit the same deterministic processing delay at the service.
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from typing import Any, Dict

from .rng import SplitMix64, mix_u64

_sentinel = object()


class MockService:
    def __init__(self, service_id: str, host: str, port: int,
                 fixed_cost_ms: int, base_error_bps: int) -> None:
        self.service_id = service_id
        self.host = host
        self.port = port
        self.fixed_cost_ms = fixed_cost_ms
        self.base_error_bps = base_error_bps
        self._lock = threading.Lock()
        self._request_count = 0
        self._active_seq: int | None = None
        self._active_error_bps: int | None = None
        self._active_seed: int | None = None
        self._stop = threading.Event()
        self._server: socket.socket | None = None

    def serve(self) -> int:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
        except OSError:
            # Distinct exit code lets the orchestrator classify port conflicts.
            return 3
        server.listen(32)
        # Short accept timeout: while a worker thread sleeps through a
        # request's fixed cost, the accept loop must still pick up queued
        # connections so concurrent observations are handled in parallel.
        server.settimeout(0.1)
        self._server = server
        print("READY", flush=True)
        while not self._stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        return 0

    def shutdown(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(5.0)
            buffer = b""
            while b"\n" not in buffer:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buffer += chunk
            line = buffer.split(b"\n", 1)[0].decode("utf-8").strip()
            response = self._dispatch(line)
            conn.sendall((response + "\n").encode("utf-8"))

    def _dispatch(self, line: str) -> str:
        command, _, payload = line.partition(" ")
        if command == "PING":
            return f"READY {self.service_id}"
        if command == "IDENT":
            return f"IDENT {self.service_id}"
        if command == "SHUTDOWN":
            response = "BYE"
            threading.Thread(target=self._delayed_shutdown, daemon=True).start()
            return response
        if command == "STATE":
            with self._lock:
                return "STATE " + json.dumps({
                    "service_id": self.service_id,
                    "request_count": self._request_count,
                    "active_seq": self._active_seq,
                    "active_error_bps": self._active_error_bps,
                }, sort_keys=True)
        if command == "INJECT":
            return self._apply_inject(payload)
        if command == "CLEAR":
            with self._lock:
                self._active_seq = None
                self._active_error_bps = None
                self._active_seed = None
            return "CLEARED"
        if command == "REQUEST":
            return self._handle_request()
        return "ERROR unknown-command"

    def _delayed_shutdown(self) -> None:
        time.sleep(0.15)
        self.shutdown()

    def _apply_inject(self, payload: str) -> str:
        try:
            data = json.loads(payload)
            seq = data["seq"]
            error_bps = data["error_bps"]
            seed = data["seed"]
            if not (isinstance(seq, int) and isinstance(error_bps, int) and isinstance(seed, int)):
                raise ValueError
            if not 0 <= error_bps <= 10000:
                raise ValueError
        except (ValueError, TypeError, KeyError):
            return "ERROR bad-inject-payload"
        with self._lock:
            self._active_seq = seq
            self._active_error_bps = error_bps
            self._active_seed = seed
        return "APPLIED"

    def _handle_request(self) -> str:
        with self._lock:
            self._request_count += 1
            request_index = self._request_count
            seq = self._active_seq
            injected_bps = self._active_error_bps
            seed = self._active_seed
            base_bps = self.base_error_bps

        if injected_bps is not None:
            # Stateless per-request decision derived from seed + sequence +
            # request index: replaying the same sequence reproduces each
            # accept/reject decision bit-for-bit.
            decision = SplitMix64(
                mix_u64([seed, seq, request_index, "error_rate"])
            ).next_below(10000)
            is_error = decision < injected_bps
        else:
            decision = SplitMix64(
                mix_u64([seed if seed is not None else 0, request_index, "base_error"])
            ).next_below(10000)
            is_error = decision < base_bps

        start = time.monotonic()
        time.sleep(max(0, self.fixed_cost_ms) / 1000.0)
        latency_ms = int(round((time.monotonic() - start) * 1000.0))
        return f"{'ERR' if is_error else 'OK'} {latency_ms}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic chaos target service")
    parser.add_argument("--service-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--fixed-cost-ms", required=True, type=int)
    parser.add_argument("--base-error-bps", required=True, type=int)
    args = parser.parse_args()

    service = MockService(
        service_id=args.service_id,
        host=args.host,
        port=args.port,
        fixed_cost_ms=args.fixed_cost_ms,
        base_error_bps=args.base_error_bps,
    )
    try:
        return service.serve()
    except KeyboardInterrupt:
        service.shutdown()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
