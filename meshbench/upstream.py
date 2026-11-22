"""Deterministic simulated upstream instances served over real loopback TCP.

Each instance binds 127.0.0.1:<high port>, accepts one JSON request line per
connection, sleeps a fixed processing time, and answers with a JSON line.  The
error/slow decision is a deterministic function of the request's stable key,
so identical request sequences reproduce identical behaviour across runs.
"""

import hashlib
import json
import socket
import threading
import time

from .config import LOOPBACK, UpstreamSpec


def _decision_bucket(payload: dict) -> int:
    key = "|".join([
        str(payload.get("route_id", "")),
        str(payload.get("client_key", "")),
        str(payload.get("sequence", 0)),
        str(payload.get("attempt", 0)),
    ])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10_000


class MockUpstream:
    """A single threaded, deterministic, loopback-only mock upstream."""

    def __init__(self, instance_id: str, port: int, spec: UpstreamSpec,
                 host: str = LOOPBACK):
        self.instance_id = instance_id
        self.host = host
        self.port = port
        self.spec = spec
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._server = None
        self._handled = 0

    @property
    def handled(self):
        with self._lock:
            return self._handled

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(64)
        server.settimeout(0.1)
        self._server = server
        self._thread = threading.Thread(
            target=self._serve, name=f"upstream-{self.instance_id}", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._server is not None:
            self._server.close()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle, args=(conn,),
                name=f"upstream-{self.instance_id}-conn", daemon=True,
            ).start()

    def _handle(self, conn):
        try:
            conn.settimeout(5.0)
            buffer = b""
            while b"\n" not in buffer:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buffer += chunk
            line, _rest = buffer.split(b"\n", 1)
            payload = json.loads(line.decode("utf-8"))
            bucket = _decision_bucket(payload)
            spec = self.spec
            is_error = bucket < spec.error_permyriad
            is_slow = (not is_error) and bucket < (
                spec.error_permyriad + spec.slow_permyriad
            )
            sleep_ms = spec.process_ms + (spec.slow_extra_ms if is_slow else 0)
            if sleep_ms:
                time.sleep(sleep_ms / 1000.0)
            response = {
                "instance_id": self.instance_id,
                "status": "error" if is_error else "ok",
                "slow": is_slow,
                "slept_ms": sleep_ms,
                "decision_bucket": bucket,
            }
            with self._lock:
                self._handled += 1
            conn.sendall(json.dumps(response, sort_keys=True).encode("utf-8") + b"\n")
        except (OSError, ValueError, json.JSONDecodeError):
            # A broken/malformed connection simply closes without a response,
            # which the dispatcher observes as a real connection-level failure.
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


def start_all(config):
    """Start one deterministic mock upstream per declared instance."""
    upstreams = []
    for service in sorted(config.services.values(), key=lambda s: s.service_id):
        for inst in sorted(service.instances, key=lambda i: i.instance_id):
            upstream = MockUpstream(inst.instance_id, inst.port, inst.upstream)
            upstream.start()
            upstreams.append(upstream)
    # Let every listening socket enter accept() before preflight connects.
    time.sleep(0.15)
    return upstreams


def stop_all(upstreams):
    for upstream in upstreams:
        upstream.stop()
