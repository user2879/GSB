"""Bus-side orchestration of processor subprocesses and real loopback TCP."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import gsb

from .clock import now_ms
from .errors import BusTimeoutError, LeaseError, PortError
from .processor import READY_MARKER

HANDSHAKE_TIMEOUT_MS = 5000
SOCKET_TIMEOUT_MS = 6000


class ProcessorClient:
    """One persistent real TCP connection to one processor subprocess."""

    def __init__(self, subscriber, log_path=None):
        self.subscriber = subscriber
        self.log_path = log_path
        self._process = None
        self._sock = None
        self._lock = threading.Lock()
        self._buffer = b""
        self._log_handle = None

    def start(self):
        command = [
            sys.executable,
            "-m",
            "gsb.processor",
            "--subscriber-id",
            self.subscriber.subscriber_id,
            "--host",
            self.subscriber.host,
            "--port",
            str(self.subscriber.port),
            "--seed",
            str(self.subscriber.seed),
            "--budget-ms",
            str(self.subscriber.budget_ms),
        ]
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.log_path, "a", encoding="utf-8")
            log_handle = self._log_handle
        else:
            log_handle = subprocess.DEVNULL
        env = dict(os.environ)
        package_root = str(Path(gsb.__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = package_root + (os.pathsep + existing if existing else "")
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=log_handle,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            raise PortError(
                f"cannot launch processor: {exc}",
                entry=self.subscriber.subscriber_id,
                field="process",
            )
        if not self._await_ready():
            self._terminate_process()
            raise PortError(
                f"processor {self.subscriber.subscriber_id} never became ready "
                f"(port {self.subscriber.port} likely in use)",
                entry=self.subscriber.subscriber_id,
                field="port",
            )
        self._connect()

    def _await_ready(self) -> bool:
        deadline = time.monotonic() + HANDSHAKE_TIMEOUT_MS / 1000.0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                return False
            line = self._process.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            if line.strip() == READY_MARKER:
                return True
        return False

    def _connect(self):
        sock = socket.create_connection(
            (self.subscriber.host, self.subscriber.port),
            timeout=HANDSHAKE_TIMEOUT_MS / 1000.0,
        )
        sock.settimeout(SOCKET_TIMEOUT_MS / 1000.0)
        self._sock = sock
        self._send_line("HELLO")
        reply = self._read_line()
        if reply != "READY":
            self.stop()
            raise PortError(
                f"bad handshake reply {reply!r}",
                entry=self.subscriber.subscriber_id,
                field="handshake",
            )

    def _send_line(self, text):
        try:
            self._sock.sendall((text + "\n").encode("utf-8"))
        except OSError as exc:
            raise BusTimeoutError(
                f"socket send failed: {exc}",
                entry=self.subscriber.subscriber_id,
                field="socket",
            )

    def _read_line(self):
        try:
            while b"\n" not in self._buffer:
                chunk = self._sock.recv(65536)
                if not chunk:
                    raise BusTimeoutError(
                        "processor closed connection",
                        entry=self.subscriber.subscriber_id,
                        field="socket",
                    )
                self._buffer += chunk
            line, self._buffer = self._buffer.split(b"\n", 1)
            return line.decode("utf-8").strip()
        except socket.timeout as exc:
            raise BusTimeoutError(
                f"socket read timeout after {SOCKET_TIMEOUT_MS}ms",
                entry=self.subscriber.subscriber_id,
                field="socket",
            ) from exc
        except OSError as exc:
            raise BusTimeoutError(
                f"socket read failed: {exc}",
                entry=self.subscriber.subscriber_id,
                field="socket",
            )

    def _request(self, message):
        # Serialize because several worker threads share one processor socket.
        with self._lock:
            self._send_line(json.dumps(message, sort_keys=True))
            reply = self._read_line()
        return json.loads(reply)

    def grant_lease(self, lease):
        message = {
            "kind": "LEASE_GRANT",
            "lease_id": lease.lease_id,
            "subscriber_id": lease.subscriber_id,
            "version": lease.version,
            "granted_at_ms": lease.granted_at_ms,
            "expires_at_ms": lease.expires_at_ms,
        }
        reply = self._request(message)
        if not reply.get("ok"):
            raise LeaseError(
                reply.get("reason", "lease grant rejected"),
                entry=lease.lease_id,
                field="lease",
            )
        return reply

    def renew_lease(self, lease_id, expected_version, new_expires_at_ms):
        message = {
            "kind": "LEASE_RENEW",
            "lease_id": lease_id,
            "expected_version": expected_version,
            "new_expires_at_ms": new_expires_at_ms,
        }
        reply = self._request(message)
        if not reply.get("ok"):
            raise LeaseError(
                reply.get("reason", "lease renew rejected"),
                entry=lease_id,
                field="version",
                current_version=reply.get("current_version"),
            )
        return reply

    def send_batch(self, batch, payloads, lease, work_ms, formats, covered_refs):
        message = {
            "kind": "BATCH",
            "batch_id": batch.batch_id,
            "lease_id": lease.lease_id,
            "event_seqs": list(batch.event_seqs),
            "payloads": list(payloads),
            "work_ms": work_ms,
            "formats": formats,
            "covered_refs": covered_refs,
        }
        reply = self._request(message)
        if not reply.get("ok"):
            if reply.get("code") == "E_LEASE":
                raise LeaseError(
                    reply.get("reason", "batch lease rejected"),
                    entry=batch.batch_id,
                    field="lease",
                )
            raise BusTimeoutError(
                reply.get("reason", "batch failed"),
                entry=batch.batch_id,
                field="processor",
            )
        return reply

    def _terminate_process(self):
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=3)
            if self._process.stdout is not None:
                self._process.stdout.close()
            self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def stop(self):
        try:
            if self._sock is not None:
                self._send_line("BYE")
        except Exception:
            pass
        try:
            if self._sock is not None:
                self._sock.close()
        finally:
            self._sock = None
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=3)
            if self._process.stdout is not None:
                self._process.stdout.close()
            self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


class ProcessorPool:
    def __init__(self, spec, processor_dir=None):
        self.spec = spec
        self.processor_dir = processor_dir
        self._clients: dict[str, ProcessorClient] = {}

    def __enter__(self):
        for subscriber in self.spec.subscribers:
            log_path = None
            if self.processor_dir:
                log_path = str(
                    Path(self.processor_dir)
                    / f"processor-{subscriber.subscriber_id}.log"
                )
                decl_path = (
                    Path(self.processor_dir)
                    / f"subscriber-{subscriber.subscriber_id}.json"
                )
                decl_path.parent.mkdir(parents=True, exist_ok=True)
                decl_path.write_text(
                    json.dumps(subscriber.__dict__, sort_keys=True, indent=2),
                    encoding="utf-8",
                )
            client = ProcessorClient(subscriber, log_path)
            client.start()
            self._clients[subscriber.subscriber_id] = client
        return self

    def get(self, subscriber_id) -> ProcessorClient:
        return self._clients[subscriber_id]

    def __exit__(self, *exc):
        for client in self._clients.values():
            client.stop()
        self._clients.clear()
        return False
