"""Main-side subprocess lifecycle and real loopback socket client."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .codes import PortError, ReplyError, TimeoutError_
from .worker import REQUIRED_REPLY_FIELDS


@dataclass
class ServiceProcess:
    service_id: str
    host: str
    port: int
    process: subprocess.Popen
    sock: socket.socket
    io_lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        try:
            self.sock.close()
        finally:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for stream in (getattr(self.process, "stdout", None),
                           getattr(self.process, "stderr", None)):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass


class ServiceClient:
    def __init__(self, *, connect_timeout_ms: int = 1500,
                 read_timeout_ms: int = 4000,
                 handshake_timeout_ms: int = 4000):
        self.connect_timeout_ms = connect_timeout_ms / 1000.0
        self.read_timeout_ms = read_timeout_ms / 1000.0
        self.handshake_timeout_ms = handshake_timeout_ms / 1000.0

    def spawn_and_connect(self, *, service_id: str, host: str, port: int,
                          config_version: int, seed: int, ttl_ms: int,
                          python_exec: str | None = None) -> ServiceProcess:
        if host != "127.0.0.1":
            raise PortError("only 127.0.0.1 loopback is permitted",
                            host=host, service_id=service_id)
        # Detect an already-bound port before launching, mapping it to E_PORT.
        self._assert_port_free(host, port, service_id)

        cmd = [
            python_exec or sys.executable,
            "-m", "rlcb.worker",
            "--service-id", service_id,
            "--host", host,
            "--port", str(port),
            "--config-version", str(config_version),
            "--seed", str(seed),
            "--ttl-ms", str(ttl_ms),
        ]
        process = subprocess.Popen(
            cmd, cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        # The worker prints a listening line; but the authoritative readiness
        # probe is a real socket connect + hello with a timeout.
        deadline = time.monotonic() + self.handshake_timeout_ms
        sock: socket.socket | None = None
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                err = process.stderr.read() if process.stderr else ""
                raise PortError(
                    f"service subprocess {service_id} exited before listening",
                    service_id=service_id, port=port, stderr=err.strip(),
                )
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.settimeout(self.connect_timeout_ms)
            try:
                probe.connect((host, port))
                sock = probe
                break
            except (ConnectionRefusedError, socket.timeout, OSError) as exc:
                probe.close()
                last_error = exc
                time.sleep(0.02)
        if sock is None:
            process.terminate()
            raise TimeoutError_(
                f"handshake timeout connecting to {service_id} on {host}:{port}",
                service_id=service_id, port=port,
                last_error=type(last_error).__name__ if last_error else None,
            )
        sp = ServiceProcess(service_id, host, port, process, sock)
        try:
            reply = self.request(sp, {"kind": "hello", "batch_id": ""},
                                 timeout_s=self.handshake_timeout_ms)
        except Exception:
            sp.close()
            raise
        if not reply.get("ready"):
            sp.close()
            raise ReplyError("handshake reply missing ready flag",
                             service_id=service_id, reply=reply)
        return sp

    def _assert_port_free(self, host: str, port: int, service_id: str) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(self.connect_timeout_ms)
        try:
            probe.connect((host, port))
        except ConnectionRefusedError:
            return
        except socket.timeout:
            return
        except OSError:
            return
        else:
            probe.close()
            raise PortError(
                f"port {port} already in use for {service_id}",
                service_id=service_id, host=host, port=port,
            )
        finally:
            try:
                probe.close()
            except OSError:
                pass

    def request(self, sp: ServiceProcess, message: dict[str, Any],
                timeout_s: float | None = None) -> dict[str, Any]:
        # Serialize request/response on this single connection so concurrent
        # worker threads cannot interleave frames. Threads still run in a real
        # pool and each service still uses its own real loopback socket.
        with sp.io_lock:
            return self._request_locked(sp, message, timeout_s)

    def _request_locked(self, sp: ServiceProcess,
                        message: dict[str, Any],
                        timeout_s: float | None) -> dict[str, Any]:
        sock = sp.sock
        sock.settimeout(timeout_s or self.read_timeout_ms)
        try:
            sock.sendall((json.dumps(message, sort_keys=True) + "\n")
                         .encode("utf-8"))
            f = sock.makefile("rb")
            raw = f.readline()
        except socket.timeout as exc:
            raise TimeoutError_(
                f"socket read timeout for {sp.service_id}",
                service_id=sp.service_id, port=sp.port,
                kind=message.get("kind"),
            ) from exc
        except OSError as exc:
            raise ReplyError(
                f"socket error for {sp.service_id}: {exc}",
                service_id=sp.service_id, port=sp.port,
            ) from exc
        if not raw:
            raise ReplyError("empty reply; service closed connection",
                             service_id=sp.service_id, port=sp.port)
        try:
            reply = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ReplyError("reply is not valid JSON",
                             service_id=sp.service_id,
                             raw=raw[:120].decode("utf-8", "replace")) from exc
        missing = [name for name in REQUIRED_REPLY_FIELDS if name not in reply]
        if missing:
            raise ReplyError(
                f"reply missing required fields {missing}",
                service_id=sp.service_id, port=sp.port, missing=missing,
            )
        if reply.get("service_id") != sp.service_id:
            raise ReplyError("reply service_id mismatch",
                             expected=sp.service_id,
                             got=reply.get("service_id"))
        return reply
