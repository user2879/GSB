"""Managed loopback target processes and the in-process delay proxy.

Responsibilities:

* spawn the deterministic mock service as a real child process;
* detect port conflicts and unreachable services as ``E_TARGET``;
* expose runtime parameter changes (error-rate rule) without touching code;
* suspend/resume/terminate the process through real OS mechanisms;
* forward loopback traffic through a proxy that injects connection-level delay.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import process_control as pc
from .errors import TargetError
from .spec import ServiceSpec

_READY_WAIT_SECONDS = 8.0
_COMMAND_TIMEOUT = 5.0


def send_command(host: str, port: int, command: str, timeout: float = _COMMAND_TIMEOUT) -> str:
    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.settimeout(timeout)
        conn.sendall((command + "\n").encode("utf-8"))
        chunks = []
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    if not chunks:
        raise TargetError(f"no response from {host}:{port}")
    return b"".join(chunks).decode("utf-8").strip()


def port_is_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class ManagedService:
    """A mock service child process plus its PID marker for orphan cleanup."""

    def __init__(self, spec: ServiceSpec, runtime_dir: Path) -> None:
        self.spec = spec
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.runtime_dir / f"{spec.service_id}.pid"
        self.proc: Optional[subprocess.Popen] = None
        self.suspended = False

    def _drain_pipes(self) -> None:
        if self.proc is None:
            return
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def start(self) -> None:
        self._cleanup_stale()
        self._drain_pipes()
        if port_is_listening(self.spec.host, self.spec.data_port):
            self._reclaim_known_orphan()
        if port_is_listening(self.spec.host, self.spec.data_port):
            raise TargetError(
                f"service {self.spec.service_id} port {self.spec.data_port} already in use by an unknown process",
                details={"service_id": self.spec.service_id, "port": self.spec.data_port},
            )
        creation_flags = 0
        if sys.platform == "win32":  # pragma: no cover
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        command = [
            sys.executable,
            "-m",
            "chaoskit.mock_service",
            "--service-id",
            self.spec.service_id,
            "--host",
            self.spec.host,
            "--port",
            str(self.spec.data_port),
            "--fixed-cost-ms",
            str(self.spec.fixed_cost_ms),
            "--base-error-bps",
            str(self.spec.base_error_bps),
        ]
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creation_flags,
        )
        self.pid_file.write_text(str(self.proc.pid), encoding="utf-8")
        self._await_ready()

    def _reclaim_known_orphan(self) -> None:
        """Shut down an orphan only when IDENT proves it is this service.

        A hard-killed orchestrator can leave the mock service running. We never
        kill an unknown listener (that could be a real system service); an
        IDENT reply matching our service id proves ownership, so SHUTDOWN is
        safe and keeps everything inside the loopback experiment boundary.
        """
        try:
            reply = send_command(
                self.spec.host, self.spec.data_port, "IDENT", timeout=1.5
            )
        except (TargetError, OSError):
            return
        if reply != f"IDENT {self.spec.service_id}":
            return
        try:
            send_command(self.spec.host, self.spec.data_port, "SHUTDOWN", timeout=1.5)
        except (TargetError, OSError):
            return
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            if not port_is_listening(self.spec.host, self.spec.data_port, timeout=0.25):
                return
            time.sleep(0.1)

    def _await_ready(self) -> None:
        deadline = time.monotonic() + _READY_WAIT_SECONDS
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                stderr = self.proc.stderr.read() if self.proc.stderr else ""
                raise TargetError(
                    f"service {self.spec.service_id} exited during startup (code {self.proc.returncode})",
                    details={"service_id": self.spec.service_id, "stderr": stderr[-800:]},
                )
            if port_is_listening(self.spec.host, self.spec.data_port, timeout=0.25):
                try:
                    reply = send_command(self.spec.host, self.spec.data_port, "IDENT", timeout=1.0)
                    if reply == f"IDENT {self.spec.service_id}":
                        return
                except TargetError:
                    pass
            time.sleep(0.05)
        raise TargetError(
            f"service {self.spec.service_id} did not become ready on {self.spec.host}:{self.spec.data_port}",
            details={"service_id": self.spec.service_id},
        )

    def _cleanup_stale(self) -> None:
        if not self.pid_file.exists():
            return
        try:
            stale_pid = int(self.pid_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            stale_pid = None
        if stale_pid is None:
            self.pid_file.unlink(missing_ok=True)
            return
        if sys.platform == "win32":  # pragma: no cover
            subprocess.run(["taskkill", "/PID", str(stale_pid), "/T", "/F"],
                           capture_output=True, check=False)
        else:
            try:
                os.kill(stale_pid, 9)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
        self.pid_file.unlink(missing_ok=True)

    def command(self, command: str) -> str:
        if self.proc is None or self.proc.poll() is not None:
            raise TargetError(
                f"service {self.spec.service_id} is not running",
                details={"service_id": self.spec.service_id, "state": "terminated"},
            )
        return send_command(self.spec.host, self.spec.data_port, command)

    def apply_error_rule(self, sequence: int, seed: int, error_bps: int) -> None:
        import json
        payload = json.dumps({"seq": sequence, "seed": seed, "error_bps": error_bps}, sort_keys=True)
        reply = self.command(f"INJECT {payload}")
        if reply != "APPLIED":
            raise TargetError(f"service rejected error rule: {reply}")

    def clear_error_rule(self) -> None:
        try:
            self.command("CLEAR")
        except TargetError:
            pass

    def suspend(self) -> None:
        if self.proc is None:
            raise TargetError(f"service {self.spec.service_id} never started")
        pc.suspend_process(self.proc)
        self.suspended = True

    def resume(self) -> None:
        if self.proc is None:
            return
        pc.resume_process(self.proc)
        self.suspended = False

    def terminate(self) -> None:
        if self.proc is None:
            return
        pc.force_terminate(self.proc)
        self._drain_pipes()
        self.suspended = False
        self.pid_file.unlink(missing_ok=True)

    def observed_state(self) -> str:
        """Distinguish suspended vs terminated vs running for the ledger."""
        if self.proc is None:
            return "terminated"
        if self.proc.poll() is not None:
            return "terminated"
        if self.suspended:
            return "suspended"
        if pc.is_suspended(self.proc):
            self.suspended = True
            return "suspended"
        return "running"

    def is_reachable(self, timeout: float = 1.0) -> bool:
        if self.proc is None or self.proc.poll() is not None:
            return False
        # A backlogged listening socket can still complete TCP handshakes while
        # the process is SIGSTOPped, so reachability requires an application
        # response, not merely a successful connect.
        try:
            reply = send_command(self.spec.host, self.spec.data_port, "PING", timeout=timeout)
            return reply.startswith("READY")
        except (TargetError, OSError):
            return False

    def stop(self) -> None:
        try:
            if self.proc is not None and self.proc.poll() is None:
                if self.suspended:
                    self.resume()
                try:
                    send_command(self.spec.host, self.spec.data_port, "SHUTDOWN", timeout=1.5)
                except (OSError, TargetError):
                    pass
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pc.force_terminate(self.proc)
                self._drain_pipes()
        finally:
            self.pid_file.unlink(missing_ok=True)


class DelayProxy:
    """Loopback TCP forwarder that adds a per-connection delay.

    The delay is an in-process runtime parameter (zero by default), so the
    network_delay injection changes forwarding behaviour without any source
    modification of either service.
    """

    def __init__(self, listen_host: str, listen_port: int,
                 target_host: str, target_port: int) -> None:
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.target_host = target_host
        self.target_port = target_port
        self._delay_ms = 0
        self._lock = threading.Lock()
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def delay_ms(self) -> int:
        with self._lock:
            return self._delay_ms

    def set_delay(self, delay_ms: int) -> None:
        with self._lock:
            self._delay_ms = max(0, int(delay_ms))

    def start(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.listen_host, self.listen_port))
        except OSError as exc:
            raise TargetError(
                f"proxy port {self.listen_port} already in use: {exc}",
                details={"port": self.listen_port},
            ) from exc
        server.listen(64)
        server.settimeout(0.25)
        self._server = server
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._pump, args=(client,), daemon=True).start()

    def _pump(self, client: socket.socket) -> None:
        delay = self.delay_ms / 1000.0
        if delay:
            time.sleep(delay)
        try:
            upstream = socket.create_connection((self.target_host, self.target_port), timeout=5)
        except OSError:
            client.close()
            return
        with client, upstream:
            t1 = threading.Thread(target=self._copy, args=(client, upstream), daemon=True)
            t2 = threading.Thread(target=self._copy, args=(upstream, client), daemon=True)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

    @staticmethod
    def _copy(source: socket.socket, destination: socket.socket) -> None:
        try:
            while True:
                data = source.recv(4096)
                if not data:
                    break
                destination.sendall(data)
        except OSError:
            pass
        finally:
            try:
                destination.shutdown(socket.SHUT_WR)
            except OSError:
                pass
