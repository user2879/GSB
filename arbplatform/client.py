"""Parent-side subprocess supervisor and real TCP client."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

from .errors import PortError, ReplyError, TimeoutError
from .protocol import recv_line, send_line, validate_reply


@dataclass
class ServiceHandle:
    env_id: str
    host: str
    port: int
    process: subprocess.Popen


class ServiceSupervisor:
    """Spawns one deterministic service subprocess per environment."""

    def __init__(self, ready_timeout_s: float = 5.0, io_timeout_s: float = 3.0,
                 poll_interval_s: float = 0.01, process_factory=None) -> None:
        self.ready_timeout_s = ready_timeout_s
        self.io_timeout_s = io_timeout_s
        self.poll_interval_s = poll_interval_s
        self.process_factory = process_factory or self._default_process_factory
        self.handles: dict[str, ServiceHandle] = {}

    @staticmethod
    def _default_process_factory(env_id: str, host: str, port: int):
        return subprocess.Popen(
            [sys.executable, "-m", "arbplatform.service", env_id, host, str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def start(self, env_id: str, host: str, port: int) -> ServiceHandle:
        # Fail fast with E_PORT when the loopback port is already occupied.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.5)
            if probe.connect_ex((host, port)) == 0:
                raise PortError(
                    f"loopback port {port} already in use",
                    field=f"environments.{env_id}.port",
                )
        finally:
            probe.close()

        process = self.process_factory(env_id, host, port)
        assert process.stdout is not None
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                err = process.stderr.read() if process.stderr else ""
                raise PortError(
                    f"service for {env_id} exited during startup: {err.strip()}",
                    field=f"environments.{env_id}.port",
                )
            line = process.stdout.readline()
            if line.startswith("READY"):
                handle = ServiceHandle(env_id, host, port, process)
                self.handles[env_id] = handle
                return handle
            time.sleep(self.poll_interval_s)
        process.kill()
        raise TimeoutError(
            f"handshake timeout waiting for READY from {env_id}",
            field=f"environments.{env_id}.port",
        )

    def _connect(self, handle: ServiceHandle) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Real socket timeout drives both connect and IO; no artificial sleep.
        sock.settimeout(self.io_timeout_s)
        try:
            sock.connect((handle.host, handle.port))
        except socket.timeout as exc:
            sock.close()
            raise TimeoutError(
                f"connect timeout to {handle.env_id}:{handle.port}",
                field=f"environments.{handle.env_id}.port",
            ) from exc
        except OSError as exc:
            sock.close()
            raise PortError(
                f"cannot connect to {handle.env_id}:{handle.port}: {exc}",
                field=f"environments.{handle.env_id}.port",
            ) from exc
        return sock

    def request(self, env_id: str, message: dict[str, Any],
                expect_reply: bool = True) -> dict[str, Any]:
        handle = self.handles.get(env_id)
        if handle is None:
            raise PortError(f"no running service for env {env_id}",
                            field=f"environments.{env_id}")
        sock = self._connect(handle)
        try:
            send_line(sock, message)
            if not expect_reply:
                return {}
            try:
                reply = recv_line(sock)
            except socket.timeout as exc:
                raise TimeoutError(
                    f"IO timeout awaiting reply from {env_id}",
                    field=f"environments.{env_id}",
                ) from exc
            except ConnectionError as exc:
                raise ReplyError(
                    f"connection broken awaiting reply from {env_id}: {exc}",
                    field=f"environments.{env_id}",
                ) from exc
            return validate_reply(reply)
        finally:
            sock.close()

    def stop(self, env_id: str) -> None:
        handle = self.handles.pop(env_id, None)
        if handle is None:
            return
        if handle.process is None:
            return
        process = handle.process
        try:
            self.request(env_id, {"type": "shutdown"})
        except (PortError, TimeoutError, ReplyError, OSError):
            handle.process.kill()
        try:
            handle.process.wait(timeout=self.ready_timeout_s)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def stop_all(self) -> None:
        for env_id in sorted(self.handles):
            self.stop(env_id)

    def force_kill(self, env_id: str) -> int:
        """Simulate forced termination; returns the exit code."""
        handle = self.handles.pop(env_id, None)
        if handle is None:
            raise PortError(f"no running service for env {env_id}",
                            field=f"environments.{env_id}")
        process = handle.process
        process.kill()
        exit_code = process.wait()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        return exit_code
