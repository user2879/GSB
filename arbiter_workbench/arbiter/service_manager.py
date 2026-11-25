"""Parent-side supervisor: spawn environment services and talk over TCP."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import ErrorContext, PortError, TimeoutErrorW
from .protocol import (
    DEFAULT_TIMEOUT_S,
    RESERVED_PORT_BAND,
    connect_loopback,
    recv_line,
    send_line,
)
from .sample import Observation
from .spec import EnvSpec

READY_WAIT_S = 8.0


def _port_candidates(preferred: int, used: set[int]) -> list[int]:
    candidates = [preferred]
    candidates.extend(p for p in RESERVED_PORT_BAND if p not in used and p != preferred)
    return candidates


@dataclass
class ServiceHandle:
    env: EnvSpec
    process: subprocess.Popen
    socket: socket.socket
    receipt: dict
    listener: socket.socket | None = None


class ServiceManager:
    def __init__(self, envs: list[EnvSpec], start_timeout_s: float = READY_WAIT_S):
        self.envs = envs
        self.start_timeout_s = start_timeout_s
        self.handles: dict[str, ServiceHandle] = {}

    def start_all(self) -> dict[str, ServiceHandle]:
        listeners: dict[str, socket.socket] = {}
        used_ports: set[int] = set()
        for env in self.envs:
            # Retry across the reserved band: parallel local workloads may race
            # on individual ports even after a free-port probe.
            last_error: Exception | None = None
            listener = None
            for candidate in _port_candidates(env.port, used_ports):
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    probe.bind((env.host, candidate))
                except OSError as exc:
                    last_error = exc
                    probe.close()
                    continue
                probe.listen(8)
                listener = probe
                object.__setattr__(env, "port", candidate)
                used_ports.add(candidate)
                break
            if listener is None:
                raise PortError(
                    "parent failed to reserve a loopback port after retries",
                    ErrorContext(env.host, str(env.port), str(last_error)),
                )
            listeners[env.env_id] = listener
        for env in self.envs:
            process = self._spawn(env, listeners[env.env_id])
            self._wait_ready(env, process)
            # child has inherited the fd; parent duplicates the listener handle
            # closed here while the child keeps its inherited descriptor.
            sock = connect_loopback(env.host, env.port)
            hello = self._roundtrip(sock, {"command": "HELLO", "env_id": env.env_id})
            if hello.get("env_id") != env.env_id:
                process.kill()
                raise PortError(
                    "handshake returned mismatched environment id",
                    ErrorContext(env.env_id, str(env.port), str(hello)),
                )
            self.handles[env.env_id] = ServiceHandle(
                env, process, sock, hello, listener=listeners[env.env_id]
            )
        return self.handles

    def _spawn(self, env: EnvSpec, listener: socket.socket) -> subprocess.Popen:
        cmd = [
            sys.executable,
            "-m",
            "arbiter.env_service",
            "--env-id",
            env.env_id,
            "--host",
            env.host,
            "--port",
            str(env.port),
            "--lease-capacity",
            str(env.lease_capacity),
            "--fd",
            str(listener.fileno()),
        ]
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                pass_fds=(listener.fileno(),),
                close_fds=True,
            )
        except OSError as exc:
            raise PortError(
                "failed to spawn environment service subprocess",
                ErrorContext(env.env_id, str(env.port), str(exc)),
            ) from exc
        return process

    def _wait_ready(self, env: EnvSpec, process: subprocess.Popen) -> None:
        deadline = time.monotonic() + self.start_timeout_s
        assert process.stdout is not None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                err = process.stderr.read() if process.stderr else ""
                raise PortError(
                    "environment service exited during startup",
                    ErrorContext(env.env_id, str(env.port), err.strip()),
                )
            line = process.stdout.readline()
            if line.startswith("READY"):
                return
            time.sleep(0.02)
        process.kill()
        raise TimeoutErrorW(
            "environment service did not become ready in time",
            ErrorContext(env.env_id, str(env.port), f"timeout={self.start_timeout_s}s"),
        )

    @staticmethod
    def _roundtrip(sock: socket.socket, request: dict) -> dict:
        send_line(sock, request)
        return recv_line(sock)

    def push_batch(self, env_id: str, batch: list[Observation]) -> dict:
        handle = self.handles[env_id]
        payload = [obs.as_dict() for obs in batch]
        return self._roundtrip(
            handle.socket, {"command": "BATCH", "env_id": env_id, "observations": payload}
        )

    def push_in_batches(
        self, env_id: str, observations: list[Observation], batch_size: int
    ) -> list[dict]:
        receipts = []
        for start in range(0, len(observations), max(1, batch_size)):
            receipts.append(
                self.push_batch(env_id, observations[start : start + max(1, batch_size)])
            )
        return receipts

    def arbitrate(self, env_id: str) -> dict:
        handle = self.handles[env_id]
        return self._roundtrip(handle.socket, {"command": "ARBITRATE", "env_id": env_id})

    def shutdown_all(self) -> None:
        for env_id, handle in self.handles.items():
            try:
                send_line(handle.socket, {"command": "SHUTDOWN", "env_id": env_id})
                handle.socket.settimeout(1.0)
                try:
                    handle.socket.recv(4096)
                except OSError:
                    pass
            finally:
                handle.socket.close()
                if handle.listener is not None:
                    handle.listener.close()
                handle.process.wait(timeout=5)
                if handle.process.stdout is not None:
                    handle.process.stdout.close()
                if handle.process.stderr is not None:
                    handle.process.stderr.close()
        self.handles.clear()

    def force_kill_all(self) -> None:
        for handle in self.handles.values():
            handle.process.kill()
            handle.socket.close()
            if handle.process.stdout is not None:
                handle.process.stdout.close()
            if handle.process.stderr is not None:
                handle.process.stderr.close()
        self.handles.clear()
