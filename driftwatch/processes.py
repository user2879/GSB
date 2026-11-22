from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import DriftError
from .models import AgentSpec
from .protocol import handshake, request_shutdown


def _configure_listener_socket(sock: socket.socket) -> None:
    if os.name == "nt":
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def port_is_free(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _configure_listener_socket(sock)
        sock.settimeout(0.5)
        sock.bind((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


@dataclass
class _RunningAgent:
    spec: AgentSpec
    process: subprocess.Popen[bytes]
    spec_path: Path


class AgentProcessManager:
    def __init__(self, agents: Iterable[AgentSpec], handshake_timeout_ms: int = 5000) -> None:
        self.agents = tuple(sorted(agents, key=lambda item: item.agent_id))
        self.handshake_timeout_ms = handshake_timeout_ms
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._running: dict[str, _RunningAgent] = {}
        self.started = False

    def __enter__(self) -> "AgentProcessManager":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def preflight(self) -> None:
        occupied = [
            {"agent_id": agent.agent_id, "host": agent.host, "port": agent.port}
            for agent in self.agents
            if not port_is_free(agent.host, agent.port)
        ]
        if occupied:
            raise DriftError.port("one or more loopback ports are already occupied", occupied=occupied)

    def start(self, *, cleanup_stale: bool = False) -> None:
        if self.started:
            return
        if cleanup_stale:
            for agent in self.agents:
                request_shutdown(agent, 300)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and any(
                not port_is_free(agent.host, agent.port) for agent in self.agents
            ):
                time.sleep(0.05)
        self.preflight()
        self._tempdir = tempfile.TemporaryDirectory(prefix="driftwatch-agents-")
        temp_path = Path(self._tempdir.name)
        env = os.environ.copy()
        package_root = str(Path(__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = package_root if not existing else package_root + os.pathsep + existing
        env["PYTHONUNBUFFERED"] = "1"
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        try:
            for agent in self.agents:
                spec_path = temp_path / f"{agent.agent_id}.json"
                spec_path.write_text(json.dumps(agent.to_dict(), ensure_ascii=False), encoding="utf-8")
                command = [
                    sys.executable,
                    "-m",
                    "driftwatch.agent_main",
                    "--spec",
                    str(spec_path),
                    "--parent-pid",
                    str(os.getpid()),
                ]
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=env,
                    creationflags=creationflags,
                    start_new_session=os.name != "nt",
                )
                self._running[agent.agent_id] = _RunningAgent(agent, process, spec_path)
            for agent in self.agents:
                self._wait_for_handshake(agent)
            self.started = True
        except BaseException:
            self.stop()
            raise

    def _wait_for_handshake(self, agent: AgentSpec) -> None:
        running = self._running[agent.agent_id]
        try:
            handshake(
                agent,
                self.handshake_timeout_ms,
                retry_until_deadline=True,
                deadline_code="E_TIMEOUT",
            )
        except DriftError as exc:
            code = running.process.poll()
            if code is not None:
                stderr = self._read_stderr(running)
                error_code = "E_PORT" if "E_PORT" in stderr or "Address already in use" in stderr else "E_PORT"
                raise DriftError.port(
                    "agent process exited before a valid handshake",
                    agent_id=agent.agent_id,
                    port=agent.port,
                    exit_code=code,
                    stderr=stderr.strip()[-2000:],
                ) from exc
            if exc.code == "E_TIMEOUT":
                raise
            raise DriftError.port(
                "agent did not complete a valid loopback handshake",
                agent_id=agent.agent_id,
                port=agent.port,
                cause=exc.to_dict(),
            ) from exc

    @staticmethod
    def _read_stderr(running: _RunningAgent) -> str:
        if running.process.stderr is None:
            return ""
        try:
            return running.process.stderr.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def stop(self) -> None:
        for running in list(self._running.values()):
            if running.process.poll() is not None:
                continue
            request_shutdown(running.spec, 500)
        deadline = time.monotonic() + 2.0
        for running in list(self._running.values()):
            remaining = max(0.05, deadline - time.monotonic())
            try:
                running.process.wait(timeout=min(0.5, remaining))
            except subprocess.TimeoutExpired:
                pass
        for running in list(self._running.values()):
            if running.process.poll() is not None:
                continue
            try:
                running.process.terminate()
            except OSError:
                pass
        for running in list(self._running.values()):
            try:
                running.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    running.process.kill()
                except OSError:
                    pass
                try:
                    running.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            if running.process.stderr is not None:
                try:
                    running.process.stderr.close()
                except OSError:
                    pass
        self._running.clear()
        self.started = False
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def process(self, agent_id: str) -> subprocess.Popen[bytes] | None:
        running = self._running.get(agent_id)
        return running.process if running else None
