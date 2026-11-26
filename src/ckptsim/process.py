from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import CkptError
from .protocol import TcpShardClient
from .spec import ShardSpec


def _pythonpath() -> str:
    src = str(Path(__file__).resolve().parents[1])
    existing = os.environ.get("PYTHONPATH", "")
    return src + (os.pathsep + existing if existing else "")


def check_ports_free(shards: Iterable[ShardSpec]) -> None:
    import socket

    for shard in shards:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((shard.host, shard.port))
        except OSError as exc:
            raise CkptError(
                "E_PORT",
                f"loopback port {shard.port} is not free: {exc}",
                item=shard.shard_id,
            ) from exc
        finally:
            probe.close()


@dataclass
class ShardProcess:
    shard: ShardSpec
    shards_path: str
    layers_path: str
    startup_timeout: float = 8.0
    lease_ttl_ms: int = 30_000
    client_timeout: float = 10.0
    process: subprocess.Popen[str] | None = None
    client: TcpShardClient | None = None
    ready: dict[str, object] | None = None
    _stdout_lines: queue.Queue[str] | None = None
    _stderr_lines: queue.Queue[str] | None = None

    def start(self) -> "ShardProcess":
        command = [
            sys.executable,
            "-m",
            "ckptsim.worker",
            "--shards",
            self.shards_path,
            "--layers",
            self.layers_path,
            "--shard-id",
            self.shard.shard_id,
            "--lease-ttl-ms",
            str(self.lease_ttl_ms),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = _pythonpath()
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
            start_new_session=True,
        )
        self._stdout_lines = queue.Queue()
        self._stderr_lines = queue.Queue()
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        threading.Thread(
            target=self._drain,
            args=(self.process.stdout, self._stdout_lines),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._drain,
            args=(self.process.stderr, self._stderr_lines),
            daemon=True,
        ).start()
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            try:
                line = self._stdout_lines.get(timeout=0.05)
            except queue.Empty:
                if self.process.poll() is not None:
                    break
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") == "ready":
                self.ready = payload
                break
        if self.ready is None:
            self.stop()
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline and self._stderr_lines is not None:
                if not self._stderr_lines.empty():
                    break
                time.sleep(0.01)
            error_payload = self._first_error()
            if error_payload is not None:
                raise CkptError(
                    str(error_payload.get("code", "E_PORT")),
                    str(error_payload.get("detail", "worker failed to start")),
                    item=error_payload.get("item"),
                    line=error_payload.get("line"),
                )
            raise CkptError(
                "E_TIMEOUT",
                f"worker for {self.shard.shard_id} did not become ready",
                item=self.shard.shard_id,
            )
        self.client = TcpShardClient(
            host=self.shard.host,
            port=self.shard.port,
            timeout=self.client_timeout,
            connect_timeout=min(3.0, self.startup_timeout),
        )
        try:
            self.client.health()
        except CkptError:
            self.stop()
            raise
        return self

    @staticmethod
    def _drain(stream: object, destination: queue.Queue[str]) -> None:
        try:
            for line in iter(stream.readline, ""):  # type: ignore[attr-defined]
                destination.put(line)
        finally:
            try:
                stream.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    def _first_error(self) -> dict[str, object] | None:
        if self._stderr_lines is None:
            return None
        lines: list[str] = []
        while True:
            try:
                lines.append(self._stderr_lines.get_nowait())
            except queue.Empty:
                break
        for line in reversed(lines):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            error = payload.get("error")
            if isinstance(error, dict):
                return error
        return None

    def stderr_text(self) -> str:
        if self._stderr_lines is None:
            return ""
        lines: list[str] = []
        while True:
            try:
                lines.append(self._stderr_lines.get_nowait())
            except queue.Empty:
                return "".join(lines)

    def kill(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.kill()


class ShardProcessManager:
    def __init__(
        self,
        shards: Iterable[ShardSpec],
        *,
        shards_path: str,
        layers_path: str,
        startup_timeout: float = 8.0,
        lease_ttl_ms: int = 30_000,
        client_timeout: float = 10.0,
    ) -> None:
        self.shards = {shard.shard_id: shard for shard in shards}
        self.shards_path = shards_path
        self.layers_path = layers_path
        self.startup_timeout = startup_timeout
        self.lease_ttl_ms = lease_ttl_ms
        self.client_timeout = client_timeout
        self.processes: dict[str, ShardProcess] = {}

    def start_all(self) -> None:
        try:
            for shard in self.shards.values():
                process = ShardProcess(
                    shard=shard,
                    shards_path=self.shards_path,
                    layers_path=self.layers_path,
                    startup_timeout=self.startup_timeout,
                    lease_ttl_ms=self.lease_ttl_ms,
                    client_timeout=self.client_timeout,
                )
                process.start()
                self.processes[shard.shard_id] = process
        except Exception:
            self.stop_all()
            raise

    def restart(self, shard_id: str) -> ShardProcess:
        old = self.processes.get(shard_id)
        if old is not None:
            old.kill()
        shard = self.shards[shard_id]
        process = ShardProcess(
            shard=shard,
            shards_path=self.shards_path,
            layers_path=self.layers_path,
            startup_timeout=self.startup_timeout,
            lease_ttl_ms=self.lease_ttl_ms,
            client_timeout=self.client_timeout,
        )
        process.start()
        self.processes[shard_id] = process
        return process

    def kill(self, shard_id: str) -> None:
        process = self.processes.get(shard_id)
        if process is not None:
            process.kill()

    def stop_all(self) -> None:
        for process in list(self.processes.values()):
            process.stop()
        self.processes.clear()

    def __enter__(self) -> "ShardProcessManager":
        self.start_all()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop_all()
