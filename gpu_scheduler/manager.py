from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Any

from .diagnostics import JsonLogger
from .errors import PortInUseError
from .spec import PoolSpec, Spec

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def assert_port_available(host: str, port: int) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        connect_result = probe.connect_ex((host, port))
        if connect_result == 0:
            raise PortInUseError(
                f"loopback port {port} already has an active listener",
                locator={"host": host, "port": port, "connect_errno": connect_result},
            )
    finally:
        probe.close()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        raise PortInUseError(
            f"loopback port {port} is already occupied or cannot bind",
            locator={"host": host, "port": port, "errno": exc.errno, "os_error": str(exc)},
        ) from exc
    finally:
        sock.close()


def health_check(host: str, port: int, timeout: float = 1.0) -> dict[str, Any]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", "/health")
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        if response.status != 200:
            raise PortInUseError(
                f"executor health endpoint returned HTTP {response.status}",
                locator={"host": host, "port": port, "status": response.status, "body": body[:200]},
            )
        return json.loads(body)
    except (OSError, json.JSONDecodeError) as exc:
        raise PortInUseError(
            f"cannot reach deterministic executor at {host}:{port}",
            locator={"host": host, "port": port, "error": str(exc)},
        ) from exc
    finally:
        conn.close()


def execute_batch(host: str, port: int, batch: dict[str, Any], *, timeout_seconds: float = 30.0) -> dict[str, Any]:
    body = json.dumps(batch, ensure_ascii=False, sort_keys=True).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=timeout_seconds)
    try:
        conn.request(
            "POST",
            "/execute",
            body=body,
            headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": str(len(body))},
        )
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        if response.status != 200:
            raise RuntimeError(f"executor HTTP {response.status}: {raw[:500]}")
        return json.loads(raw)
    finally:
        conn.close()


class ExecutorManager:
    def __init__(self, spec: Spec, runtime_dir: Path, logger: JsonLogger | None = None):
        self.spec = spec
        self.runtime_dir = Path(runtime_dir)
        self.log_dir = self.runtime_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or JsonLogger(self.log_dir / "manager.jsonl")
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_handles: list[Any] = []

    def start(self) -> None:
        for pool in sorted(self.spec.pools, key=lambda p: p.id):
            assert_port_available(pool.host, pool.port)
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        try:
            import numpy as np

            numpy_libs = Path(np.__file__).resolve().parent.parent / "numpy.libs"
            if numpy_libs.is_dir():
                existing_ld_path = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = str(numpy_libs) + (os.pathsep + existing_ld_path if existing_ld_path else "")
        except Exception:
            pass
        for pool in sorted(self.spec.pools, key=lambda p: p.id):
            out_path = self.log_dir / f"executor-{pool.id}.out.log"
            err_path = self.log_dir / f"executor-{pool.id}.err.log"
            out_handle = out_path.open("ab")
            err_handle = err_path.open("ab")
            self.log_handles.extend([out_handle, err_handle])
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gpu_scheduler.executor",
                    "--host",
                    pool.host,
                    "--port",
                    str(pool.port),
                ],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdin=subprocess.PIPE,
                stdout=out_handle,
                stderr=err_handle,
            )
            self.processes[pool.id] = proc
            self.logger.event("INFO", "executor_spawned", pool_id=pool.id, port=pool.port, pid=proc.pid)
        deadline = time.monotonic() + 15.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            all_healthy = True
            for pool in self.spec.pools:
                proc = self.processes[pool.id]
                if proc.poll() is not None:
                    raise PortInUseError(
                        f"executor subprocess for pool {pool.id} exited early",
                        locator={
                            "pool_id": pool.id,
                            "port": pool.port,
                            "returncode": proc.returncode,
                            "stderr_log": str(self.log_dir / f"executor-{pool.id}.err.log"),
                        },
                    )
                try:
                    health_check(pool.host, pool.port, timeout=0.5)
                except PortInUseError as exc:
                    last_error = exc
                    all_healthy = False
                    break
            if all_healthy:
                return
            time.sleep(0.05)
        raise PortInUseError(
            "timed out waiting for deterministic executors",
            locator={"last_error": str(last_error), "ports": {p.id: p.port for p in self.spec.pools}},
        )

    def client_execute_shard(
        self,
        pool: PoolSpec,
        shard_request: dict[str, Any],
        *,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        batch = {"batch_id": shard_request.get("batch_id"), "shards": [shard_request]}
        response = execute_batch(pool.host, pool.port, batch, timeout_seconds=timeout_seconds)
        results = response.get("results", [])
        if len(results) != 1:
            raise RuntimeError(f"executor returned {len(results)} receipts for one shard")
        return results[0]

    def stop(self) -> None:
        for pool_id, proc in list(self.processes.items()):
            if proc.poll() is None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                    proc.terminate()
                    proc.wait(timeout=2.0)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass
            self.logger.event("INFO", "executor_stopped", pool_id=pool_id, returncode=proc.returncode)
        for handle in self.log_handles:
            try:
                handle.close()
            except Exception:
                pass
        self.processes.clear()

    def __enter__(self) -> "ExecutorManager":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


def occupy_port_until_released(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(1)
    return sock
