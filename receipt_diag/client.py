"""Worker subprocess lifecycle and loopback TCP client."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json
from .errors import DiagnosticError, fail
from .specs import SampleSpec, WorkerSpec


MAX_RESPONSE_BYTES = 8 * 1024 * 1024


@dataclass
class WorkerRuntime:
    spec: WorkerSpec
    process: subprocess.Popen[bytes]


class WorkerPool:
    """Starts one real subprocess per declared worker and opens real sockets."""

    def __init__(self, spec: SampleSpec) -> None:
        self.spec = spec
        self._runtimes: dict[str, WorkerRuntime] = {}

    def __enter__(self) -> "WorkerPool":
        self.start_all()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop_all()

    @property
    def pids(self) -> dict[str, int]:
        return {worker_id: runtime.process.pid for worker_id, runtime in self._runtimes.items()}

    def _worker_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        package_parent = str(Path(__file__).resolve().parent.parent)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            package_parent if not existing else package_parent + os.pathsep + existing
        )
        return environment

    def _start_one(self, worker: WorkerSpec) -> None:
        command = [
            sys.executable,
            "-m",
            "receipt_diag.worker",
            "--host",
            worker.host,
            "--port",
            str(worker.port),
            "--worker-id",
            worker.worker_id,
            "--seed",
            str(worker.seed),
            "--param-version",
            str(worker.param_version),
            "--model-dim",
            str(worker.model_dim),
            "--hidden-dim",
            str(worker.hidden_dim),
            "--parent-pid",
            str(os.getpid()),
            "--io-timeout-ms",
            str(max(self.spec.rules.socket_timeout_ms, 1000)),
        ]
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self._worker_environment(),
            creationflags=creationflags,
        )
        runtime = WorkerRuntime(spec=worker, process=process)
        self._runtimes[worker.worker_id] = runtime
        deadline = time.monotonic() + self.spec.rules.worker_start_timeout_ms / 1000.0
        last_error = "worker did not become ready"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                last_error = f"worker exited during startup with code {process.returncode}"
                break
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            try:
                response = self._rpc(
                    worker,
                    {"op": "ping"},
                    timeout_ms=min(250, remaining_ms),
                )
                if response.get("worker_id") != worker.worker_id:
                    raise OSError("ping returned a different worker_id")
                return
            except (OSError, DiagnosticError) as exc:
                last_error = str(exc)
                time.sleep(0.03)
        self.stop_worker(worker.worker_id)
        fail(
            "E_WORKER",
            f"worker_spec.workers[{worker.worker_id}]",
            f"worker failed to start on {worker.host}:{worker.port}: {last_error}",
        )

    def start_all(self, workers: Iterable[WorkerSpec] | None = None) -> None:
        selected = list(workers if workers is not None else self.spec.workers)
        try:
            for worker in selected:
                self._start_one(worker)
        except BaseException:
            self.stop_all()
            raise

    def _rpc(
        self,
        worker: WorkerSpec,
        payload: dict[str, Any],
        *,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        effective_timeout = self.spec.rules.socket_timeout_ms if timeout_ms is None else timeout_ms
        timeout_seconds = effective_timeout / 1000.0
        try:
            with socket.create_connection(
                (worker.host, worker.port), timeout=timeout_seconds
            ) as connection:
                connection.settimeout(timeout_seconds)
                connection.sendall(canonical_json(payload).encode("utf-8") + b"\n")
                chunks = bytearray()
                while True:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    chunks.extend(chunk)
                    if len(chunks) > MAX_RESPONSE_BYTES:
                        fail(
                            "E_WORKER",
                            f"worker:{worker.worker_id}",
                            "response exceeds maximum message size",
                        )
                    if b"\n" in chunk:
                        break
        except (socket.timeout, TimeoutError) as exc:
            fail(
                "E_WORKER",
                f"worker:{worker.worker_id}",
                f"real socket timeout after {effective_timeout} ms: {exc}",
            )
        except OSError as exc:
            fail(
                "E_WORKER",
                f"worker:{worker.worker_id}",
                f"loopback socket failure: {exc}",
            )
        line = bytes(chunks).split(b"\n", 1)[0]
        if not line:
            fail("E_WORKER", f"worker:{worker.worker_id}", "empty socket response")
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            fail(
                "E_WORKER",
                f"worker:{worker.worker_id}",
                f"invalid socket response: {exc}",
            )
        if not isinstance(response, dict):
            fail("E_WORKER", f"worker:{worker.worker_id}", "response is not an object")
        if not response.get("ok"):
            fail(
                str(response.get("code", "E_WORKER")),
                f"worker:{worker.worker_id}",
                str(response.get("detail", "worker rejected request")),
            )
        return response

    def forward_batch(self, batch_id: str, worker_id: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
        worker = self.spec.worker_by_id[worker_id]
        payload = {
            "op": "forward",
            "batch_id": batch_id,
            "worker_id": worker_id,
            "seed": worker.seed,
            "param_version": worker.param_version,
            "model_dim": worker.model_dim,
            "hidden_dim": worker.hidden_dim,
            "requests": requests,
        }
        return self._rpc(worker, payload)

    def stop_worker(self, worker_id: str) -> None:
        runtime = self._runtimes.pop(worker_id, None)
        if runtime is None:
            return
        process = runtime.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass

    def stop_all(self) -> None:
        for worker_id in sorted(tuple(self._runtimes)):
            self.stop_worker(worker_id)
