from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .models import Batch, WorkerSpec
from .net import call_worker


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WorkerProcess:
    def __init__(self, worker: WorkerSpec, spec_dir: str | Path, startup_timeout: float = 8.0) -> None:
        self.worker = worker
        self.spec_dir = Path(spec_dir).resolve()
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> "WorkerProcess":
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "receipt_ledger.worker",
                "--spec",
                str(self.spec_dir),
                "--worker-id",
                self.worker.worker_id,
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + self.startup_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stderr = b""
                if self.process.stderr is not None:
                    stderr = self.process.stderr.read()
                raise RuntimeError(f"worker {self.worker.worker_id} exited early: {stderr.decode('utf-8', 'replace')}")
            try:
                response = call_worker(self.worker.port, {"op": "health"}, host=self.worker.host, timeout=0.5)
                if response.get("worker_id") == self.worker.worker_id:
                    return self
            except OSError as exc:
                last_error = exc
                time.sleep(0.05)
        self.kill()
        raise TimeoutError(f"worker {self.worker.worker_id} did not become ready: {last_error}")

    def forward(self, batch: Batch, timeout: float = 10.0) -> dict[str, Any]:
        response = call_worker(
            self.worker.port,
            {"op": "forward", "batch": batch.to_json()},
            host=self.worker.host,
            timeout=timeout,
        )
        return response["result"]

    def terminate(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.kill()

    def kill(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.kill()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

    def __enter__(self) -> "WorkerProcess":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.kill()


class WorkerPool:
    def __init__(self, workers: tuple[WorkerSpec, ...] | list[WorkerSpec], spec_dir: str | Path) -> None:
        self.processes = {worker.worker_id: WorkerProcess(worker, spec_dir) for worker in workers}

    def start_all(self) -> "WorkerPool":
        for process in self.processes.values():
            process.start()
        return self

    def forward(self, worker_id: str, batch: Batch) -> dict[str, Any]:
        return self.processes[worker_id].forward(batch)

    def kill_worker(self, worker_id: str) -> None:
        self.processes[worker_id].kill()

    def restart_worker(self, worker_id: str) -> WorkerProcess:
        process = self.processes[worker_id]
        process.kill()
        new_process = WorkerProcess(process.worker, process.spec_dir).start()
        self.processes[worker_id] = new_process
        return new_process

    def stop_all(self) -> None:
        for process in self.processes.values():
            process.kill()

    def __enter__(self) -> "WorkerPool":
        return self.start_all()

    def __exit__(self, *exc: object) -> None:
        self.stop_all()
