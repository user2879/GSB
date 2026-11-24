"""Subprocess supervision and loopback HTTP calls for deterministic workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import load_config
from .jsonutil import canonical_dumps
from .models import ConfigBundle, RequestSpec, WorkerSpec


@dataclass
class WorkerProcess:
    worker: WorkerSpec
    process: subprocess.Popen[bytes]
    stdout_path: Path
    stderr_path: Path

    def stop(self, timeout: float = 5.0) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout)


def _post_json(url: str, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
    body = canonical_dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("worker returned a non-object response")
    return parsed


def _health(worker: WorkerSpec, timeout: float = 0.5) -> bool:
    url = f"http://{worker.host}:{worker.port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("status") == "ok" and payload.get("worker_id") == worker.worker_id
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def start_workers(
    config_path: str | Path,
    *,
    worker_ids: Iterable[str] | None = None,
    startup_timeout: float = 10.0,
) -> tuple[ConfigBundle, list[WorkerProcess]]:
    result = load_config(config_path)
    if result.bundle is None:
        raise RuntimeError("; ".join(f"{item.code}: {item.message}" for item in result.failures))
    bundle = result.bundle
    root = Path(bundle.root)
    log_dir = root / "reports" / "worker-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    selected = set(worker_ids) if worker_ids is not None else {worker.worker_id for worker in bundle.workers}
    processes: list[WorkerProcess] = []
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    package_root = Path(__file__).resolve().parent.parent
    path_parts = [str(root), str(package_root)]
    if existing_pythonpath:
        path_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)
    try:
        for worker in bundle.workers:
            if worker.worker_id not in selected:
                continue
            stdout_path = log_dir / f"{worker.worker_id}.stdout.log"
            stderr_path = log_dir / f"{worker.worker_id}.stderr.log"
            stdout_handle = stdout_path.open("wb")
            stderr_handle = stderr_path.open("wb")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mmledger.worker",
                    "--config",
                    str(Path(bundle.config_path)),
                    "--worker-id",
                    worker.worker_id,
                ],
                cwd=str(root),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            stdout_handle.close()
            stderr_handle.close()
            processes.append(WorkerProcess(worker, process, stdout_path, stderr_path))
        deadline = time.monotonic() + startup_timeout
        for item in processes:
            while time.monotonic() < deadline:
                if item.process.poll() is not None:
                    stderr = item.stderr_path.read_text(encoding="utf-8", errors="replace")
                    raise RuntimeError(
                        f"worker {item.worker.worker_id} exited early with {item.process.returncode}: {stderr}"
                    )
                if _health(item.worker):
                    break
                time.sleep(0.05)
            else:
                raise TimeoutError(f"worker {item.worker.worker_id} did not become healthy")
    except Exception:
        for item in processes:
            item.stop()
        raise
    return bundle, processes


def infer_batch(worker: WorkerSpec, batch_id: str, requests: Iterable[RequestSpec], *, timeout: float = 10.0) -> Mapping[str, Any]:
    payload = {
        "batch_id": batch_id,
        "requests": [request.as_dict() for request in requests],
    }
    return _post_json(f"http://{worker.host}:{worker.port}/infer", payload, timeout)


def stop_workers(processes: Iterable[WorkerProcess]) -> None:
    for item in reversed(list(processes)):
        item.stop()
