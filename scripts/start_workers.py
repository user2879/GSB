#!/usr/bin/env python3
"""Start all configured loopback workers as child processes."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / ".deps"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from gsb_ledger.config import load_config  # noqa: E402
from gsb_ledger.util import atomic_write_json  # noqa: E402


def _port_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def main() -> int:
    project = Path(sys.argv[1] if len(sys.argv) > 1 else "sample").resolve()
    config = load_config(project / "config.json")
    runtime_dir = ROOT / "artifacts" / "runtime"
    log_dir = ROOT / "artifacts" / "worker-logs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    paths = [str(ROOT / "src")]
    if (ROOT / ".deps").is_dir():
        paths.insert(0, str(ROOT / ".deps"))
    environment["PYTHONPATH"] = os.pathsep.join(paths + [environment.get("PYTHONPATH", "")])
    processes: dict[str, subprocess.Popen] = {}
    records: list[dict[str, object]] = []
    try:
        for worker in config.workers:
            log_path = log_dir / f"{worker.worker_id}.log"
            log_handle = log_path.open("ab")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gsb_ledger.worker",
                    "--project",
                    str(project),
                    "--worker-id",
                    worker.worker_id,
                ],
                cwd=str(ROOT),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            log_handle.close()
            processes[worker.worker_id] = process
            records.append(
                {
                    "worker_id": worker.worker_id,
                    "pid": process.pid,
                    "host": "127.0.0.1",
                    "port": worker.port,
                    "log": str(log_path.relative_to(ROOT)),
                }
            )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            failed = [
                worker_id
                for worker_id, process in processes.items()
                if process.poll() is not None
            ]
            if failed:
                raise RuntimeError(f"worker process exited during startup: {failed}")
            if all(_port_ready(worker.port) for worker in config.workers):
                break
            time.sleep(0.1)
        else:
            raise TimeoutError("workers did not bind within 10 seconds")
        state = {
            "schema_version": 1,
            "project": str(project),
            "workers": records,
        }
        atomic_write_json(runtime_dir / "workers.json", state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print(f"start_workers failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
