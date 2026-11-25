from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def load_sample(name: str) -> dict[str, Any]:
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def dynamic_workers(
    path: Path,
    *,
    ports: tuple[int, int] | None = None,
    worker_set_version: int | None = None,
    alpha_param_version: int | None = None,
    beta_batch_size: int | None = None,
) -> dict[str, Any]:
    workers = copy.deepcopy(load_sample("workers.json"))
    chosen = ports or (free_port(), free_port())
    workers["workers"][0]["port"] = chosen[0]
    workers["workers"][1]["port"] = chosen[1]
    if worker_set_version is not None:
        workers["worker_set_version"] = worker_set_version
    if alpha_param_version is not None:
        workers["workers"][0]["param_version_seq"] = alpha_param_version
    if beta_batch_size is not None:
        workers["workers"][1]["batch_size"] = beta_batch_size
    path.write_text(
        json.dumps(workers, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return workers


def cli_env() -> dict[str, str]:
    env = os.environ.copy()
    source = str(ROOT / "src")
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source if not previous else source + os.pathsep + previous
    return env


def run_cli(*args: str, timeout: float = 40.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "inference_attribution.cli", *args],
        cwd=ROOT,
        env=cli_env(),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def cli_json(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return json.loads(result.stdout)
