from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "spec.json"


def free_ports(count: int) -> list[int]:
    ports: list[int] = []
    sockets: list[socket.socket] = []
    try:
        while len(ports) < count:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
            if port >= 1024 and port not in ports:
                ports.append(port)
                sockets.append(sock)
            else:
                sock.close()
    finally:
        for sock in sockets:
            sock.close()
    return ports


def base_payload(ports: list[int] | None = None) -> dict[str, Any]:
    with EXAMPLE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if ports is not None:
        for cluster, port in zip(payload["clusters"], ports):
            cluster["port"] = port
    return copy.deepcopy(payload)


def write_spec(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    entries = [str(ROOT)]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def run_module(args: list[str], *, cwd: Path = ROOT, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=cwd,
        env=child_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def parse_json_stdout(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        return json.loads(process.stdout)
    except Exception as exc:
        raise AssertionError(
            f"stdout was not one JSON document\nstdout={process.stdout!r}\nstderr={process.stderr!r}"
        ) from exc
