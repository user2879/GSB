from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
CERT_PATH = SAMPLES / "certificates.json"
MODIFIED_CERT_PATH = SAMPLES / "certificates_modified.json"
HANDSHAKE_PATH = SAMPLES / "handshake.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def temporary_directory(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(Path(tempfile.gettempdir()))))


def free_loopback_port() -> int:
    for port in range(24000, 25000):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            probe.close()
    raise RuntimeError("no free loopback port in 24000-24999")


def cli_environment() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not existing else str(ROOT) + os.pathsep + existing
    return env


def run_cli(args: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tlschain.cli", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=cli_environment(),
    )


def parse_last_json(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise AssertionError("command produced no JSON output")
    return json.loads(lines[-1])
