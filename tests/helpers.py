from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def load_config(name: str = "session.json") -> dict[str, Any]:
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


def write_temp_config(
    temp_root: Path,
    *,
    source_name: str = "session.json",
    ledger: str = "state/ledger.sqlite3",
    snapshots: str = "snapshots",
    backup: str = "backup",
    out: str = "out",
    logs: str = "logs/structured.jsonl",
) -> Path:
    document = load_config(source_name)
    document["paths"] = {
        "ledger": str(temp_root / ledger),
        "snapshots": str(temp_root / snapshots),
        "backup": str(temp_root / backup),
        "out": str(temp_root / out),
        "logs": str(temp_root / logs),
    }
    path = temp_root / source_name
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_cli(*args: str, log_file: Path | None = None, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "mwlab", *args]
    if log_file is not None:
        command.extend(["--log-file", str(log_file)])
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def parse_stdout_json(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"no JSON stdout; stderr={process.stderr!r}")
    return json.loads(lines[-1])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_sample(temp_root: Path, *, config_name: str = "session.json", out_name: str = "out") -> tuple[Path, dict[str, Any]]:
    config = write_temp_config(temp_root, source_name=config_name, out=out_name)
    process = run_cli(
        "run",
        "--config",
        str(config),
        "--records",
        str(SAMPLES / "records.jsonl"),
        "--expected",
        str(SAMPLES / "expected_summary.json"),
        log_file=temp_root / "logs" / "structured.jsonl",
    )
    result = parse_stdout_json(process)
    if process.returncode != 0:
        raise AssertionError(f"sample run failed: {result}")
    return config, result

