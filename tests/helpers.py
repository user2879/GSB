"""Shared test helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cli(session_path: Path, *extra: str) -> tuple[int, dict]:
    command = [sys.executable, "-m", "streamsim", "run", str(session_path), *extra]
    process = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = process.stdout.strip()
    if not stdout:
        raise AssertionError(
            f"CLI produced no JSON. exit={process.returncode} stderr={process.stderr}"
        )
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"CLI stdout is not JSON: {stdout[:500]} stderr={process.stderr[:500]}"
        ) from exc
    return process.returncode, payload


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def session_dirs(session_path: Path) -> tuple[Path, Path]:
    payload = read_json(session_path)
    base = session_path.parent
    state = base / payload["state_dir"]
    snapshots = base / payload["snapshot_dir"]
    return state, snapshots


def fragment(size_bytes: int, bitrate_bps: int) -> dict:
    return {"bitrate_bps": bitrate_bps, "size_bytes": size_bytes}


def make_session_fixture(
    root: Path,
    *,
    fragments: list[dict] | None = None,
    trace: list[dict] | None = None,
    client: dict | None = None,
    events: list[dict] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    media = {
        "fragments": fragments
        or [
            {"duration_ms": 2000, "representations": [fragment(50_000, 200_000)]},
            {"duration_ms": 2000, "representations": [fragment(50_000, 200_000)]},
        ]
    }
    network = {"segments": trace or [{"bandwidth_bps": 2_000_000, "rtt_ms": 20}]}
    event_script = {"events": events or []}
    client_payload = {
        "startup_buffer_ms": 1000,
        "max_buffer_ms": 4000,
        "safety_margin_ms": 200,
        "strategy": "hybrid",
        "switch_penalty": 0.2,
        "concurrency": 1,
        "memory_budget_bytes": 5_000_000,
        "throughput_safety_ratio": 0.9,
    }
    client_payload.update(client or {})
    write_json(root / "media.json", media)
    write_json(root / "network.json", network)
    write_json(root / "events.json", event_script)
    session_path = root / "session.json"
    write_json(
        session_path,
        {
            "media_manifest": "media.json",
            "network_trace": "network.json",
            "event_script": "events.json",
            "client": client_payload,
            "state_dir": "state",
            "snapshot_dir": "snapshots",
        },
    )
    return session_path
