#!/usr/bin/env python3
"""Local SIGTERM/SIGKILL recovery verification script."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "samples" / "generated" / "interrupt.json"
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def command(state: Path, output: Path, delay: int) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "subtitle_engine",
        "--input",
        str(INPUT),
        "--state-dir",
        str(state),
        "--output-dir",
        str(output),
        "--block-delay-ms",
        str(delay),
    ]


def wait_for_block(proc: subprocess.Popen[str], state: Path) -> None:
    deadline = time.time() + 15
    while time.time() < deadline:
        if list(state.glob("*/blocks/[0-9]*.npz")):
            return
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(f"engine exited early: {proc.returncode}\n{stdout}\n{stderr}")
        time.sleep(0.025)
    raise RuntimeError("timed out waiting for first block")


def run_restart(state: Path, output: Path) -> dict:
    proc = subprocess.run(
        command(state, output, 0),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"restart failed: {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return json.loads((output / "report.json").read_text())


def verify(signal_name: str, signal_value: int, base: Path) -> dict:
    state = base / f"{signal_name}-state"
    output = base / f"{signal_name}-output"
    proc = subprocess.Popen(
        command(state, output, 120),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    wait_for_block(proc, state)
    if signal_name == "sigkill":
        proc.kill()
    else:
        proc.send_signal(signal_value)
    stdout, stderr = proc.communicate(timeout=20)
    if signal_name == "sigterm" and proc.returncode != 130:
        raise RuntimeError(f"SIGTERM exit code {proc.returncode}: {stdout}\n{stderr}")
    if (output / "frames.npy").exists():
        raise RuntimeError("interrupted run unexpectedly exported frames")
    report = run_restart(state, output)
    if report["state"]["reused_block_count"] < 1:
        raise RuntimeError("restart did not reuse a persisted block")
    return {
        "signal": signal_name,
        "interrupted_exit_code": proc.returncode,
        "reused_blocks": report["state"]["reused_block_count"],
        "recomputed_blocks": report["state"]["recomputed_block_count"],
        "frame_count": report["frame_count"],
        "status": "ok",
    }


def main() -> int:
    if not INPUT.exists():
        subprocess.run([str(PYTHON), "samples/generate_samples.py"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory(prefix="subtitle-interrupt-") as temporary:
        base = Path(temporary)
        results = [verify("sigterm", signal.SIGTERM, base)]
        if os.name != "nt":
            results.append(verify("sigkill", signal.SIGKILL, base))
    print(json.dumps({"ok": True, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
