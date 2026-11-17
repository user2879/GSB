"""Force-kill (SIGKILL) a run, then verify recovery from complete artifacts."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_kill() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    output = os.path.join(ROOT, "outputs", "kill_demo")
    cmd = [
        sys.executable,
        "-m",
        "tsbench.cli",
        "full-final",
        "--config",
        os.path.join(ROOT, "inputs", "signal_large.yaml"),
        "--updates",
        os.path.join(ROOT, "inputs", "signal_updates.json"),
        "--output",
        output,
    ]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    time.sleep(1.5)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    print("after SIGKILL returncode:", proc.returncode)

    # A rerun against the same output dir must recover via cache reuse.
    rerun = subprocess.run(
        cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120
    )
    print("rerun returncode:", rerun.returncode)
    print(rerun.stdout)

    recovery = subprocess.run(
        [
            sys.executable,
            "-m",
            "tsbench.cli",
            "recovery",
            "--config",
            os.path.join(ROOT, "inputs", "signal_large.yaml"),
            "--output",
            output,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    print("recovery point:")
    print(recovery.stdout)


if __name__ == "__main__":
    run_kill()
