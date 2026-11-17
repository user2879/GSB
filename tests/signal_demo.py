"""Drive a CLI run in a subprocess and send SIGTERM mid-flight."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    output = os.path.join(ROOT, "outputs", "signal_demo")
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
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(2.0)
    proc.send_signal(signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=60)
    print("returncode:", proc.returncode)
    try:
        print(json.dumps(json.loads(stdout), indent=2, sort_keys=True))
    except json.JSONDecodeError:
        print("stdout:", stdout)
        print("stderr:", stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
