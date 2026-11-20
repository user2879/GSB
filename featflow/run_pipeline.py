"""Convenience entry point so Windows users can run:

    python run_pipeline.py --tier small
    python run_pipeline.py --mode resume --data data --out out/small

No Docker, no external services, no network access required.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from featflow.common import read_json  # noqa: E402
from featflow.run import main as pipeline_main  # noqa: E402


def regenerate(tier: str, data_dir: str) -> None:
    subprocess.check_call(
        [sys.executable, os.path.join(ROOT, "scripts", "generate_inputs.py"),
         "--tier", tier, "--out", data_dir])


if __name__ == "__main__":
    if "--regen" in sys.argv:
        tier = "small"
        if "--tier" in sys.argv:
            tier = sys.argv[sys.argv.index("--tier") + 1]
        regenerate(tier, os.path.join(ROOT, "data"))
        sys.argv = [a for a in sys.argv if a not in ("--regen", "--tier", tier)]
    raise SystemExit(pipeline_main())
