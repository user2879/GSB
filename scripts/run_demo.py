"""End-to-end local demo: generate, run both formats and emit reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detbox.cli import main  # noqa: E402

DATASET = ROOT / "data" / "sample"
WORK_V1 = ROOT / "work" / "demo_v1"
WORK_V2 = ROOT / "work" / "demo_v2"


def run() -> int:
    base = [
        "--dataset", str(DATASET),
        "--detection-params", str(DATASET / "detection_params.json"),
        "--match-config", str(DATASET / "match_config.json"),
        "--injections", str(DATASET / "injections.json"),
    ]
    rc = main(["generate", "--dataset", str(DATASET)])
    if rc:
        return rc
    rc = main(["run", *base, "--workdir", str(WORK_V1),
               "--format", "v1", "--report", str(WORK_V1 / "report.json")])
    rc2 = main(["run", *base, "--workdir", str(WORK_V2),
                "--format", "v2", "--report", str(WORK_V2 / "report.json")])
    print(json.dumps({"v1_exit": rc, "v2_exit": rc2}, indent=2))
    return rc or rc2


if __name__ == "__main__":
    raise SystemExit(run())
