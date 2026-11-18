#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from session_pipeline.runner import run_incremental  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the independent fault-injection demonstration")
    parser.add_argument("--scenario", type=Path, default=ROOT / "samples" / "scenario_01")
    parser.add_argument("--workdir", type=Path, default=ROOT / "artifacts" / "fault-work")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "fault-report.json")
    args = parser.parse_args()
    if args.workdir.exists():
        shutil.rmtree(args.workdir)
    report, _ = run_incremental(
        scenario_dir=args.scenario,
        workdir=args.workdir,
        report_path=args.report,
        reset=True,
        enable_injections=True,
    )
    print(
        json.dumps(
            {
                "injection_hits": report["injection_hits"],
                "failures": report["failures"],
                "session_status": {
                    key: value["status"] for key, value in report["sessions"].items()
                },
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    expected = {item["injection_id"] for item in report["injection_hits"]}
    return 0 if len(expected) == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())

