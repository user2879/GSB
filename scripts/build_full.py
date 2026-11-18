#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from session_pipeline.runner import run_full_build  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply all edits and run one serial full rebuild")
    parser.add_argument("--scenario", type=Path, default=ROOT / "samples" / "scenario_01")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "full-report.json")
    parser.add_argument("--parallel", action="store_true", help="Use configured workers instead of serial mode")
    parser.add_argument("--injections", action="store_true", help="Enable independent fault injection config")
    args = parser.parse_args()
    report = run_full_build(
        scenario_dir=args.scenario,
        report_path=args.report,
        serial=not args.parallel,
        enable_injections=args.injections,
    )
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "sessions": report["summary"]["session_count"],
                "committed_sessions": report["summary"]["committed_session_count"],
                "failure_count": report["failure_count"],
                "state_projection_hash": report["state_projection_hash"],
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

