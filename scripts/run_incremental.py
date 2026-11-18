#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from session_pipeline.runner import run_incremental  # noqa: E402


STOP_SIGNAL = False


def _handle_signal(signum, frame):  # noqa: ANN001
    del signum, frame
    global STOP_SIGNAL
    STOP_SIGNAL = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or resume the incremental conversation-state pipeline")
    parser.add_argument("--scenario", type=Path, default=ROOT / "samples" / "scenario_01")
    parser.add_argument("--workdir", type=Path, default=ROOT / "artifacts" / "incremental-work")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "incremental-report.json")
    parser.add_argument("--reset", action="store_true", help="Delete the work directory before starting")
    parser.add_argument("--injections", action="store_true", help="Enable fault_injections.json")
    parser.add_argument("--crash-after-batch", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-batch", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    global STOP_SIGNAL
    STOP_SIGNAL = False
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    report, exit_code = run_incremental(
        scenario_dir=args.scenario,
        workdir=args.workdir,
        report_path=args.report,
        reset=args.reset,
        enable_injections=args.injections,
        stop_requested=lambda: STOP_SIGNAL,
        crash_after_batch=args.crash_after_batch,
        max_batch=args.max_batch,
    )
    print(
        json.dumps(
            {
                "run_status": report["run_status"],
                "batch_number": report["batch_number"],
                "recovery": report["recovery"],
                "snapshot": report["snapshot"],
                "summary": report["summary"],
                "injection_hits": [item.get("injection_id") for item in report["injection_hits"]],
                "failure_count": report["failure_count"],
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

