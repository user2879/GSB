#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from session_pipeline.runner import (  # noqa: E402
    projection_from_report,
    run_full_build,
    run_incremental,
)
from session_pipeline.util import atomic_write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Force-terminate after a batch, resume, and compare with full rebuild")
    parser.add_argument("--scenario", type=Path, default=ROOT / "samples" / "scenario_01")
    parser.add_argument("--workdir", type=Path, default=ROOT / "artifacts" / "recovery-work")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "recovery-report.json")
    args = parser.parse_args()
    if args.workdir.exists():
        shutil.rmtree(args.workdir)
    args.workdir.parent.mkdir(parents=True, exist_ok=True)
    crash_report = args.workdir.parent / "recovery-crash-report.json"
    resume_report = args.workdir.parent / "recovery-resume-report.json"
    for path in (crash_report, resume_report):
        if path.exists():
            path.unlink()

    child = (
        "from pathlib import Path;"
        "from session_pipeline.runner import run_incremental;"
        f"run_incremental(scenario_dir=Path({str(args.scenario)!r}),"
        f"workdir=Path({str(args.workdir)!r}),"
        f"report_path=Path({str(crash_report)!r}),reset=True,crash_after_batch=1)"
    )
    forced = subprocess.run(
        [sys.executable, "-c", child],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    resumed, resume_exit_code = run_incremental(
        scenario_dir=args.scenario,
        workdir=args.workdir,
        report_path=resume_report,
        reset=False,
    )
    full = run_full_build(
        scenario_dir=args.scenario,
        report_path=args.workdir.parent / "recovery-full-report.json",
        serial=True,
    )
    equal = projection_from_report(resumed) == projection_from_report(full)
    result = {
        "forced_exit_code": forced.returncode,
        "crash_report_present": crash_report.exists(),
        "resume_exit_code": resume_exit_code,
        "recovery": resumed["recovery"],
        "resumed_snapshot": resumed["snapshot"],
        "resumed_equal_full": equal,
        "resumed_state_projection_hash": resumed["state_projection_hash"],
        "full_state_projection_hash": full["state_projection_hash"],
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if forced.returncode == 137 and not crash_report.exists() and equal else 1


if __name__ == "__main__":
    raise SystemExit(main())

