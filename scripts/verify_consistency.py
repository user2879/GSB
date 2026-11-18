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

from session_pipeline.loading import load_scenario_base  # noqa: E402
from session_pipeline.runner import (  # noqa: E402
    projection_from_report,
    run_full_build,
    run_incremental,
)
from session_pipeline.util import atomic_write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify every incremental batch against a serial full build")
    parser.add_argument("--scenario", type=Path, default=ROOT / "samples" / "scenario_01")
    parser.add_argument("--workdir", type=Path, default=ROOT / "artifacts" / "verification-work")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "verification.json")
    args = parser.parse_args()

    if args.workdir.exists():
        shutil.rmtree(args.workdir)
    args.workdir.mkdir(parents=True, exist_ok=True)
    base = load_scenario_base(args.scenario)
    batch_count = len(base["edit_plan"].get("batches", []))
    checks: list[dict] = []
    all_equal = True
    previous_incremental: dict | None = None

    for batch_number in range(0, batch_count + 1):
        incremental, _ = run_incremental(
            scenario_dir=args.scenario,
            workdir=args.workdir,
            report_path=args.workdir / f"incremental-{batch_number}.json",
            reset=False,
            enable_injections=False,
            max_batch=batch_number,
        )
        full = run_full_build(
            scenario_dir=args.scenario,
            report_path=args.workdir / f"full-{batch_number}.json",
            serial=True,
            max_edit_batches=batch_number,
        )
        incremental_projection = projection_from_report(incremental)
        full_projection = projection_from_report(full)
        equal = incremental_projection == full_projection
        all_equal = all_equal and equal

        reused_identical = True
        reused_sessions: list[str] = []
        if previous_incremental is not None:
            reused_sessions = incremental.get("reuse", {}).get("reused_sessions", [])
            for session_id in reused_sessions:
                before = projection_from_report(previous_incremental).get(session_id)
                after = incremental_projection.get(session_id)
                if before != after:
                    reused_identical = False
                if incremental.get("sessions", {}).get(session_id, {}).get("reuse", {}).get(
                    "replayed_events", 0
                ) != 0:
                    reused_identical = False
        all_equal = all_equal and reused_identical
        checks.append(
            {
                "batch_number": batch_number,
                "incremental_state_hash": incremental.get("state_projection_hash"),
                "full_state_hash": full.get("state_projection_hash"),
                "equal": equal,
                "reused_sessions": reused_sessions,
                "reused_sessions_identical_without_replay": reused_identical,
                "contamination": {
                    "affected_sessions": incremental.get("contamination", {}).get(
                        "affected_sessions", []
                    ),
                    "dirty_slots": incremental.get("contamination", {}).get("dirty_slots", {}),
                    "dirty_slot_count": incremental.get("contamination", {}).get("dirty_slot_count", 0),
                    "replayed_events": incremental.get("reuse", {}).get("replayed_events", 0),
                },
                "snapshot": incremental.get("snapshot", {}),
            }
        )
        previous_incremental = incremental

    result = {
        "scenario": str(args.scenario),
        "batch_count": batch_count,
        "all_equal": all_equal,
        "comparison_scope": [
            "slot values and confidence",
            "current state",
            "transition trace",
            "rejected transitions",
            "slot change diagnostics",
            "state cache key",
            "cursor and state entry count",
        ],
        "checks": checks,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())

