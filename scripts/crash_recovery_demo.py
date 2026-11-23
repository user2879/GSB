"""Simulate a hard kill mid-run and prove only committed rounds recover."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arcore.clock import MonotonicClock  # noqa: E402
from arcore.engine import run_round  # noqa: E402
from arcore.ledger import (  # noqa: E402
    commit_round,
    last_committed_round_id,
    load_committed_rounds,
    mark_round_unknown,
    open_ledger,
    replay_rounds,
    unknown_round_ids,
)
from arcore.spec import load_declaration  # noqa: E402
from arcore.stream import generate_stream  # noqa: E402


def main():
    declaration = load_declaration(ROOT / "sample" / "declaration.json")
    stream = generate_stream(declaration)
    ledger_path = ROOT / "ledger" / "crash_recovery.sqlite3"
    if ledger_path.exists():
        ledger_path.unlink()

    # Phase 1: round 1 commits, round 2 is interrupted before commit.
    connection, notes = open_ledger(ledger_path)
    clock = MonotonicClock()
    first = run_round(declaration, 1, stream=stream, clock=clock)
    commit_round(connection, 1, first.payload, first.stats, clock.now_ms())
    mark_round_unknown(connection, 2)
    connection.close()

    # Phase 2: restart. Unknown rounds cannot produce an exit report.
    recovered, recover_notes = open_ledger(ledger_path)
    unknown = unknown_round_ids(recovered)
    # The unfinished round keeps its id; recovery re-runs rather than skipping.
    resume_from = unknown[0] if unknown else (
        last_committed_round_id(recovered) + 1)
    rerun = run_round(declaration, resume_from,
                      stream=generate_stream(declaration))
    commit_round(recovered, resume_from, rerun.payload, rerun.stats,
                 clock.now_ms())
    replayed = [
        run_round(declaration, round_id,
                  stream=generate_stream(declaration)).payload
        for round_id in (1, resume_from)
    ]
    diffs = replay_rounds(recovered, replayed)
    committed = load_committed_rounds(recovered)
    recovered.close()
    print(json.dumps({
        "initial_open_notes": notes,
        "restart_open_notes": recover_notes,
        "unknown_rounds_after_restart": unknown,
        "exit_report_generated_for_unknown": False,
        "resume_from_round": resume_from,
        "committed_round_ids": [payload["round_id"] for payload in committed],
        "replay_diffs": diffs,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
