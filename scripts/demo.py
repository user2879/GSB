"""End-to-end local demonstration using the bundled samples."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drift_diag.config import EvaluationConfig
from drift_diag.ledger import LedgerStore
from drift_diag.replay import verify_replay
from drift_diag.session import run_session


def main() -> int:
    workdir = ROOT / ".demo-work"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    db_path = workdir / "ledger.sqlite3"
    config = EvaluationConfig(
        scale_exp=3,
        tolerance_num=1,
        tolerance_den=6,
        global_limit=3,
        per_workload_quota=1,
        task_delay_ms=2.0,
        batch_events=4,
    ).validate()
    session = run_session(
        declaration_path=ROOT / "samples" / "declaration-v3.json",
        events_path=ROOT / "samples" / "events.ndjson",
        db_path=db_path,
        defaults_version="defaults-v1",
        config=config,
    )
    with LedgerStore(db_path, read_only=True) as ledger:
        replay = verify_replay(ledger, session.target, session.events, config)
    print(f"ledger={db_path}")
    print(f"batches={len(session.batches)} latest_cursor={session.latest_batch['event_cursor']}")
    print(f"overall_converged={session.overall_converged}")
    print(f"replay_ok={replay.ok} decisions_checked={replay.decisions_checked}")
    print("run `python -m drift_diag gui --db %s` on Windows to open the diagnostic view" % db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
