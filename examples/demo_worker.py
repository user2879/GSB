"""Worker process used by the demo so the parent can hard-kill it mid-run.

A forced kill of this process leaves only fully committed ledger transactions;
whatever was in flight stays uncommitted and is reclassified as unknown.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chaoskit.engine import ChaosEngine  # noqa: E402
from chaoskit.spec import load_spec  # noqa: E402


def main() -> int:
    spec_path = sys.argv[1]
    workdir = sys.argv[2]
    max_injections = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9

    spec = load_spec(spec_path)
    events = []

    def on_event(event: dict) -> None:
        events.append(event)
        print(f"[worker] {event}", flush=True)

    engine = ChaosEngine(spec, workdir, on_event=on_event)
    try:
        engine.setup()
        count = 0
        for entry in engine.entries:
            row = engine.ledger.injection_row(entry.index)
            if row is not None and row["status"] == "committed":
                continue
            with engine._parallel_slots:
                engine._run_one(entry)
            count += 1
            if count >= max_injections:
                print("[worker] reached requested partial count, exiting cleanly", flush=True)
                break
        print("[worker] done", flush=True)
    finally:
        engine.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
