"""Subprocess worker that commits partial results then terminates abruptly."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .canonical import write_canonical_json
from .ledger import Ledger
from .trust import document_from_revision_contents
from .verify import validate_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tlstrust deliberate crash worker")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--crash-after", type=int, required=True)
    args = parser.parse_args(argv)
    ledger = Ledger(args.ledger, auto_recover=False)
    current = ledger.current_trust_version()
    if current is None:
        return 2
    document = document_from_revision_contents(json.loads(current["revisions_json"]))
    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    run_id = ledger.start_run(int(current["version"]), snapshot, document.revision_numbers)
    recorded: list[int] = []
    for revision in document.revisions:
        result = validate_snapshot(snapshot, revision)
        ledger.record_revision_result(run_id, result)
        recorded.append(revision.revision)
        write_canonical_json(
            args.marker,
            {
                "run_id": run_id,
                "trust_version": int(current["version"]),
                "recorded_revisions": recorded,
                "completed": False,
            },
        )
        if len(recorded) >= args.crash_after:
            # Deliberately bypass sqlite close/finalization.  No exit report is
            # emitted; only the per-revision commits above survive.
            os._exit(90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

