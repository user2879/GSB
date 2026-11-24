#!/usr/bin/env python3
"""Minimal package integration example."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probelease import LeaseOrchestrator, load_declarations


def main() -> int:
    samples = REPO_ROOT / "samples" / "config"
    declarations = load_declarations(
        samples / "sessions.json",
        samples / "probes.json",
        samples / "ack_policy.json",
    )
    with tempfile.TemporaryDirectory(prefix="probelease-example-") as temp:
        temp_path = Path(temp)
        with LeaseOrchestrator(declarations, temp_path / "ledger.sqlite3") as orchestrator:
            first = orchestrator.execute_round("alpha")
            orchestrator.pause_session("alpha", "INTEGRATION_PAUSE")
            resumed = orchestrator.resume_session("alpha")
            second = orchestrator.execute_round("alpha")
            snapshot = orchestrator.save_snapshot(temp_path / "snapshot.json")
            report = {
                "code": "OK",
                "first": first.to_dict(),
                "paused_then_resumed_status": resumed["status"],
                "second": {
                    "cycle": second.cycle,
                    "ratio_bp": second.ratio_bp,
                    "lease_version": second.lease_version,
                },
                "lease": orchestrator.query_lease("alpha"),
                "snapshot": snapshot,
            }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
