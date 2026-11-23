"""Reproducible end-to-end demo on the bundled deterministic samples."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arcore.clock import MonotonicClock  # noqa: E402
from arcore.engine import run_round  # noqa: E402
from arcore.ledger import commit_round, open_ledger, replay_rounds  # noqa: E402
from arcore.spec import load_declaration  # noqa: E402
from arcore.stream import generate_stream  # noqa: E402


def main():
    report = {}
    def work_hook(batch, lease):
        # Real blocking work on the worker thread; small enough to stay fast.
        import time
        time.sleep(0.003)
    for name in ("declaration.json", "declaration_large.json"):
        declaration = load_declaration(ROOT / "sample" / name)
        ledger_path = ROOT / "ledger" / (Path(name).stem + ".sqlite3")
        if ledger_path.exists():
            ledger_path.unlink()
        first = generate_stream(declaration)
        second = generate_stream(declaration)
        assert first.global_digest == second.global_digest
        assert [a.to_dict() for a in first.alerts] == [
            a.to_dict() for a in second.alerts]
        connection, notes = open_ledger(ledger_path)
        clock = MonotonicClock()
        result = run_round(declaration, 1, stream=first, clock=clock,
                           work_hook=work_hook)
        commit_round(connection, 1, result.payload, result.stats,
                     clock.now_ms())
        replayed = run_round(declaration, 1, stream=second).payload
        diffs = replay_rounds(connection, [replayed])
        connection.close()
        report[name] = {
            "open_notes": notes,
            "global_digest": first.global_digest,
            "alert_count": len(first.alerts),
            "path": result.payload["path_decision"]["selected_path"],
            "path_reason": result.payload["path_decision"]["reason"],
            "exact_pairs": result.payload["correlation"]["exact_pair_count"],
            "approx_pairs": result.payload["correlation"]["approximate_pair_count"],
            "error_bounds": result.payload["error_bounds"],
            "suppressed": result.payload["suppression"]["suppressed_count"],
            "top_root_cause": (
                result.payload["root_causes"]["candidates"][0]["service_id"]
                if result.payload["root_causes"]["candidates"] else None),
            "lease_occupancy": result.payload["lease_occupancy"],
            "fixed_point_stats": result.stats,
            "replay_diffs": diffs,
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
