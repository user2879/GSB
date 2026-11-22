from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json

from driftwatch.config import load_inputs
from driftwatch.engine import DriftEngine
from driftwatch.errors import DriftError
from driftwatch.processes import port_is_free


ROOT = Path(__file__).resolve().parent


def _summary(report, label: str) -> dict:
    return {
        "label": label,
        "generation": report.generation,
        "replaced": report.replaced,
        "view": report.view_summary,
        "affected_count": len(report.affected),
        "affected_sample": report.affected[:4],
        "differences": report.differences[:6],
        "replay": report.replay,
        "cursor": report.cursor,
    }


def _reset_database(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete deterministic DriftWatch example")
    parser.add_argument("--db", default=str(ROOT / "demo-data" / "driftwatch.sqlite3"))
    parser.add_argument("--reset", action="store_true", help="delete the example SQLite ledger before starting")
    args = parser.parse_args()
    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    if args.reset:
        _reset_database(db)

    v1 = load_inputs(ROOT / "snapshots.json", ROOT / "agents_v1.json", ROOT / "criteria.json")
    occupied = [agent.port for agent in v1[1] if not port_is_free(agent.host, agent.port)]
    if occupied:
        raise DriftError.port("example ports are already occupied", occupied=occupied)

    outputs = []
    engine = DriftEngine(v1[0], v1[1], v1[2], str(db))
    try:
        outputs.append(_summary(engine.run(), "v1: three snapshots, sparse edge-c, fixed-point drift"))
    finally:
        engine.close()

    v2 = load_inputs(ROOT / "snapshots_v2.json", ROOT / "agents_v2.json", ROOT / "criteria.json")
    occupied = [agent.port for agent in v2[1] if not port_is_free(agent.host, agent.port)]
    if occupied:
        raise DriftError.port("example ports are already occupied", occupied=occupied)
    engine = DriftEngine(v2[0], v2[1], v2[2], str(db))
    try:
        outputs.append(_summary(engine.run(), "v2: agent/aggregation/snapshot evolution with whole-view replacement"))
        outputs.append(_summary(engine.run(verify_replay=True), "v2: deterministic ledger replay verification"))
        view = engine.ledger.session_snapshot(engine.ledger.active_generation() or 0)
        outputs.append(
            {
                "label": "active row decisions",
                "rows": [
                    {
                        "snapshot": row["snapshot_seq"],
                        "host": row["host_id"],
                        "metric": row["metric_id"],
                        "decision": row["decision"],
                        "observed": row["observed"],
                        "expected": row["expected"],
                        "tolerance_bps": row["tolerance_bps"],
                        "error_affected": row["error_affected"],
                    }
                    for row in view["drift_view"]
                ],
            }
        )
    finally:
        engine.close()

    print(json.dumps({"ok": True, "db": str(db), "runs": outputs}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
