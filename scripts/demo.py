"""Repeatable end-to-end demonstration (no GUI / network required).

Run:  python scripts/demo.py
It regenerates samples, executes bounded traversals, exercises budget
exhaustion, cap indistinguishability, staleness/diff and snapshot replay.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphdiag.core.engine import Engine
from graphdiag.core.errors import ReplayError
from graphdiag.core.models import content_digest
from graphdiag.sample_data import generate


def main() -> int:
    paths = generate()
    tiny = paths["tiny"]
    db = tempfile.mktemp(prefix="graphdiag-demo-", suffix=".db")
    print(f"demo database: {db}")

    engine = Engine(db, error_mode="exact")
    print("\n=== 1. load with strict validation ===")
    print(json.dumps(engine.load(tiny), indent=2))

    print("\n=== 2. bounded traversal with error propagation ===")
    view = engine.run(budget_limit=1000, max_workers=2)
    for schedule in view.report.ordered():
        result = schedule.result
        print(
            f"{result.query_id}: visited={result.visited_sequence} "
            f"pruned={len(result.pruned)} budget={result.budget_visited}"
        )
        for target, tr in sorted(result.targets.items()):
            if tr.reached:
                print(
                    f"  target={target} verdict={tr.classification} "
                    f"dist_fp={tr.dist_fp} exact_err_micro={tr.exact_bound_micro} "
                    f"cons_ulp={tr.conservative_bound_ulp} "
                    f"affected={tr.error_affected} dominant={tr.dominant_edge}"
                )

    print("\n=== 3. shared budget exhaustion / monotonic waits ===")
    blocked = Engine(db, error_mode="exact")
    blocked.load(tiny)
    report = blocked.run(budget_limit=3, max_workers=3, wait_timeout=0.1)
    for schedule in report.report.ordered():
        print(
            f"{schedule.query_id}: visited={schedule.visited_nodes} "
            f"exhausted={schedule.budget_exhausted} wait_ms={schedule.wait_duration_ms}"
        )
    blocked.store.close()

    print("\n=== 4. staleness + whole-view replacement diff ===")
    with open(tiny, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["edges"][0]["weight_micro"] = 3_500_000
    modified = tempfile.mktemp(suffix=".json")
    with open(modified, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print("reload:", engine.load(modified))
    engine.run(budget_limit=1000)
    for change in engine.diff_views():
        print(
            f"  {change.query_id} target={change.target} "
            f"delta_micro={change.distance_delta_micro} "
            f"{change.old_classification} -> {change.new_classification}"
        )
    os.unlink(modified)

    print("\n=== 5. snapshot + deterministic replay ===")
    snapshot_id = "demo-snapshot"
    digest = engine.save_snapshot(snapshot_id)
    print("saved digest:", digest[:24], "...")
    print("restore:", engine.restore_snapshot(snapshot_id))

    print("\n=== 6. tamper detection -> E_REPLAY ===")
    row = engine.store.conn.execute(
        "SELECT payload_json FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
    ).fetchone()
    snap = json.loads(row[0])
    qid = snap["conclusions"]["order"][0]
    tkey = next(iter(snap["conclusions"]["schedules"][qid]["result"]["targets"]))
    snap["conclusions"]["schedules"][qid]["result"]["targets"][tkey]["dist_fp"] += 1
    body = {k: v for k, v in snap.items() if k != "integer_format_version"}
    engine.store.conn.execute(
        "UPDATE snapshots SET content_digest=?, payload_json=? WHERE snapshot_id=?",
        (content_digest(body), json.dumps(snap, sort_keys=True,
                                           separators=(",", ":")), snapshot_id),
    )
    engine.store.conn.commit()
    try:
        engine.restore_snapshot(snapshot_id)
    except ReplayError as exc:
        print(
            f"  {exc.code}: query={exc.detail['query_id']} "
            f"target={exc.detail['target']} field={exc.detail['field']}"
        )

    engine.store.close()
    print("\ndemo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
