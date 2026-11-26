from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .clock import monotonic_ms
from .cluster import NodeCluster
from .ledger import Ledger
from .scheduler import Scheduler
from .snapshot import SessionSnapshot
from .spec import load_spec


def build_snapshot(spec: dict[str, Any], result: dict[str, Any]) -> SessionSnapshot:
    capacity_version = max(int(node["capacity_version"]) for node in spec["nodes"])
    watermarks = {
        "scale_factor": int(spec["scale_factor"]),
        "budget_units": int(spec["budget_units"]),
        "final_active_units": int(result.get("final_active_units", 0)),
        "shard_water_q": {
            decision["shard_id"]: decision["water_q"]
            for decision in result.get("decisions", [])
            if decision.get("kind") == "admit"
        },
    }
    return SessionSnapshot(
        capacity_version=capacity_version,
        spec_digest=spec["spec_digest"],
        watermarks=watermarks,
        traces=result.get("rounds", []),
        decisions=result.get("decisions", []),
        wait_records=result.get("wait_records", []),
        generated_monotonic_ms=monotonic_ms(),
    )


def run_spec(
    spec_path: str | Path,
    db_path: str | Path,
    *,
    log_dir: str | Path | None = None,
    shard_ids: Iterable[str] | None = None,
    snapshot_path: str | Path | None = None,
    recover: bool = True,
    run_id: str = "run",
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    ledger = Ledger(db_path)
    ledger.set_spec_digest(spec["spec_digest"])
    recovery = ledger.recover(spec) if recover else {
        "reclaimed_leases": 0,
        "reclaimed_units": 0,
        "unknown_shards": spec["topo_order"],
    }
    cluster = NodeCluster(spec, log_dir=log_dir or Path(db_path).with_suffix(".logs"))
    scheduler = Scheduler(spec, cluster, ledger, run_id=run_id)
    try:
        cluster.start()
        result = scheduler.run(shard_ids)
        snapshot = build_snapshot(spec, result)
        if snapshot_path:
            snapshot.save_atomic(snapshot_path)
        return {
            "ok": True,
            "spec": spec,
            "recovery": recovery,
            "result": result,
            "snapshot": snapshot.to_dict(),
            "db_path": str(db_path),
        }
    finally:
        cluster.stop()
        ledger.close()
