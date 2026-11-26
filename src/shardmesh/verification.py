"""Incremental invalidation versus blank full-rebuild verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_dumps, sha256_hex
from .engine import Engine
from .errors import ShardMeshError
from .ledger import Ledger
from .replay import replay_ledger
from .spec import Spec


def core_projection(record: dict[str, Any]) -> dict[str, Any]:
    lease = record.get("lease", {})
    attribution = dict(record.get("error_attribution", {}))
    attribution.pop("active_count", None)
    adjudication = record.get("adjudication") or {}
    return {
        "shard_id": record["shard_id"],
        "node_id": record.get("node_id"),
        "capacity_version": record.get("capacity_version"),
        "topology_index": record.get("topology_index"),
        "parameter_revision": record.get("parameter_revision"),
        "outcome_summary": record.get("outcome", {}).get("summary"),
        "lease_identity": {
            key: lease.get(key)
            for key in ("lease_id", "version", "renew_count", "capacity_version")
        },
        "resource": {
            key: record.get("resource", {}).get(key)
            for key in ("units", "global_budget", "fixed_units", "budget_fixed")
        },
        "error_attribution": attribution,
        "adjudication": {
            "winner": adjudication.get("winner"),
            "error_affected": adjudication.get("error_affected"),
            "rule": adjudication.get("rule"),
            "tie_break": adjudication.get("tie_break"),
        },
    }


def _diff_values(left: Any, right: Any, path: str) -> str | None:
    if type(left) != type(right):
        return path
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return f"{path}.{key}"
            found = _diff_values(left[key], right[key], f"{path}.{key}")
            if found:
                return found
    elif isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.length"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            found = _diff_values(left_item, right_item, f"{path}[{index}]")
            if found:
                return found
    elif left != right:
        return path
    return None


def assert_projections_match(
    incremental: dict[str, dict[str, Any]],
    full: dict[str, dict[str, Any]],
) -> None:
    for shard_id in sorted(set(incremental) | set(full)):
        if shard_id not in incremental:
            raise ShardMeshError("E_DIVERGE", f"incremental missing shard {shard_id}", category="shard", field_name="shard_id")
        if shard_id not in full:
            raise ShardMeshError("E_DIVERGE", f"full rebuild missing shard {shard_id}", category="shard", field_name="shard_id")
        left = core_projection(incremental[shard_id])
        right = core_projection(full[shard_id])
        field_path = _diff_values(left, right, "record")
        if field_path:
            field_name = field_path.split(".", 1)[-1]
            raise ShardMeshError(
                "E_DIVERGE",
                f"incremental/full mismatch at {shard_id}: {field_path}",
                category="shard",
                field_name=field_name,
                details={
                    "shard_id": shard_id,
                    "node_id": right.get("node_id"),
                    "field_path": field_path,
                    "incremental": left,
                    "full_rebuild": right,
                },
            )


def _raw_records(ledger: Ledger) -> dict[str, str]:
    return {row["shard_id"]: row["record_json"] for row in ledger.list_rows()}


def run_incremental(spec: Spec, samples: dict[str, Any], root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "incremental.sqlite3"
    if ledger_path.exists():
        ledger_path.unlink()
    for suffix in ("-wal", "-shm"):
        side = Path(str(ledger_path) + suffix)
        if side.exists():
            side.unlink()
    ledger = Ledger(ledger_path)
    ledger.initialize_spec(spec.to_dict())
    engine = Engine(spec, samples, ledger, work_dir=root / "runtime")
    byte_check: dict[str, Any] = {"unchanged": [], "changed": [], "violations": []}
    try:
        with engine:
            before: dict[str, str] | None = None
            arrival_chunk = []
            for event in spec.events:
                if event.type == "arrive":
                    arrival_chunk.append(event)
                    continue
                if arrival_chunk:
                    engine.run_events(arrival_chunk)
                    arrival_chunk = []
                engine.settle()
                if before is None:
                    before = _raw_records(ledger)
                engine.run_events([event])
            if arrival_chunk:
                engine.run_events(arrival_chunk)
            if before is None:
                before = _raw_records(ledger)
            after = _raw_records(ledger)
            affected: set[str] = set()
            for plan in engine.invalidation_plans:
                affected.update(plan["changed"])
            for shard_id, before_json in (before or {}).items():
                if shard_id in affected:
                    if after.get(shard_id) != before_json:
                        byte_check["changed"].append(shard_id)
                else:
                    if after.get(shard_id) == before_json:
                        byte_check["unchanged"].append(shard_id)
                    else:
                        byte_check["violations"].append(shard_id)
            if byte_check["violations"]:
                raise ShardMeshError(
                    "E_DIVERGE",
                    "unaffected shard record bytes changed",
                    category="shard",
                    field_name="record_json",
                    details={"shards": byte_check["violations"]},
                )
            records = dict(engine.records)
            plans = list(engine.invalidation_plans)
    finally:
        replay = replay_ledger(ledger, samples)
        ledger.close()
    return {"records": records, "byte_check": byte_check, "plans": plans, "replay": replay, "ledger": ledger_path}


def run_full_rebuild(spec: Spec, samples: dict[str, Any], root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "full-rebuild.sqlite3"
    if ledger_path.exists():
        ledger_path.unlink()
    for suffix in ("-wal", "-shm"):
        side = Path(str(ledger_path) + suffix)
        if side.exists():
            side.unlink()
    ledger = Ledger(ledger_path)
    ledger.initialize_spec(spec.to_dict())
    engine = Engine(spec, samples, ledger, work_dir=root / "full-runtime")
    try:
        with engine:
            engine.run_events(spec.events)
            records = dict(engine.records)
    finally:
        replay = replay_ledger(ledger, samples)
        ledger.close()
    return {"records": records, "replay": replay, "ledger": ledger_path}


def verify_spec(spec: Spec, samples: dict[str, Any], root: str | Path) -> dict[str, Any]:
    root = Path(root)
    incremental = run_incremental(spec, samples, root / "incremental")
    full = run_full_rebuild(spec, samples, root / "full")
    assert_projections_match(incremental["records"], full["records"])
    report = {
        "consistent": True,
        "code": "OK",
        "incremental_ledger": str(incremental["ledger"]),
        "full_ledger": str(full["ledger"]),
        "byte_check": incremental["byte_check"],
        "invalidation_plans": incremental["plans"],
        "replay": {"incremental": incremental["replay"], "full": full["replay"]},
        "compared_core_digests": {
            shard_id: sha256_hex(core_projection(record))
            for shard_id, record in sorted(incremental["records"].items())
        },
    }
    (root / "verification-report.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "verification-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report
