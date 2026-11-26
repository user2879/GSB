from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes
from .errors import DivergeError
from .graph import derive_invalidation, propagation_basis
from .ledger import Ledger
from .runner import run_spec


def _read_outcomes(db_path: str | Path) -> dict[str, dict[str, Any]]:
    ledger = Ledger(db_path)
    try:
        return {row["shard_id"]: row for row in ledger.list_outcomes()}
    finally:
        ledger.close()


def _record_bytes(db_path: str | Path, shard_id: str) -> bytes:
    ledger = Ledger(db_path)
    try:
        return ledger.outcome_record_bytes(shard_id)
    finally:
        ledger.close()


def run_evolution_check(
    *,
    old_spec_path: str | Path,
    new_spec_path: str | Path,
    work_dir: str | Path,
) -> dict[str, Any]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    old_db = work / "old-full.db"
    incremental_db = work / "incremental.db"
    full_db = work / "blank-full.db"
    for path in (old_db, incremental_db, full_db):
        if path.exists():
            path.unlink()

    old_run = run_spec(old_spec_path, old_db, log_dir=work / "old-logs", run_id="old")
    shutil.copy2(old_db, incremental_db)

    from .spec import load_spec

    old_spec = load_spec(old_spec_path)
    new_spec = load_spec(new_spec_path)
    invalid, steps = derive_invalidation(old_spec, new_spec)
    invalid_ordered = [sid for sid in new_spec["topo_order"] if sid in invalid]
    unaffected = [sid for sid in new_spec["topo_order"] if sid not in invalid]

    before_bytes = {sid: _record_bytes(incremental_db, sid) for sid in unaffected}
    inc_run = run_spec(
        new_spec_path,
        incremental_db,
        log_dir=work / "incremental-logs",
        shard_ids=invalid_ordered,
        run_id="incremental",
    )
    after_bytes = {sid: _record_bytes(incremental_db, sid) for sid in unaffected}
    unchanged_checks = []
    for shard_id in unaffected:
        same = before_bytes[shard_id] == after_bytes[shard_id]
        unchanged_checks.append({"shard_id": shard_id, "byte_identical": same})
        if not same:
            raise DivergeError(
                f"unaffected shard {shard_id} ledger record changed",
                field=f"outcomes.{shard_id}",
                position=shard_id,
            )

    full_run = run_spec(new_spec_path, full_db, log_dir=work / "full-logs", run_id="full")
    comparisons = compare_outputs(new_spec, incremental_db, full_db, steps)

    return {
        "ok": True,
        "invalid_set": invalid_ordered,
        "propagation_steps": steps,
        "unaffected_byte_checks": unchanged_checks,
        "comparisons": comparisons,
        "incremental_db": str(incremental_db),
        "full_db": str(full_db),
        "old_recovery": old_run["recovery"],
        "incremental_recovery": inc_run["recovery"],
        "full_recovery": full_run["recovery"],
    }


def compare_outputs(
    new_spec: dict[str, Any],
    incremental_db: str | Path,
    full_db: str | Path,
    steps: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    incremental_outcomes = _read_outcomes(incremental_db)
    full_outcomes = _read_outcomes(full_db)
    step_by_shard = {step["shard_id"]: step for step in (steps or [])}
    comparisons: list[dict[str, Any]] = []
    for shard_id in new_spec["topo_order"]:
        inc = incremental_outcomes.get(shard_id)
        full = full_outcomes.get(shard_id)
        if inc is None or full is None:
            raise DivergeError(
                f"missing outcome for {shard_id} during incremental/full comparison",
                field=f"outcomes.{shard_id}",
                position=shard_id,
                incremental_present=inc is not None,
                full_present=full is not None,
            )
        expected_basis = propagation_basis(new_spec, shard_id)
        fields = {
            "summary_digest": (inc["summary_digest"], full["summary_digest"]),
            "summary_json": (canonical_bytes(inc["summary"]), canonical_bytes(full["summary"])),
            "water_q": (int(inc["water_q"]), int(full["water_q"])),
            "node_id": (inc["node_id"], full["node_id"]),
            "capacity_version": (int(inc["capacity_version"]), int(full["capacity_version"])),
            "lease_attribution": (
                canonical_bytes({"node_id": inc["node_id"], "capacity_version": inc["capacity_version"]}),
                canonical_bytes({"node_id": full["node_id"], "capacity_version": full["capacity_version"]}),
            ),
            "propagation_basis_json": (
                inc.get("propagation_basis_json"),
                full.get("propagation_basis_json"),
            ),
        }
        for field_name, (actual, expected) in fields.items():
            if actual != expected:
                raise DivergeError(
                    f"incremental/full first divergence at {shard_id}.{field_name}",
                    field=f"compare.{shard_id}.{field_name}",
                    position=shard_id,
                    actual=_safe_decode(actual),
                    expected=_safe_decode(expected),
                    first_difference={"shard_id": shard_id, "field": field_name},
                )

        comparisons.append(
            {
                "shard_id": shard_id,
                "matched": True,
                "summary_digest": inc["summary_digest"],
                "water_q": int(inc["water_q"]),
                "node_id": inc["node_id"],
                "propagation_basis_digest": expected_basis["basis_digest"],
                "invalidation": step_by_shard.get(shard_id),
            }
        )
    return comparisons


def _safe_decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
