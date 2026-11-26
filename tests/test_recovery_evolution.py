from __future__ import annotations

import copy
import json

import pytest

from workbench.canonical import canonical_json
from workbench.engine import WorkbenchEngine
from workbench.invalidation import plan_evolution
from workbench.recovery import (
    compare_conclusion_views,
    unchanged_byte_records,
    verify_ledger_replay,
)
from workbench.snapshot import SnapshotStore
from workbench.spec import load_spec
from workbench.testkit import write_test_spec
from workbench.timeutil import ScriptedClock


def _serial_engine(spec, workdir):
    return WorkbenchEngine(
        spec,
        workdir,
        clock=ScriptedClock(start_ms=4242),
        parallel=False,
    )


def _assert_full_shard_bytes(before, after, shard_ids):
    before_shards = {r["shard_id"]: r for r in before["shards"]}
    after_shards = {r["shard_id"]: r for r in after["shards"]}
    before_leases = {r["lease_id"]: r for r in before["leases"]}
    after_leases = {r["lease_id"]: r for r in after["leases"]}
    before_water = {r["record_key"]: r for r in before["water_records"]}
    after_water = {r["record_key"]: r for r in after["water_records"]}
    before_errors = {r["shard_id"]: r for r in before["error_attribution"]}
    after_errors = {r["shard_id"]: r for r in after["error_attribution"]}
    for shard_id in shard_ids:
        assert canonical_json(before_shards[shard_id]) == canonical_json(after_shards[shard_id])
        lease_id = before_shards[shard_id]["lease_id"]
        assert canonical_json(before_leases[lease_id]) == canonical_json(after_leases[lease_id])
        water_key = f"shard:{shard_id}"
        assert canonical_json(before_water[water_key]) == canonical_json(after_water[water_key])
        assert canonical_json(before_errors[shard_id]) == canonical_json(after_errors[shard_id])


def test_forced_kill_recovery_matches_blank_rebuild(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec = write_test_spec(spec_path)

    baseline = _serial_engine(spec, tmp_path / "baseline")
    baseline.run_until_complete()
    baseline_view = baseline.ledger.export_view()
    verify_ledger_replay(baseline.ledger, spec.topo_order)
    baseline.close()

    engine = _serial_engine(spec, tmp_path / "recovery-db")
    try:
        engine.start_nodes()
        engine.run_shard_now("s1")
        context = engine.admit_shard("s3")
        assert context is not None
        before_kill = engine.ledger.export_view()
        returncode = engine.orchestrator.kill_node("n1")
        assert returncode == -9

        unknown = engine.recover_after_termination(restart_nodes=True)
        assert [row["shard_id"] for row in unknown] == ["s3"]
        after_recovery = engine.ledger.export_view()
        unchanged_byte_records(before_kill, after_recovery, ["s1"])
        _assert_full_shard_bytes(before_kill, after_recovery, ["s1"])
        s3 = engine.ledger.get_shard("s3")
        assert s3["status"] == "unknown"
        assert s3["error_code"] == "E_REPLAY_RERUN_REQUIRED"

        result = engine.run_until_complete(recover=False)
        assert len(result["completed"]) == 6
        recovered_view = engine.ledger.export_view()
        comparison = compare_conclusion_views(baseline_view, recovered_view, spec.topo_order)
        assert comparison["identical"] is True
        verify_ledger_replay(engine.ledger, spec.topo_order)
    finally:
        engine.close()


def test_capacity_evolution_recomputes_only_invalid_subset(tmp_path):
    old_path = tmp_path / "old-spec.json"
    old_spec = write_test_spec(old_path)
    old_engine = _serial_engine(old_spec, tmp_path / "evolving")
    old_engine.run_until_complete()
    attribution = {
        row["shard_id"]: row["node_id"] for row in old_engine.ledger.list_shards()
    }
    old_engine.close()

    new_raw = copy.deepcopy(old_spec.raw)
    for node in new_raw["nodes"]:
        if node["node_id"] == "n1":
            node["capacity_version"] = 3
    new_path = tmp_path / "new-spec.json"
    new_path.write_text(json.dumps(new_raw), encoding="utf-8")
    new_spec = load_spec(new_path)
    plan = plan_evolution(old_spec, new_spec, old_attribution=attribution)
    assert plan["invalid_shards"] == ["s1", "s3", "s5", "s6"]
    assert plan["unchanged_shards"] == ["s2", "s4"]
    assert any(step["stage"] == "propagated" and step["shard_id"] == "s5" for step in plan["propagation_steps"])

    evolved = _serial_engine(new_spec, tmp_path / "evolving")
    now = evolved.now_ms()
    for shard_id in plan["invalid_shards"]:
        evolved.ledger.mark_invalid_for_recompute(shard_id, "E_STALE_CAPACITY_VERSION", now)
        evolved.ledger.record_invalidation(new_spec.digest, shard_id, "capacity_or_dependency", plan, now)
    evolved.ledger.set_meta("spec_digest", new_spec.digest)
    evolved.ledger.set_meta("spec_json", canonical_json_for_test(new_spec.raw))
    before = evolved.ledger.export_view()
    try:
        evolved.run_until_complete(recover=False)
        after = evolved.ledger.export_view()
        _assert_full_shard_bytes(before, after, ["s2", "s4"])
        unchanged_byte_records(before, after, ["s2", "s4"])
    finally:
        evolved.close()

    blank = _serial_engine(new_spec, tmp_path / "blank-new")
    blank.run_until_complete()
    blank_view = blank.ledger.export_view()
    blank.close()

    evolved_db = WorkbenchEngine(new_spec, tmp_path / "evolving")
    try:
        evolved_view = evolved_db.ledger.export_view()
        result = compare_conclusion_views(blank_view, evolved_view, new_spec.topo_order)
        assert result["identical"] is True
    finally:
        evolved_db.close()


def canonical_json_for_test(value):
    from workbench.canonical import canonical_json

    return canonical_json(value)


def test_snapshot_spec_mismatch_replaces_whole_file(tmp_path):
    spec = write_test_spec(tmp_path / "spec.json")
    engine = _serial_engine(spec, tmp_path / "snap")
    engine.run_until_complete()
    store = SnapshotStore(tmp_path / "snapshot.json")
    stale = {
        "snapshot_format_version": 999,
        "spec_digest": "stale",
        "view": {"old": True},
        "content_digest": "stale",
    }
    store.write_atomic(stale)
    snapshot, replaced, reason = store.open_or_replace(spec, engine.ledger)
    assert replaced is True
    assert reason == "format_version_mismatch"
    assert snapshot["spec_digest"] == spec.digest

    stale_spec_raw = copy.deepcopy(spec.raw)
    stale_spec_raw["global_budget_units"] += 1
    stale_spec = write_test_spec(tmp_path / "changed.json", stale_spec_raw)
    snapshot, replaced, reason = store.open_or_replace(stale_spec, engine.ledger)
    assert replaced is True
    assert reason == "spec_digest_mismatch"
    assert "shards" in snapshot["view"]
    engine.close()


def test_compare_locates_first_replay_mismatch(tmp_path):
    spec = write_test_spec(tmp_path / "spec.json")
    left = _serial_engine(spec, tmp_path / "left")
    left.run_until_complete()
    view = left.ledger.export_view()
    left.close()
    mutated = copy.deepcopy(view)
    mutated["shards"][0]["capacity_version"] = 999
    with pytest.raises(Exception) as exc:
        compare_conclusion_views(view, mutated, spec.topo_order)
    assert exc.value.code == "E_REPLAY"
    assert exc.value.details["shard_id"] == "s1"
    assert exc.value.details["field"] == "shards.capacity_version"
