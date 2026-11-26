from __future__ import annotations

from workbench.engine import WorkbenchEngine
from workbench.timeutil import ScriptedClock
from workbench.testkit import write_test_spec


def test_parallel_and_serial_have_same_summary_and_attribution(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec = write_test_spec(spec_path)

    parallel = WorkbenchEngine(
        spec, tmp_path / "parallel", clock=ScriptedClock(7), parallel=True, max_workers=4
    )
    parallel.run_until_complete()
    parallel_rows = {r["shard_id"]: r for r in parallel.ledger.list_shards()}
    parallel.close()

    serial = WorkbenchEngine(
        spec, tmp_path / "serial", clock=ScriptedClock(7), parallel=False
    )
    serial.run_until_complete()
    serial_rows = {r["shard_id"]: r for r in serial.ledger.list_shards()}
    serial.close()

    for shard_id in spec.topo_order:
        assert parallel_rows[shard_id]["summary_digest"] == serial_rows[shard_id]["summary_digest"]
        assert parallel_rows[shard_id]["node_id"] == serial_rows[shard_id]["node_id"]
        assert parallel_rows[shard_id]["capacity_version"] == serial_rows[shard_id]["capacity_version"]


def test_lease_release_and_node_restart_restore_optimistic_state(tmp_path):
    spec = write_test_spec(tmp_path / "spec.json")
    engine = WorkbenchEngine(spec, tmp_path / "work", clock=ScriptedClock())
    try:
        engine.start_nodes()
        assert engine.admit_shard("s1") is not None
        released = engine.release_lease("lease-s1")
        assert released is True
        assert engine.ledger.get_lease("lease-s1")["state"] == "released"

        assert engine.admit_shard("s2") is not None
        returncode = engine.orchestrator.kill_node("n2")
        assert returncode == -9
        engine.orchestrator.restart_node("n2")
        restored = engine.restore_active_leases_to_nodes()
        assert restored == 1
        renewed = engine.renew_lease("lease-s2", 1)
        assert renewed["lease_version"] == 2
        assert renewed["renew_count"] == 1
    finally:
        engine.close()


def test_lease_limit_is_separate_from_concurrency_quota(tmp_path):
    raw = write_test_spec(tmp_path / "base.json").raw
    n1 = next(node for node in raw["nodes"] if node["node_id"] == "n1")
    n1["concurrency_quota"] = 5
    n1["lease_limit"] = 1
    spec = write_test_spec(tmp_path / "lease-limit.json", raw)
    engine = WorkbenchEngine(spec, tmp_path / "work", clock=ScriptedClock())
    try:
        engine.start_nodes()
        assert engine.admit_shard("s1") is not None
        assert engine.admission_block_reason("s3") == "lease_limit"
    finally:
        engine.close()
