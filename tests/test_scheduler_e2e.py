import json
import os

import pytest

from edge_cache.clock import DeterministicClock
from edge_cache.config import load_config_dir
from edge_cache.errors import E_LEASE_EXPIRED
from edge_cache.ledger import Ledger
from edge_cache.node_pool import NodePool
from edge_cache.receipts import C1, C2, C3
from edge_cache.runner import RunOrchestrator, full_rebuild
from edge_cache.scheduler import DeliveryEngine


def _clock(work_dir, config):
    steps = max(256, len(config.shards) * len(config.nodes) * 16)
    schedule = list(range(1, steps + 1))
    path = os.path.join(work_dir, "clock-schedule.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"schedule": schedule}, handle)
    return DeterministicClock(schedule), path


def test_engine_batch_concludes_every_shard_with_real_nodes(sample_config_dir, tmp_workdir):
    config = load_config_dir(sample_config_dir)
    clock, schedule_path = _clock(tmp_workdir, config)
    ledger = Ledger(os.path.join(tmp_workdir, "ledger"))
    pool = NodePool(config, os.path.join(tmp_workdir, "specs"), schedule_path)
    pool.start_all()
    try:
        engine = DeliveryEngine(config, ledger, pool, clock=clock)
        result = engine.run_batch(C3)
    finally:
        pool.close()

    assert all(
        conclusion.available for conclusion in engine.conclusions.values()
    )
    assert result["format_counts"] == {C1: 5, C2: 5, C3: 5}
    business = ledger.business_rows()
    assert [row["shard_id"] for row in business] == [
        shard.shard_id for shard in sorted(config.shards, key=lambda item: item.index)
    ]
    # Every lease produced an acquire + release algebra pair.
    algebra = Ledger._read_jsonl(os.path.join(ledger.root, "algebra.jsonl"))
    assert len([e for e in algebra if e["action"] == "acquire"]) == len(config.shards)
    assert len([e for e in algebra if e["action"] == "release"]) == len(config.shards)
    tags = {(event["lease_id"]): event["new_tag"] for event in algebra if event["action"] == "release"}
    assert all(tag == 1 for tag in tags.values())


def test_expired_lease_reclamation_uses_real_node_release(sample_config_dir, tmp_workdir):
    config = load_config_dir(sample_config_dir)
    clock, schedule_path = _clock(tmp_workdir, config)
    ledger = Ledger(os.path.join(tmp_workdir, "ledger"))
    pool = NodePool(config, os.path.join(tmp_workdir, "specs"), schedule_path)
    pool.start_all()
    try:
        engine = DeliveryEngine(config, ledger, pool, clock=clock, lease_ttl_ms=1)
        task = engine.config.shards[0]
        from edge_cache.scheduler import ShardTask

        shard_task = ShardTask(task.shard_id, task.index, task.size_units, task.size_units)
        node_id = sorted(node.node_id for node in engine.config.nodes)[0]
        lease_id = engine._lease_id(node_id, shard_task.shard_id)
        client = pool.client(node_id)
        acquired = client.request({
            "type": "acquire", "lease_id": lease_id,
            "shard_id": shard_task.shard_id, "at_ms": clock.now_ms(),
        })
        assert acquired["ok"]
        from edge_cache.leases import Lease

        lease = Lease(
            lease_id=lease_id, node_id=node_id, shard_id=shard_task.shard_id,
            tag=0, issued_ms=1, expires_ms=2, slots=shard_task.slots,
        )
        engine.lease_manager.register(lease)
        reclaimed = engine.lease_manager.reclaim_expired(now_ms=10)
        assert len(reclaimed) == 1
        assert reclaimed[0][1].code == E_LEASE_EXPIRED
        assert ledger.algebra_rows()[-1]["action"] == "expiry-release"
        # The node really advanced its generation and freed the slot.
        assert engine.lease_manager.active_lease_count() == 0
        snapshot = client.request({"type": "snapshot"})
        assert all(item["lease_id"] != lease_id for item in snapshot["leases"])
    finally:
        pool.close()


def _business_signature(work_dir: str) -> list[dict]:
    ledger = Ledger(os.path.join(work_dir, "ledger"))
    return ledger.business_rows()


def test_incremental_run_matches_clean_slate_full_rebuild(sample_config_dir, tmp_workdir):
    incremental_dir = os.path.join(tmp_workdir, "incremental")
    rebuild_dir = os.path.join(tmp_workdir, "rebuild")

    incremental = RunOrchestrator(sample_config_dir, incremental_dir, C3)
    inc_report = incremental.run()

    full_report = full_rebuild(sample_config_dir, rebuild_dir, C3)

    inc_rows = _business_signature(incremental_dir)
    rebuild_rows = _business_signature(rebuild_dir)
    assert inc_rows == rebuild_rows

    inc_failures = Ledger._read_jsonl(
        os.path.join(incremental_dir, "ledger", "failures.jsonl")
    )
    rebuild_failures = Ledger._read_jsonl(
        os.path.join(rebuild_dir, "ledger", "failures.jsonl")
    )
    assert inc_failures == rebuild_failures
    assert inc_report["format_counts"] == full_report["format_counts"]

    # Reuse must have happened on the incremental workspace (artifacts persist
    # across batches), while a clean rebuild starts empty.  Reused summaries
    # skip real node leases by design, so the incremental run performs no more
    # generation advances than the clean rebuild.
    inc_reuse = Ledger._read_jsonl(
        os.path.join(incremental_dir, "ledger", "reuse.jsonl")
    )
    assert any(row["kind"] == "reuse-hit" for row in inc_reuse)
    assert any(row["kind"] == "invalidation" for row in inc_reuse)
    inc_advances = sum(inc_report["node_generation_advances"].values())
    rebuild_advances = sum(full_report["node_generation_advances"].values())
    assert inc_advances <= rebuild_advances


def test_format_roundtrip_batches_cover_all_three_formats(sample_config_dir, tmp_workdir):
    work_dir = os.path.join(tmp_workdir, "run")
    report = RunOrchestrator(sample_config_dir, work_dir, C3).run()
    assert report["format_counts"][C1] > 0
    assert report["format_counts"][C2] > 0
    assert report["format_counts"][C3] > 0
    snapshots = os.listdir(os.path.join(work_dir, "snapshots", "committed"))
    # One initial batch plus one per command in the sample command stream.
    assert len(snapshots) == 1 + len(load_config_dir(sample_config_dir).commands)
