from __future__ import annotations

from shard_diag.leases import LeaseManager
from shard_diag.scheduler import ProbeScheduler


def test_parallel_scheduler_respects_three_limits_and_records_waits(running_pool, spec_bundle):
    spec = spec_bundle["spec"]
    manager = LeaseManager(
        global_lease_limit=spec.global_lease_limit,
        databases=spec.databases,
    )
    scheduler = ProbeScheduler(spec, manager, running_pool.client, lease_ttl_ms=10_000)
    outcome = scheduler.run_batch(list(spec.events), batch_id="scheduler-test", sample_count=2)
    assert len(outcome.event_outcomes) == len(spec.events)
    assert outcome.high_water["max_active_global"] <= spec.global_lease_limit
    assert outcome.high_water["max_inflight"] <= sum(database.concurrency for database in spec.databases)
    for database in spec.databases:
        assert outcome.high_water["max_active_by_db"][database.id] <= database.lease_limit
        assert outcome.high_water["max_inflight_by_db"][database.id] <= database.concurrency
    assert outcome.high_water["max_inflight"] > 1
    assert all(item.thread_name.startswith("shard-probe") for item in outcome.event_outcomes)
    assert len({item.thread_name for item in outcome.event_outcomes}) > 1
    assert outcome.wait_records
    for wait in outcome.wait_records:
        assert wait["wait_started_ms"] <= wait["wait_ended_ms"]
        assert wait["wait_ms"] == wait["wait_ended_ms"] - wait["wait_started_ms"]
    assert outcome.quota_snapshot["active_leases"] == 0
    records = [item["route_record"] for item in outcome.as_dict()["events"]]
    assert [record["event_seq"] for record in records] == sorted(event.seq for event in spec.events)
    assert all(record["new_shard"] in event["hit_shards"] for record, event in zip(records, outcome.as_dict()["events"], strict=True))

