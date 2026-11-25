from __future__ import annotations

from shard_diag.canonical import canonical_bytes
from shard_diag.invalidation import InvalidationEngine


def _record(seq, db_id, table, key, key_column, version=1):
    return {
        "event_seq": seq,
        "db_id": db_id,
        "table": table,
        "key_column": key_column,
        "route_key": key,
        "old_route_version": version,
        "new_route_version": version,
        "old_shard": "before",
        "new_shard": "after",
        "changed": True,
        "drift_direction": "right",
        "boundary_index": 1,
        "boundary_old": 100,
        "boundary_new": 80,
        "boundary_offset": -20,
        "reason": "fixture",
    }


def _lease(seq, db_id, shard, lease_id, version=1):
    return {
        "lease_id": lease_id,
        "db_id": db_id,
        "shard_id": shard,
        "event_seq": seq,
        "route_version": version,
        "status": "released",
    }


def _fixture_records():
    records = [
        _record(1, "db-a", "orders", 210, "order_id"),
        _record(2, "db-a", "audit", 210, "audit_id"),
        _record(3, "db-a", "audit", 999, "audit_id"),
        _record(4, "db-b", "orders", 450, "order_id"),
        _record(5, "db-b", "audit", 450, "audit_id"),
        _record(6, "db-b", "orders", 90, "order_id"),
        _record(7, "db-b", "users", 777, "user_id"),
    ]
    leases = [
        _lease(seq, "db-b", f"shard:s{index}", f"lease-{seq}")
        for index, seq in enumerate((1, 2, 3, 4, 5, 6, 7))
    ]
    return records, leases


def test_invalidation_propagates_through_key_column_route_key_response_and_lease(spec_bundle):
    records, leases = _fixture_records()
    engine = InvalidationEngine(records, leases)
    report = engine.invalidate_table_version(spec_bundle["spec"].rule("orders", 1), spec_bundle["spec"].rule("orders", 2))
    assert report.table == "orders"
    assert report.old_version == 1
    assert report.new_version == 2
    assert report.boundary_index == 1
    assert report.boundary_offset == -20
    assert report.invalid_event_seqs == (1, 2, 3, 4, 5, 6)
    assert 7 not in report.invalid_event_seqs
    assert report.invalid_lease_ids == ("lease-1", "lease-2", "lease-3", "lease-4", "lease-5", "lease-6")
    assert report.invalid_response_keys
    node_types = {step["node_type"] for step in report.steps}
    assert {"table_rule", "route_result", "response", "lease"} <= node_types
    assert len(report.direct_record_keys) == 3
    assert len(report.invalid_record_keys) == 6
    assert all(step["order"] == index + 1 for index, step in enumerate(report.steps))


def test_invalidation_is_byte_stable_with_reversed_input_order(spec_bundle):
    records, leases = _fixture_records()
    first = InvalidationEngine(records, leases).invalidate_table_version(
        spec_bundle["spec"].rule("orders", 1), spec_bundle["spec"].rule("orders", 2)
    )
    second = InvalidationEngine(list(reversed(records)), list(reversed(leases))).invalidate_table_version(
        spec_bundle["spec"].rule("orders", 1), spec_bundle["spec"].rule("orders", 2)
    )
    assert first.digest == second.digest
    assert canonical_bytes(first.as_dict()) == canonical_bytes(second.as_dict())


def test_only_invalid_subset_is_recomputed_and_unaffected_bytes_remain(spec_bundle):
    records, leases = _fixture_records()
    engine = InvalidationEngine(records, leases)
    report = engine.invalidate_table_version(spec_bundle["spec"].rule("orders", 1), spec_bundle["spec"].rule("orders", 2))
    calls = []

    def recompute(record):
        calls.append((record["event_seq"], record["db_id"], record["table"]))
        result = dict(record)
        result["new_route_version"] = 2
        result["new_shard"] = "v2-shard"
        result["reason"] = "recomputed after boundary evolution"
        return result

    result = engine.recompute_invalid_subset(report, recompute)
    assert calls == [
        (record["event_seq"], record["db_id"], record["table"])
        for record in records
        if (record["event_seq"], record["db_id"], record["table"], record["route_key"]) in report.invalid_record_keys
    ]
    assert len(result.recomputed_record_keys) == 6
    assert result.unaffected_before_digest == result.unaffected_after_digest
    assert result.unaffected_byte_identical is True
    unaffected = [record for record in result.records if record["event_seq"] == 7]
    assert canonical_bytes(unaffected) == canonical_bytes([records[6]])
