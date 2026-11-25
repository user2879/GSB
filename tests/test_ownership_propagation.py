from lease_middleware.canonical import canonical_text
from lease_middleware.engine import ReplayEngine, replay_bundle, result_digest


def test_ownership_is_unique_and_reclamation_not_double_counted(bundle):
    result = replay_bundle(bundle)
    effective = [
        record
        for record in result["records"]
        if record.get("entry_kind") == "ownership"
        and record.get("counted_reclamation") is True
    ]
    keys = [
        (record["equity_id"], record["generation"])
        for record in effective
    ]
    assert len(keys) == len(set(keys))
    assert result["summary"]["effective_reclamation_count"] == 3
    assert result["summary"]["suppressed_alternative_count"] == 2

    active_owners = {(item["equity_id"], item["node_id"]) for item in result["ownership"]}
    assert ("equity-1", "cache-a") in active_owners
    assert ("equity-2", "cache-b") in active_owners
    assert len({equity for equity, _ in active_owners}) == len(active_owners)


def test_replay_is_byte_stable(bundle):
    first = replay_bundle(bundle)
    second = replay_bundle(bundle)
    assert canonical_text(first) == canonical_text(second)
    assert result_digest(first) == result_digest(second)


def test_propagation_subset_and_unchanged_capacity_bytes(bundle):
    engine = ReplayEngine(bundle)
    for event in bundle.events:
        if event["seq"] == 10:
            unaffected_ids = sorted(
                lease_id
                for lease_id in engine.leases
                if lease_id not in {"lease-b1", "lease-b2"}
            )
            unaffected_nodes = {"cache-a", "cache-c"}
            before_leases = canonical_text(
                {
                    lease_id: {
                        "generation": lease["generation"],
                        "state": lease["state"],
                        "ownership": engine.owners.get(lease["equity_id"]),
                    }
                    for lease_id in unaffected_ids
                    for lease in [engine.leases[lease_id]]
                }
            )
            before_capacity = canonical_text(
                [item for item in engine.capacity() if item["node_id"] in unaffected_nodes]
            )
            engine.apply_event(event)
            report = engine.propagation_reports[-1]
            assert report["affected_leases"] == ["lease-b1", "lease-b2"]
            assert "lease-a1" in report["unchanged_leases"]
            after_leases = canonical_text(
                {
                    lease_id: {
                        "generation": lease["generation"],
                        "state": lease["state"],
                        "ownership": engine.owners.get(lease["equity_id"]),
                    }
                    for lease_id in unaffected_ids
                    for lease in [engine.leases[lease_id]]
                }
            )
            after_capacity = canonical_text(
                [item for item in engine.capacity() if item["node_id"] in unaffected_nodes]
            )
            assert before_leases == after_leases
            assert before_capacity == after_capacity
            return
        engine.apply_event(event)
    raise AssertionError("seq 10 not found")
