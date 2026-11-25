from __future__ import annotations

import copy

from lease_arbiter.canonical import canonical_bytes
from lease_arbiter.engine import LeaseArbiterEngine, choose_path, run_arbitration
from lease_arbiter.errors import DivergenceError, LeaseError
from lease_arbiter.state import LeaseState, transition
from lease_arbiter.verifier import compare_arbitration_results, verify_spec


def _record() -> LeaseState:
    return LeaseState(
        lease_id="lease-1",
        node_id="n1",
        lease_type="cache",
        ttl_ms=1000,
        renew_interval_ms=100,
        quota_version=1,
    )


def test_cross_generation_jump_is_rejected_without_mutation():
    record = _record()
    transition(record, "granted", "granted", 0, 1, 0)
    before = (record.state, record.label)
    try:
        transition(record, "renewed", "jump", 1, 2, 10, new_label=3)
    except LeaseError as exc:
        assert exc.code == "E_LEASE"
    else:
        raise AssertionError("cross-generation jump was accepted")
    assert (record.state, record.label) == before


def test_repeated_terminal_transition_is_rejected():
    record = _record()
    transition(record, "granted", "granted", 0, 1, 0)
    transition(record, "released", "released", 1, 2, 10)
    try:
        transition(record, "released", "released again", 2, 3, 20)
    except LeaseError as exc:
        assert exc.code == "E_LEASE"
    else:
        raise AssertionError("repeated terminal transition was accepted")
    assert record.state == "released"
    assert record.label == 2


def test_stale_renewal_preserves_generation(sample_spec):
    engine = LeaseArbiterEngine(sample_spec)
    for event in sample_spec.events[:5]:
        engine.expire_due(event.at_ms, event.seq)
        engine.apply_event(event)
    stale = sample_spec.events[7]
    result = engine.apply_event(stale)
    assert result.outcome["status"] == "E_STALE"
    assert result.outcome["label"] == 2
    assert result.transitions == []
    assert engine.leases["a-cache-1"].label == 2


def test_two_paths_are_byte_identical(sample_spec):
    sequential = run_arbitration(sample_spec, "sequential")
    batch = run_arbitration(sample_spec, "batch")
    assert canonical_bytes(sequential["comparison_projection"]) == canonical_bytes(batch["comparison_projection"])
    assert sequential["comparison_sha256"] == batch["comparison_sha256"]
    verified = verify_spec(sample_spec)
    assert verified["comparison"]["byte_identical"] is True


def test_expiry_and_quota_invalidation_have_unique_ownership(sample_spec):
    result = run_arbitration(sample_spec, "sequential")
    reclaims = result["reclaims"]
    assert [item["lease_id"] for item in reclaims] == ["b-session-1", "a-cache-1"]
    assert [item["reason"] for item in reclaims] == ["expired", "quota_invalidated"]
    keys = [(item["global_slot"], item["node_slot"]) for item in reclaims]
    assert len(keys) == len(set(keys))
    grants = [item for item in result["event_outcomes"] if item["status"] == "granted"]
    assert {item["global_slot"] for item in grants} <= {1, 2, 3}
    assert [item["global_slot"] for item in grants[:3]] == [1, 2, 3]
    assert grants[3]["global_slot"] == 2
    final = {item["lease_id"]: item for item in result["final_state"]["leases"]}
    assert final["a-cache-1"]["state"] == "expired"
    assert final["b-session-1"]["state"] == "expired"
    assert all(item["owner"] is None for item in final.values())


def test_quota_denial_does_not_consume_generation_or_slot(sample_spec):
    result = run_arbitration(sample_spec, "sequential")
    denied = next(item for item in result["event_outcomes"] if item["seq"] == 4)
    assert denied["status"] == "E_QUOTA"
    assert denied["label"] == 0
    assert denied["global_slot"] is None
    assert denied["node_slot"] is None


def test_auto_path_reports_boundaries_when_not_triggered():
    selection = choose_path(
        requested_path="auto",
        node_count=2,
        event_count=9,
        remaining_capacity=3,
    )
    assert selection["selected"] == "sequential"
    assert selection["conditions"]["event_count"]["met"] is False
    assert selection["boundary"]["event_count"] == 10
    assert selection["not_triggered_boundary"]["event_count"]["needed_delta"] == 1
    assert selection["reason"] == "event_count is below its batch threshold"


def test_auto_path_triggers_batch(sample_spec):
    selection = choose_path(
        requested_path="auto",
        node_count=len(sample_spec.nodes),
        event_count=len(sample_spec.events),
        remaining_capacity=sample_spec.global_concurrency,
    )
    assert selection["selected"] == "batch"
    assert selection["auto_triggered"] is True
    assert selection["params"] == {"event_count": 12, "node_count": 2, "remaining_capacity": 3}



def test_divergence_locator_reports_first_event_and_field(sample_spec):
    sequential = run_arbitration(sample_spec, "sequential")
    batch = copy.deepcopy(run_arbitration(sample_spec, "batch"))
    batch["comparison_projection"]["events"][2]["expires_at_ms"] = 999_999
    try:
        compare_arbitration_results(sequential, batch)
    except DivergenceError as exc:
        assert exc.code == "E_DIVERGE"
        assert exc.details["event_seq"] == 3
        assert exc.details["field"] == "expires_at_ms"
    else:
        raise AssertionError("divergence was not detected")
