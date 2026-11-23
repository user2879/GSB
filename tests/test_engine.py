from __future__ import annotations

import json

import pytest

from leaseaudit.clock import ManualClock
from leaseaudit.engine import LeaseEngine
from leaseaudit.errors import BusinessError, ErrorCode
from leaseaudit.ledger import AuditLedger
from leaseaudit.replay import replay_ledger
from leaseaudit.scenario import validate_scenario


def test_issue_renew_expiry_and_reissue_other_source(tmp_path, scenario):
    ledger = AuditLedger(tmp_path / "audit.sqlite", scenario=scenario, verify=True)
    clock = ManualClock(value_ms=2_000_000)
    engine = LeaseEngine(scenario, ledger, clock=clock)
    try:
        alpha = engine.issue("alpha", 2)
        alpha_extra = engine.issue("alpha", 2)
        assert engine.sources["alpha"].active_granted == 4
        with pytest.raises(BusinessError) as caught:
            engine.issue("alpha", 1)
        assert caught.value.code == ErrorCode.E_CAPACITY

        clock.advance(100)
        renewed = engine.renew(alpha.lease_id, "alpha")
        assert renewed.expires_at_ms == 2_000_100 + scenario.lease_duration_ms
        assert renewed.renew_count == 1

        clock.advance(1001)
        expired = engine.process_expirations()
        assert {event.lease_id for event in expired} == {alpha.lease_id, alpha_extra.lease_id}
        with pytest.raises(BusinessError) as caught:
            engine.renew(alpha.lease_id, "alpha")
        assert caught.value.code == ErrorCode.E_RACE
        assert engine.sources["alpha"].active_granted == 0

        gamma = engine.issue("gamma", 2)
        assert gamma.amount == 2
        assert engine.total_active <= scenario.cluster_capacity
        assert engine.retained_total + engine.global_allocatable == scenario.cluster_capacity
    finally:
        engine.close()
    assert replay_ledger(tmp_path / "audit.sqlite", scenario)["status"] == "OK"


def test_quota_reclaim_only_idle_above_reserve_and_is_idempotent(tmp_path, scenario_doc):
    scenario_doc["cluster_capacity"] = 6
    scenario_doc["reclaim_threshold"] = 900000
    scenario_doc["reclaim_window_rounds"] = 1
    scenario_doc["sources"] = [
        {
            "source_id": "alpha",
            "address": "127.0.0.1",
            "port": 55601,
            "lease_share": 4,
            "reserve_quota": 1,
        }
    ]
    scenario = validate_scenario(scenario_doc)
    ledger = AuditLedger(tmp_path / "audit.sqlite", scenario=scenario, verify=True)
    clock = ManualClock(value_ms=3_000_000)
    engine = LeaseEngine(scenario, ledger, clock=clock)
    try:
        lease = engine.issue("alpha", 3)
        assert engine.sources["alpha"].retained_quota == 3
        assert engine.global_allocatable == 3
        engine.observe_all(force=True)
        clock.advance(100)
        engine.observe_all(force=True)
        assert not [
            event for event in ledger.events() if event.event_type == "QUOTA_RECLAIMED"
        ]

        clock.advance(scenario.lease_duration_ms)
        engine.process_expirations()
        for _ in range(3):
            engine.observe_all(force=True)
            clock.advance(100)
        reclaimed = [event for event in ledger.events() if event.event_type == "QUOTA_RECLAIMED"]
        assert len(reclaimed) == 1
        assert reclaimed[0].amount == 2
        assert engine.sources["alpha"].retained_quota == 1
        assert engine.global_allocatable == 5
        before = engine.snapshot()
        with pytest.raises(BusinessError) as caught:
            engine.request_quota_reclaim("alpha")
        assert caught.value.code == ErrorCode.E_RACE
        after = engine.snapshot()
        assert before == after
        assert any(event.event_type == "RACE_REJECTED" for event in ledger.events())
    finally:
        engine.close()
    assert replay_ledger(tmp_path / "audit.sqlite", scenario)["status"] == "OK"


def test_duplicate_release_keeps_ownership_and_logs_erace(tmp_path, scenario):
    ledger = AuditLedger(tmp_path / "audit.sqlite", scenario=scenario, verify=True)
    clock = ManualClock(value_ms=4_000_000)
    engine = LeaseEngine(scenario, ledger, clock=clock)
    try:
        lease = engine.issue("alpha", 2)
        engine.release(lease.lease_id)
        state_before = engine.snapshot()
        with pytest.raises(BusinessError) as caught:
            engine.release(lease.lease_id)
        assert caught.value.code == ErrorCode.E_RACE
        assert engine.snapshot() == state_before
        race = ledger.events()[-1]
        assert race.event_type == "RACE_REJECTED"
        assert race.payload["ownership_unchanged"] is True
        assert race.payload["ownership"] == []
    finally:
        engine.close()
    assert replay_ledger(tmp_path / "audit.sqlite", scenario)["status"] == "OK"


def test_multiple_leases_never_exceed_source_share(scenario):
    scenario = validate_scenario({**scenario.to_json_dict(), "cluster_capacity": 20})
    # Build a short-lived in-memory-style engine via tempfile would obscure the cap check.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp:
        ledger = AuditLedger(f"{temp}/audit.sqlite", scenario=scenario, verify=True)
        engine = LeaseEngine(scenario, ledger, clock=ManualClock(value_ms=5_000_000))
        try:
            engine.issue("beta", 1)
            engine.issue("beta", 1)
            engine.issue("beta", 1)
            assert engine.sources["beta"].active_granted == 3
            with pytest.raises(BusinessError) as caught:
                engine.issue("beta", 1)
            assert caught.value.code == ErrorCode.E_CAPACITY
        finally:
            engine.close()


def test_issue_many_is_independent_of_input_source_order(tmp_path, scenario):
    def run(order, filename):
        ledger = AuditLedger(tmp_path / filename, scenario=scenario, verify=True)
        engine = LeaseEngine(scenario, ledger, clock=ManualClock(value_ms=9_000_000))
        try:
            engine.issue_many(order, now_ms=9_000_010)
            events = ledger.events()
            return [
                (
                    event.event_type,
                    event.source_id,
                    event.amount,
                    event.payload.get("basis"),
                    event.payload.get("ownership_digest"),
                )
                for event in events
            ]
        finally:
            engine.close()

    forward = run([("alpha", 1), ("beta", 1), ("gamma", 1)], "forward.sqlite")
    reverse = run([("gamma", 1), ("beta", 1), ("alpha", 1)], "reverse.sqlite")
    assert forward == reverse
