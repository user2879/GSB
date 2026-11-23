from __future__ import annotations

import json
from collections import Counter

from leasebench.engine import LeaseEngine
from leasebench.models import canonical_json
from leasebench.service import group_events
from leasebench.validation import load_declaration

from .conftest import SAMPLES, parse_with_functions


def run_engine(declaration, events, session):
    engine = LeaseEngine(declaration, session)
    payloads = []
    for group in group_events(events):
        batch_id = engine.make_batch_id(group)
        engine.begin_batch(batch_id)
        payloads.append(engine.run_batch(group))
    return engine, payloads


def test_function_order_cannot_change_payloads(events, session) -> None:
    raw = json.loads((SAMPLES / "declarations_v1.json").read_text(encoding="utf-8"))
    original = load_declaration(SAMPLES / "declarations_v1.json")
    reversed_order = parse_with_functions(raw, [2, 1, 0])
    assert original.digest == reversed_order.digest
    _, first = run_engine(original, events, session)
    _, second = run_engine(reversed_order, events, session)
    assert [canonical_json(item) for item in first] == [
        canonical_json(item) for item in second
    ]


def test_borrow_preempt_requeue_and_unique_return(events, session, declaration_v1) -> None:
    engine, payloads = run_engine(declaration_v1, events, session)
    operations = [operation for payload in payloads for operation in payload["operations"]]

    loans = [op["loan"] for op in operations if op["type"] == "BORROW"]
    assert loans
    assert any(
        loan["lender_id"] == "report" and loan["borrower_id"] == "api"
        for loan in loans
    )
    returns = [op["loan"] for op in operations if op["type"] == "LOAN_RETURN"]
    assert len({loan["loan_id"] for loan in returns}) == len(returns)
    assert set(loan["loan_id"] for loan in loans) == set(
        loan["loan_id"] for loan in returns
    )
    assert all(loan["disposition"] for loan in returns)

    preemptions = [
        op["preemption"] for op in operations if op["type"] == "PREEMPT_SCHEDULE"
    ]
    assert preemptions
    assert all(
        item["suspend_at_ms"] >= item["scheduled_ms"] for item in preemptions
    )
    suspend_ops = [
        op["preemption"] for op in operations if op["type"] == "PREEMPT_SUSPEND"
    ]
    assert suspend_ops
    assert all(item["status"] == "SUSPENDED" for item in suspend_ops)
    assert all(item["requeued_queue_id"] for item in suspend_ops)

    release_ids = [
        op["release"]["lease_id"] for op in operations if op["type"] == "RELEASE"
    ]
    assert len(release_ids) == len(set(release_ids))
    dispositions = {
        op["release"]["lease_id"]: op["release"]["disposition"]
        for op in operations
        if op["type"] == "RELEASE"
    }
    assert "PREEMPTED" in dispositions.values()
    assert "BORROW_RETURN_DUE" in dispositions.values()


def test_quota_and_global_limits_hold_for_every_grant(events, session, declaration_v1) -> None:
    engine, payloads = run_engine(declaration_v1, events, session)
    operations = [operation for payload in payloads for operation in payload["operations"]]
    global_used = 0
    per_function = {key: 0 for key in declaration_v1.functions}
    for operation in operations:
        if operation["type"] == "GRANT":
            lease = operation["lease"]
            global_used += lease["concurrency"]
            per_function[lease["function_id"]] += lease["concurrency"]
            assert global_used <= declaration_v1.global_concurrency
            assert per_function[lease["function_id"]] >= 0
        elif operation["type"] == "RELEASE":
            release = operation["release"]
            global_used -= release["concurrency"]
            per_function[release["function_id"]] -= release["concurrency"]
            assert global_used >= 0
            assert per_function[release["function_id"]] >= 0
    assert global_used == engine.global_used
    assert per_function == engine.function_used


def test_explicit_release_frees_budget_and_returns_loan(session, declaration_v1, events) -> None:
    engine = LeaseEngine(declaration_v1, session)
    first = group_events(events)[0]
    engine.begin_batch("B-first")
    engine.run_batch(first)
    second = group_events(events)[1]
    engine.begin_batch("B-second")
    engine.run_batch(second)
    engine.advance_to(200)
    active_loans = [loan for loan in engine.loans.values() if loan.status == "ACTIVE"]
    assert active_loans
    lease = next(
        item for item in engine.leases.values() if item.lease_id == active_loans[0].lease_id
    )
    before = engine.global_used
    engine.begin_batch("B-explicit")
    release = engine.release_lease(lease.lease_id, 250, requeue=False)
    assert engine.global_used == before - lease.concurrency
    assert release.disposition == "EXPLICIT_RELEASE"
    for loan in active_loans:
        assert engine.loans[loan.loan_id].status in {"RETURNED", "STALE"}
        assert engine.loans[loan.loan_id].returned_ms is not None


def test_queue_global_budget_custom(session) -> None:
    from leasebench.validation import parse_declaration

    raw = {
        "declaration_version": 1,
        "global_concurrency": 2,
        "preemption_checkpoint_ms": 10,
        "functions": [
            {
                "function_id": "f",
                "memory_mib": 128,
                "instance_concurrency": 1,
                "lease_ms": 100,
                "min_concurrency": 0,
                "max_concurrency": 2,
                "priority": 1,
            }
        ],
        "preemptions": [],
    }
    declaration = parse_declaration(raw)
    events = [
        {"event_seq": 1, "arrival_ms": 0, "function_id": "f", "expected_occupancy_ms": 50, "concurrency_demand": 2},
        {"event_seq": 2, "arrival_ms": 0, "function_id": "f", "expected_occupancy_ms": 10, "concurrency_demand": 1},
    ]
    from leasebench.models import CallEvent

    calls = [CallEvent(**item) for item in events]
    engine = LeaseEngine(declaration, session)
    engine.begin_batch("B1")
    payload = engine.run_batch(calls)
    app = [op for op in payload["operations"] if op["type"] == "APPLICATION"]
    assert [item["outcome"] for item in app] == ["GRANTED", "QUEUED"]
    assert "GLOBAL_BUDGET" in [item.get("reason") for item in app if item["outcome"] == "QUEUED"]


def test_cancelled_preemption_restores_victim_natural_release(session) -> None:
    from leasebench.models import CallEvent
    from leasebench.validation import parse_declaration

    declaration = parse_declaration(
        {
            "declaration_version": 1,
            "global_concurrency": 7,
            "preemption_checkpoint_ms": 100,
            "functions": [
                {
                    "function_id": "high",
                    "memory_mib": 128,
                    "instance_concurrency": 1,
                    "lease_ms": 1000,
                    "min_concurrency": 0,
                    "max_concurrency": 2,
                    "priority": 30,
                },
                {
                    "function_id": "low",
                    "memory_mib": 128,
                    "instance_concurrency": 1,
                    "lease_ms": 1000,
                    "min_concurrency": 0,
                    "max_concurrency": 2,
                    "priority": 10,
                },
                {
                    "function_id": "middle",
                    "memory_mib": 128,
                    "instance_concurrency": 1,
                    "lease_ms": 1000,
                    "min_concurrency": 0,
                    "max_concurrency": 2,
                    "priority": 20,
                },
            ],
            "preemptions": [
                {"higher": "high", "lower": "low", "return_deadline_ms": 300},
                {"higher": "high", "lower": "middle", "return_deadline_ms": 300},
            ],
        }
    )
    events = [
        CallEvent(1, 0, "high", 1000, 2),
        CallEvent(2, 0, "low", 1000, 2),
        CallEvent(3, 0, "middle", 50, 2),
        CallEvent(4, 10, "high", 100, 1),
    ]
    engine = LeaseEngine(declaration, session)
    engine.begin_batch("B0")
    engine.run_batch(events[:3])
    engine.begin_batch("B10")
    engine.run_batch(events[3:])
    assert any(item.status == "SCHEDULED" for item in engine.preemptions.values())

    engine.advance_to(50)
    assert any(
        item.status == "CANCELLED_BENEFICIARY_GRANTED"
        for item in engine.preemptions.values()
    )
    low = next(lease for lease in engine.leases.values() if lease.function_id == "low")
    assert low.status == "ACTIVE"
    engine.advance_to(1000)
    assert low.status == "RELEASED"
    assert low.release_ms == 1000
    assert low.release_disposition == "CALL_COMPLETED"
