"""要求5/6：租约生命周期、双重上限、真实计时与唯一归属裁决。"""

import time

import pytest

from slo_burn.arbitration import (
    OwnershipArbiter,
    OwnershipEvent,
    RecycleReason,
)
from slo_burn.errors import LeaseError
from slo_burn.lease_client import EvaluatorProcess, LeaseClient
from slo_burn.scheduler import ScheduledTask, WindowScheduler


@pytest.fixture()
def evaluator(tmp_path):
    socket_path = str(tmp_path / "evaluator.sock")
    with EvaluatorProcess(socket_path, global_limit=2, per_service_limit=1) as process:
        yield socket_path


def test_grant_renew_revoke_happy_path(evaluator):
    with LeaseClient(evaluator) as client:
        lease = client.grant("svc-a", ttl_ms=300)["lease"]
        assert lease["version"] == 1
        renewed = client.renew(lease["lease_id"], 1, ttl_ms=300)["lease"]
        assert renewed["granted_ms"] >= lease["granted_ms"]
        assert renewed["expires_ms"] >= lease["expires_ms"]
        time.sleep(0.01)
        renewed_late = client.renew(lease["lease_id"], 1, ttl_ms=300)["lease"]
        assert renewed_late["expires_ms"] > renewed["expires_ms"]
        revoked = client.revoke(lease["lease_id"], 1)["lease"]
        assert revoked["state"] == "revoked"


def test_renew_with_wrong_version_rejected_without_extension(evaluator):
    with LeaseClient(evaluator) as client:
        lease = client.grant("svc-a", ttl_ms=300)["lease"]
        original_expiry = lease["expires_ms"]
        with pytest.raises(LeaseError) as exc:
            client.renew(lease["lease_id"], lease["version"] + 1, ttl_ms=999)
        assert exc.value.code == "E_LEASE"
        snapshot = client.snapshot()
        stored = next(item for item in snapshot["leases"] if item["lease_id"] == lease["lease_id"])
        assert stored["expires_ms"] == original_expiry


def test_per_service_and_global_limits_enforced(evaluator):
    with LeaseClient(evaluator) as client:
        client.grant("svc-a", ttl_ms=500)
        with pytest.raises(LeaseError) as exc:
            client.grant("svc-a", ttl_ms=500)
        assert exc.value.context["per_service_limit"] == 1
        client.grant("svc-b", ttl_ms=500)
        with pytest.raises(LeaseError) as exc:
            client.grant("svc-c", ttl_ms=500)
        assert exc.value.context["global_limit"] == 2


def test_real_monotonic_expiry_reaping(evaluator):
    with LeaseClient(evaluator) as client:
        lease = client.grant("svc-a", ttl_ms=40)["lease"]
        time.sleep(0.08)
        reaped = client.reap()["reaped"]
        assert [item["lease_id"] for item in reaped] == [lease["lease_id"]]
        assert reaped[0]["state"] == "expired"
        with pytest.raises(LeaseError):
            client.renew(lease["lease_id"], lease["version"], ttl_ms=100)


def test_scheduler_projects_to_real_thread_pool_and_records_waits(evaluator):
    with LeaseClient(evaluator) as client:
        scheduler = WindowScheduler(client, global_limit=2, per_service_limit=1)
        tasks = [
            ScheduledTask(
                task_id=f"t-{index}",
                service_id="svc-a" if index % 2 == 0 else "svc-b",
                minute=index,
                work=lambda lease: {"slot": lease["slot_id"]},
                ttl_ms=500,
            )
            for index in range(6)
        ]
        report = scheduler.run(tasks)
        assert report.max_concurrent_observed <= 2
        assert all(record.started_ms is not None for record in report.records)
        assert all(record.wait_ms is not None and record.wait_ms >= 0 for record in report.records)
        assert all(record.status == "completed" for record in report.records)


def _event(ms, slot, service, lease, version, reason, order):
    return OwnershipEvent(ms, service, slot, lease, version, reason, order)


def test_arbitration_prevents_double_count_and_is_order_invariant():
    events = [
        _event(10, 0, "svc-a", "lease-a0v1", 1, RecycleReason.GRANT, 1),
        _event(20, 0, "svc-a", "lease-a0v1", 1, RecycleReason.EXPIRY, 2),
        _event(20, 0, "svc-a", "lease-a0v1", 1, RecycleReason.REVOKE, 3),
    ]
    arbiter = OwnershipArbiter()
    decisions = arbiter.adjudicate_all(events)
    labels = [item.decision for item in decisions]
    assert labels == ["acquired", "recycled-revoke", "rejected-double-count"]
    assert arbiter.slot_owners()[0] is None

    arbiter_reversed = OwnershipArbiter()
    decisions_reversed = arbiter_reversed.adjudicate_all(list(reversed(events)))
    assert [item.decision for item in decisions_reversed] == labels


def test_slot_cannot_be_held_by_two_services_simultaneously():
    arbiter = OwnershipArbiter()
    arbiter.adjudicate_all(
        [_event(0, 0, "svc-a", "lease-a0v1", 1, RecycleReason.GRANT, 1)]
    )
    with pytest.raises(LeaseError):
        arbiter.adjudicate_all(
            [_event(1, 0, "svc-b", "lease-b0v1", 1, RecycleReason.GRANT, 2)]
        )
