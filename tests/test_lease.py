from __future__ import annotations

import time
from dataclasses import replace

import pytest

from ossdiag.errors import LeaseError
from ossdiag.lease import LeaseManager
from ossdiag.models import LeasePolicy


def policy(max_leases: int = 2, lease_ms: int = 500, renew_ms: int = 100) -> LeasePolicy:
    return LeasePolicy(lease_ms=lease_ms, renew_interval_ms=renew_ms, max_concurrent_leases=max_leases)


def test_duplicate_active_grant_returns_same_version_and_is_recorded() -> None:
    manager = LeaseManager(global_quota=4)
    first = manager.grant("a", 0, policy())
    second = manager.grant("a", 0, policy())
    assert second.version == first.version
    assert manager.duplicate_rejected_count == 1
    assert manager.snapshot()["duplicate_rejected_count"] == 1


def test_global_and_object_quota_reject_overflow() -> None:
    manager = LeaseManager(global_quota=1)
    manager.grant("a", 0, policy())
    with pytest.raises(LeaseError) as caught:
        manager.grant("b", 0, policy())
    assert caught.value.code == "E_LEASE"
    assert caught.value.detail["scope"] == "global"

    manager = LeaseManager(global_quota=8)
    manager.grant("a", 0, policy(max_leases=1))
    with pytest.raises(LeaseError) as caught:
        manager.grant("a", 1, policy(max_leases=1))
    assert caught.value.code == "E_LEASE"
    assert caught.value.detail["scope"] == "object"


def test_renew_requires_matching_version_and_does_not_extend_on_failure() -> None:
    manager = LeaseManager(global_quota=2)
    lease = manager.grant("a", 0, policy())
    active_before = manager.snapshot()["active_leases"][0]
    with pytest.raises(LeaseError) as caught:
        manager.renew(replace(lease, version=999), 500)
    assert caught.value.code == "E_LEASE"
    active_after = manager.snapshot()["active_leases"][0]
    assert active_after["expires_ms"] == active_before["expires_ms"]
    renewed = manager.renew(lease, 500)
    assert renewed.renew_count == 1
    assert renewed.expires_ms >= lease.expires_ms


def test_real_monotonic_expiry_reclaims_quota() -> None:
    events: list[dict] = []
    manager = LeaseManager(global_quota=1, event_callback=events.append)
    first = manager.grant("a", 0, policy(lease_ms=25, renew_ms=5))
    time.sleep(0.04)
    assert manager.active_count == 0
    assert any(item.get("event") == "lease_expired" for item in events)
    second = manager.grant("b", 0, policy(lease_ms=25, renew_ms=5))
    assert second.version > first.version
    assert manager.snapshot()["active_global"] == 1
