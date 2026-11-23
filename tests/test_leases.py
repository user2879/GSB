import time

import pytest

from shardbench.clock import ManualClock
from shardbench.errors import BizError, E_LEASE
from shardbench.leases import LeaseManager


def test_lease_fields_and_renew():
    clock = ManualClock(1000)
    mgr = LeaseManager(ttl_ms=500, clock=clock)
    lease = mgr.acquire(db=1)
    assert lease.granted_ms == 1000
    assert lease.expires_ms == 1500
    assert lease.version == 1
    renewed = mgr.renew(lease.lease_id, expected_version=1)
    assert renewed.version == 2
    assert renewed.expires_ms == 1500  # clock did not move


def test_renew_with_stale_version_rejected_and_not_extended():
    clock = ManualClock(0)
    mgr = LeaseManager(ttl_ms=500, clock=clock)
    lease = mgr.acquire(db=0)
    mgr.renew(lease.lease_id, expected_version=1)
    before = lease.expires_ms
    with pytest.raises(BizError) as ei:
        mgr.renew(lease.lease_id, expected_version=1)  # stale: now version 2
    assert ei.value.code == E_LEASE
    assert ei.value.detail["current_version"] == 2
    assert lease.expires_ms == before  # not extended


def test_real_expiry_releases_allowance_and_records_reason():
    clock = ManualClock(0)
    mgr = LeaseManager(ttl_ms=200, clock=clock)
    lease = mgr.acquire(db=3)
    assert mgr.active_count(3) == 1
    clock.advance(201)
    reclaimed = mgr.reap()
    assert [l.lease_id for l in reclaimed] == [lease.lease_id]
    assert mgr.active_count(3) == 0
    assert lease.state == "expired"
    assert "ttl_expired" in lease.reclaim_reason
    with pytest.raises(BizError) as ei:
        mgr.renew(lease.lease_id, expected_version=1)
    assert ei.value.code == E_LEASE


def test_expiry_uses_real_monotonic_clock():
    mgr = LeaseManager(ttl_ms=30)  # real clock
    lease = mgr.acquire(db=0)
    assert mgr.active_count(0) == 1
    time.sleep(0.06)  # genuinely wait past the TTL
    assert mgr.active_count(0) == 0
    assert lease.state == "expired"


def test_release():
    mgr = LeaseManager(ttl_ms=1000, clock=ManualClock(0))
    lease = mgr.acquire(db=0)
    mgr.release(lease.lease_id)
    assert lease.state == "released"
    assert mgr.active_count(0) == 0
