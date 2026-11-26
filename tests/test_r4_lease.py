import pytest

from ckpt_diag.clock import VirtualClock
from ckpt_diag.errors import LeaseError
from ckpt_diag.lease import LeaseManager


def test_issue_and_real_expiry():
    clock = VirtualClock(100)
    manager = LeaseManager(clock.now_ms)
    lease = manager.issue("shard-alpha", ttl_ms=50, batch_id="b1")
    assert lease.lease_version == 1
    assert manager.is_active(lease.lease_id)
    clock.advance(50)
    assert not manager.is_active(lease.lease_id)
    assert manager.expire_due()[0].lease_id == lease.lease_id


def test_renew_optimistic_version_success_then_stale_rejected():
    clock = VirtualClock(0)
    manager = LeaseManager(clock.now_ms)
    lease = manager.issue("shard-alpha", ttl_ms=10, batch_id="b1")
    renewed = manager.renew(lease.lease_id, expected_version=1, ttl_ms=10)
    assert renewed.lease_version == 2 and renewed.renew_count == 1
    # stale version must be E_LEASE and must not extend expiry
    before = renewed.expires_ms
    with pytest.raises(LeaseError) as exc:
        manager.renew(lease.lease_id, expected_version=1, ttl_ms=10)
    assert exc.value.code == "E_LEASE"
    assert manager.get(lease.lease_id).expires_ms == before


def test_same_lease_not_held_by_two_batches():
    manager = LeaseManager(VirtualClock(0).now_ms)
    manager.issue("shard-alpha", ttl_ms=10, batch_id="b1")
    with pytest.raises(LeaseError):
        manager.issue("shard-alpha", ttl_ms=10, batch_id="b1")


def test_cannot_renew_expired_lease():
    clock = VirtualClock(0)
    manager = LeaseManager(clock.now_ms)
    lease = manager.issue("shard-alpha", ttl_ms=5, batch_id="b1")
    clock.advance(6)
    with pytest.raises(LeaseError):
        manager.renew(lease.lease_id, expected_version=1, ttl_ms=10)


def test_release_frees_batch_holder():
    manager = LeaseManager(VirtualClock(0).now_ms)
    lease = manager.issue("shard-alpha", ttl_ms=10, batch_id="b1")
    manager.release(lease.lease_id, expected_version=1)
    lease2 = manager.issue("shard-alpha", ttl_ms=10, batch_id="b1")
    assert lease2.lease_id != lease.lease_id
