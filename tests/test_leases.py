import threading
import time

import pytest

from kvdiag.clock import mono_ms
from kvdiag.errors import LeaseError
from kvdiag.leases import LeaseManager
from kvdiag.log import MemoryLogger
from kvdiag.spec import load_spec


def _mgr(tree):
    return LeaseManager(load_spec(tree["spec"]), logger=MemoryLogger())


def test_issue_and_renew_bumps_version(tree):
    mgr = _mgr(tree)
    lease = mgr.issue("shard-0", "digest-a")
    assert lease["version"] == 1
    renewed = mgr.renew(lease["lease_id"], expected_version=1)
    assert renewed["version"] == 2
    assert renewed["expiry_mono_ms"] >= lease["expiry_mono_ms"]


def test_renew_with_stale_version_rejected(tree):
    mgr = _mgr(tree)
    lease = mgr.issue("shard-0", "digest-a")
    mgr.renew(lease["lease_id"], expected_version=1)
    with pytest.raises(LeaseError) as exc:
        mgr.renew(lease["lease_id"], expected_version=1)
    assert exc.value.code == "E_LEASE"
    assert exc.value.shard_id == "shard-0"
    assert exc.value.details["actual_version"] == 2


def test_expiry_uses_real_monotonic_clock(tree):
    mgr = _mgr(tree)  # ttl 120ms in fixture
    lease = mgr.issue("shard-0", "digest-a")
    assert mgr.expire_due() == []
    deadline = mono_ms() + 2000
    expired = []
    while mono_ms() < deadline and not expired:
        time.sleep(0.02)
        expired = mgr.expire_due()
    assert [l["lease_id"] for l in expired] == [lease["lease_id"]]
    assert mgr.expiry_position("shard-0") == 1
    with pytest.raises(LeaseError):
        mgr.renew(lease["lease_id"], expected_version=1)


def test_inflight_quota_queues_and_records_wait(tree):
    mgr = _mgr(tree)  # max_inflight_renewals = 2
    first = mgr.acquire_renewal_slot("shard-0")
    second = mgr.acquire_renewal_slot("shard-0")
    assert first["wait_end_mono_ms"] >= first["wait_start_mono_ms"]
    acquired = []

    def third():
        acquired.append(mgr.acquire_renewal_slot("shard-0"))

    t = threading.Thread(target=third)
    t.start()
    time.sleep(0.05)
    assert not acquired, "third slot must queue while quota exhausted"
    mgr.release_renewal_slot("shard-0")
    t.join(timeout=5)
    mgr.release_renewal_slot("shard-0")
    mgr.release_renewal_slot("shard-0")
    assert len(acquired) == 1
    wait = acquired[0]
    assert wait["wait_end_mono_ms"] > wait["wait_start_mono_ms"]
    assert wait["wait_end_mono_ms"] - wait["wait_start_mono_ms"] >= 40
