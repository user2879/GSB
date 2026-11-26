import time

from vertical_workbench.clock import ManualClock, MonotonicClock
from vertical_workbench.codes import E_LEASE_GENERATION
from vertical_workbench.leases import LeaseExecutor


def test_wrong_generation_is_rejected_and_does_not_advance():
    executor = LeaseExecutor(ManualClock(), max_active_leases=4)
    lease = executor.acquire("img:c0", 0, 50, 3, 1000)["lease"]
    before = executor.current_generation
    response = executor.release(lease["lease_id"], lease["generation"] + 1)
    assert response["code"] == E_LEASE_GENERATION
    assert executor.current_generation == before
    assert executor.active_lease_for_target("img:c0") is not None


def test_real_monotonic_clock_reclaims_expired_lease():
    executor = LeaseExecutor(MonotonicClock(), max_active_leases=4)
    granted = executor.acquire("img:c1", 60, 120, reserved_points=9, ttl_ms=30)
    assert granted["ok"]
    assert executor.reserved_points == 9
    time.sleep(0.1)
    expired = executor.reap_expired()
    assert len(expired) == 1
    assert expired[0]["code"] == "E_LEASE_EXPIRED"
    assert executor.active_lease_for_target("img:c1") is None
    assert executor.reserved_points == 0
