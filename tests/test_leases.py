from __future__ import annotations

from quantledger.errors import QLabError
from quantledger.leases import LeaseBook


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000

    def __call__(self) -> int:
        return self.now


def test_lease_generation_stale_rejections_and_real_expiry_reclaim():
    clock = FakeClock()
    book = LeaseBook(worker_id="w", concurrency_quota=1, lease_limit=1, clock_ms=clock)
    first = book.grant(batch_id="b1", revision=1, ttl_ms=10)
    try:
        book.renew(lease_id=first.lease_id, generation=first.generation, ttl_ms=10)
    except QLabError:
        raise AssertionError("current generation renewal must succeed")
    try:
        book.renew(lease_id=first.lease_id, generation=first.generation - 1, ttl_ms=10)
        raise AssertionError("stale renewal must fail")
    except QLabError as exc:
        assert exc.code == "E_STALE_GENERATION"
        assert book.get(first.lease_id).generation == first.generation  # type: ignore[union-attr]
    clock.now += 20
    events = book.expire()
    assert [event["code"] for event in events] == ["E_LEASE_EXPIRED"]
    assert book.status()["active_leases"] == 0
    second = book.grant(batch_id="b2", revision=2, ttl_ms=10)
    assert second.generation > first.generation
    released = book.release(lease_id=second.lease_id, generation=second.generation)
    assert released["active_leases"] == 0
    try:
        book.release(lease_id=second.lease_id, generation=second.generation - 1)
        raise AssertionError("stale release must fail")
    except QLabError as exc:
        assert exc.code in {"E_STALE_GENERATION", "E_LEASE_EXPIRED"}

