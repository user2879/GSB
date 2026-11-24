from __future__ import annotations

from faultisolation.budget import compute_usage
from faultisolation.clock import LogicalClock
from faultisolation.engine import Engine


def test_peak_usage_within_domain_slots(small_bundle):
    engine = Engine(small_bundle, LogicalClock(0))
    usage, global_usage = compute_usage(engine)
    for snap in usage:
        assert 0 <= snap.current_used <= snap.slots
        assert 0 <= snap.peak_used <= snap.slots
    assert 0 <= global_usage.peak_used <= global_usage.total_slots
    assert global_usage.current_used <= global_usage.total_slots


def test_peak_at_least_current(small_bundle):
    engine = Engine(small_bundle, LogicalClock(0))
    usage, _ = compute_usage(engine)
    for snap in usage:
        assert snap.peak_used >= snap.current_used
