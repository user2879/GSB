from __future__ import annotations

import threading
import time

import pytest

from dns_local.cache import Cache, CacheEntry, CacheKey, FillDiscarded
from dns_local.clock import FakeClock
from dns_local.wire import ResourceRecord


def make_entry(key: CacheKey, ttl: int, source_serial: int = 1, value: str = "192.0.2.1") -> CacheEntry:
    return CacheEntry(
        key=key,
        rcode=0,
        answers=(ResourceRecord(key.name, key.qtype, 1, ttl, value),),
        authority=(),
        ttl=ttl,
        stored_ns=1_000_000_000_000,
        source="ZONE",
        source_serial=source_serial,
        attribution=("test=true",),
    )


def test_ttl_decrements_on_monotonic_clock_and_expires() -> None:
    clock = FakeClock()
    cache = Cache(clock, capacity=4)
    key = CacheKey.create("www.lab.test.", "A")
    entry = make_entry(key, 10)
    entry = CacheEntry(**{**entry.__dict__, "stored_ns": clock.monotonic_ns()})
    cache.put(entry)
    assert cache.get(key).remaining_ttl == 10
    clock.advance(3.9)
    assert cache.get(key).remaining_ttl == 7
    clock.advance(6.1)
    assert cache.get(key) is None
    assert len(cache) == 0


def test_aged_records_are_complete_and_decremented() -> None:
    clock = FakeClock()
    cache = Cache(clock, capacity=4)
    key = CacheKey.create("www.lab.test.", "A")
    entry = make_entry(key, 10)
    entry = CacheEntry(**{**entry.__dict__, "stored_ns": clock.monotonic_ns()})
    cache.put(entry)
    clock.advance(4)
    hit = cache.get(key)
    answers, authority = hit.entry.aged_records(clock.monotonic_ns())
    assert answers[0].ttl == 6
    assert authority == ()


def test_capacity_eviction_is_lru_with_stable_tiebreak() -> None:
    clock = FakeClock()
    cache = Cache(clock, capacity=2)
    a = CacheKey.create("a.lab.test.", "A")
    b = CacheKey.create("b.lab.test.", "A")
    c = CacheKey.create("c.lab.test.", "A")
    cache.put(make_entry(a, 60))
    cache.put(make_entry(b, 60))
    assert cache.get(a) is not None
    cache.put(make_entry(c, 60))
    assert cache.get(b) is None
    assert cache.get(a) is not None
    assert cache.get(c) is not None


def test_singleflight_converges_to_one_source_serial() -> None:
    clock = FakeClock()
    cache = Cache(clock, capacity=8)
    key = CacheKey.create("api.upstream.test.", "A")
    barrier = threading.Barrier(8)
    counter = {"loads": 0}
    results: list[tuple[int, bool]] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        hit = cache.get_or_fill(
            key,
            lambda: (
                counter.__setitem__("loads", counter["loads"] + 1)
                or time.sleep(0.05)
                or make_entry(key, 30, source_serial=9)
            ),
        )
        with lock:
            results.append((hit.entry.source_serial, hit.filled))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert counter["loads"] == 1
    assert len(results) == 8
    assert {serial for serial, _ in results} == {9}
    assert sum(1 for _, filled in results if filled) == 1
    assert len(cache.entries()) == 1


def test_stale_fill_is_discarded() -> None:
    clock = FakeClock()
    cache = Cache(clock, capacity=8)
    key = CacheKey.create("www.lab.test.", "A")
    with pytest.raises(FillDiscarded):
        cache.get_or_fill(key, lambda: make_entry(key, 30), commit_guard=lambda: False)
    assert cache.get(key) is None


def test_replace_all_is_an_atomic_old_or_new_view() -> None:
    clock = FakeClock()
    cache = Cache(clock, capacity=8)
    old = [make_entry(CacheKey.create(f"old{i}.lab.test.", "A"), 30) for i in range(4)]
    new = [make_entry(CacheKey.create(f"new{i}.lab.test.", "A"), 30, source_serial=2) for i in range(4)]
    cache.replace_all(old)
    stop = threading.Event()
    errors: list[str] = []

    def reader() -> None:
        while not stop.is_set():
            rows = cache.entries()
            prefixes = {row.key.name.split(".")[0][:3] for row in rows}
            if prefixes not in ({"old"}, {"new"}):
                errors.append(repr(prefixes))
                return

    thread = threading.Thread(target=reader)
    thread.start()
    for _ in range(100):
        cache.replace_all(new)
        cache.replace_all(old)
    stop.set()
    thread.join(timeout=3)
    assert errors == []
    assert {entry.key.name.split(".")[0][:3] for entry in cache.entries()} in ({"old"}, {"new"})
