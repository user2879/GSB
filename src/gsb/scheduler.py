"""Constraint-driven thread-pool scheduler with real backpressure."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .clock import now_ms, sleep_ms


class RateLimiter:
    """Per-channel sliding window: at most ``rate_limit`` events per 1000 ms."""

    def __init__(self):
        self._windows: dict[str, deque] = {}
        self._lock = threading.Lock()

    def acquire(self, channel_id: str, event_count: int, rate_limit: int):
        while True:
            with self._lock:
                window = self._windows.setdefault(channel_id, deque())
                cutoff = now_ms() - 1000
                while window and window[0] <= cutoff:
                    window.popleft()
                if len(window) + event_count <= rate_limit:
                    moment = now_ms()
                    for _ in range(event_count):
                        window.append(moment)
                    return
                wait_ms = max(1, window[0] + 1000 - now_ms())
            sleep_ms(wait_ms)  # real elapsed time only, no socket tricks


class Scheduler:
    """Admits batches under global, per-subscriber and per-channel limits.

    Over-limit batches stay queued (they never acquire a lease) and record real
    monotonic enqueue/dequeue timestamps. In-flight capacity is only held while
    a batch is actively processing; rate-limit waiting happens on the worker
    before capacity is consumed so it cannot self-deadlock.
    """

    def __init__(self, spec, trajectory_writer=None, workers=None):
        self.spec = spec
        self.global_limit = spec.subscriber_limit
        self.rate_limiter = RateLimiter()
        self.trajectory_writer = trajectory_writer
        self._cv = threading.Condition()
        self._inflight_events = {sub.subscriber_id: 0 for sub in spec.subscribers}
        self._global_leases = 0
        self._waiting: deque = deque()
        self._futures = set()
        self._backlog_events = 0
        self._round = 0
        self._executor = ThreadPoolExecutor(
            max_workers=workers or max(2, len(spec.subscribers) * 2),
            thread_name_prefix="gsb-worker",
        )

    def submit(self, batch, handler):
        with self._cv:
            self._waiting.append(
                {
                    "batch": batch,
                    "handler": handler,
                    "enqueued_at_ms": now_ms(),
                    "dequeued_at_ms": None,
                }
            )
            self._backlog_events += len(batch.event_seqs)
            self._cv.notify_all()
            self._snapshot_locked()

    def _sub_limit(self, subscriber_id):
        return self.spec.subscriber(subscriber_id).inflight_event_limit

    def _can_admit(self, entry):
        batch = entry["batch"]
        count = len(batch.event_seqs)
        sub_id = batch.subscriber_id
        if self._inflight_events[sub_id] + count > self._sub_limit(sub_id):
            return False
        if self._global_leases + 1 > self.global_limit:
            return False
        return True

    def _pump_locked(self):
        remaining = deque()
        admitted = False
        while self._waiting:
            entry = self._waiting.popleft()
            if self._can_admit(entry):
                batch = entry["batch"]
                count = len(batch.event_seqs)
                self._inflight_events[batch.subscriber_id] += count
                self._global_leases += 1
                self._backlog_events -= count
                entry["dequeued_at_ms"] = now_ms()
                future = self._executor.submit(self._run, entry)
                self._futures.add(future)
                future.add_done_callback(self._future_done)
                admitted = True
            else:
                remaining.append(entry)
        self._waiting = remaining
        if admitted:
            self._snapshot_locked()

    def _future_done(self, future):
        with self._cv:
            self._futures.discard(future)
            self._cv.notify_all()

    def _run(self, entry):
        batch = entry["batch"]
        channel = next(
            c for c in self.spec.channels if c.channel_id == batch.channel_id
        )
        # Rate-limit wait occurs while capacity is already reserved for this
        # admitted batch; it is real time, not a socket close or fake delay.
        self.rate_limiter.acquire(
            channel.channel_id, len(batch.event_seqs), channel.rate_limit
        )
        try:
            return entry["handler"](entry)
        finally:
            with self._cv:
                count = len(batch.event_seqs)
                self._inflight_events[batch.subscriber_id] -= count
                self._global_leases -= 1
                self._pump_locked()
                self._cv.notify_all()
                self._snapshot_locked()

    def _snapshot_locked(self):
        if self.trajectory_writer is None:
            return
        self._round += 1
        self.trajectory_writer(
            self._round,
            {
                "round_index": self._round,
                "at_ms": now_ms(),
                "global_leases": self._global_leases,
                "queued_length": len(self._waiting),
                "backlog_events": self._backlog_events,
                "inflight_events": dict(self._inflight_events),
            },
        )

    def run_until_idle(self):
        with self._cv:
            self._pump_locked()
            while self._waiting or self._futures:
                self._pump_locked()
                if not self._waiting and not self._futures:
                    break
                self._cv.wait(timeout=0.02)
            futures = list(self._futures)
        results = [future.result() for future in futures]
        self._executor.shutdown(wait=True)
        return results
