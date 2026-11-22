"""Circuit-breaker state machines timed by the real monotonic clock.

States: CLOSED -> OPEN -> HALF_OPEN.  Timing is always taken from
:func:`meshbench.clock.mono_ms` (a real local monotonic source), never from the
possibly-virtual window clock.  Every state transition returns a migration
event that the engine persists in the same SQLite transaction as the batch.
"""

import threading

from .clock import mono_ms

CLOSED = "CLOSED"
OPEN = "OPEN"
HALF_OPEN = "HALF_OPEN"

STATES = (CLOSED, OPEN, HALF_OPEN)


class Breaker:
    """One breaker instance with an explicit half-open probe budget."""

    def __init__(self, instance_id, failure_threshold, cooldown_ms, half_open_probes):
        self.instance_id = instance_id
        self.failure_threshold = int(failure_threshold)
        self.cooldown_ms = int(cooldown_ms)
        self.half_open_probes = int(half_open_probes)
        self._lock = threading.RLock()
        self.state = CLOSED
        self.consecutive_failures = 0
        self.opened_at_ms = None
        self._half_open_inflight = 0
        self._half_open_allowed = 0

    def snapshot(self):
        with self._lock:
            return {
                "instance_id": self.instance_id,
                "state": self.state,
                "consecutive_failures": self.consecutive_failures,
                "opened_at_ms": self.opened_at_ms,
                "half_open_inflight": self._half_open_inflight,
                "half_open_allowed": self._half_open_allowed,
            }

    def restore(self, state, consecutive_failures, opened_at_ms,
                half_open_inflight=0, half_open_allowed=0):
        with self._lock:
            if state not in STATES:
                raise ValueError(f"unknown breaker state {state}")
            self.state = state
            self.consecutive_failures = int(consecutive_failures)
            self.opened_at_ms = opened_at_ms
            self._half_open_inflight = int(half_open_inflight)
            self._half_open_allowed = int(half_open_allowed)

    def _now(self):
        return mono_ms()

    def allow_request(self):
        """Return (allowed, current_state).  May perform OPEN->HALF_OPEN."""
        with self._lock:
            now = self._now()
            if self.state == OPEN:
                if self.opened_at_ms is not None and \
                        now - self.opened_at_ms >= self.cooldown_ms:
                    self.state = HALF_OPEN
                    self._half_open_allowed = self.half_open_probes
                    self._half_open_inflight = 0
                    return True, HALF_OPEN
                return False, OPEN
            if self.state == HALF_OPEN:
                if self._half_open_inflight < self._half_open_allowed:
                    self._half_open_inflight += 1
                    return True, HALF_OPEN
                return False, HALF_OPEN
            return True, CLOSED

    def record_success(self):
        """Return a migration event when the breaker closes, else None."""
        with self._lock:
            if self.state == HALF_OPEN:
                # One explicit probe success closes the breaker.
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
                if self._half_open_inflight == 0:
                    return self._transition_to_closed()
                return None
            if self.state == CLOSED:
                self.consecutive_failures = 0
            return None

    def record_failure(self):
        """Record a real attempt failure; return an OPEN migration event or None."""
        with self._lock:
            now = self._now()
            if self.state == HALF_OPEN:
                # A failed trial reopens immediately and resets the counter.
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
                return self._transition_to_open(now, reason="half_open_probe_failed")
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.failure_threshold:
                return self._transition_to_open(now, reason="threshold_reached")
            return None

    def _transition_to_open(self, now, reason):
        previous = self.state
        self.state = OPEN
        self.opened_at_ms = now
        # Reset the consecutive counter on each (re)open.
        self.consecutive_failures = 0
        self._half_open_inflight = 0
        self._half_open_allowed = 0
        return self._migration(previous, OPEN, reason, now)

    def _transition_to_closed(self):
        previous = self.state
        now = self._now()
        self.state = CLOSED
        self.consecutive_failures = 0
        self.opened_at_ms = None
        self._half_open_inflight = 0
        self._half_open_allowed = 0
        return self._migration(previous, CLOSED, "probe_succeeded", now)

    def _migration(self, previous, current, reason, now):
        return {
            "instance_id": self.instance_id,
            "from_state": previous,
            "to_state": current,
            "reason": reason,
            "mono_ms": now,
        }


class BreakerRegistry:
    """Owns one breaker per instance keyed by (service_id, instance_id)."""

    def __init__(self):
        self._lock = threading.RLock()
        self._breakers = {}

    def get(self, service_id, instance):
        key = (service_id, instance.instance_id)
        with self._lock:
            breaker = self._breakers.get(key)
            if breaker is None:
                breaker = Breaker(
                    instance.instance_id, instance.failure_threshold,
                    instance.cooldown_ms, instance.half_open_probes,
                )
                self._breakers[key] = breaker
            return breaker

    def all_snapshots(self):
        with self._lock:
            rows = []
            for (service_id, instance_id), breaker in sorted(self._breakers.items()):
                snap = breaker.snapshot()
                snap["service_id"] = service_id
                rows.append(snap)
            return rows

    def replace_all(self, rows):
        """Wholesale replacement used by rule evolution and snapshot restore."""
        with self._lock:
            self._breakers = {}
            for row in rows:
                breaker = Breaker(
                    row["instance_id"],
                    row.get("failure_threshold", 1),
                    row.get("cooldown_ms", 0),
                    row.get("half_open_probes", 1),
                )
                breaker.restore(
                    row["state"], row["consecutive_failures"],
                    row.get("opened_at_ms"),
                    row.get("half_open_inflight", 0),
                    row.get("half_open_allowed", 0),
                )
                self._breakers[(row["service_id"], row["instance_id"])] = breaker
