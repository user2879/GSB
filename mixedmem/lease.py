"""Block pool with exclusive, expiring leases backed by a monotonic clock."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .errors import budget_error, lease_error
from .util import Clock, monotonic_ms


def _blocks_text(start: int, count: int) -> list[int]:
    return [start, start + count]


@dataclass
class Lease:
    lease_id: str
    batch_id: str
    layer_id: str
    plan_version: int
    block_start: int
    block_count: int
    granted_ms: int
    expires_ms: int
    status: str = "active"
    released_ms: int | None = None
    reason: str | None = None
    release_result: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def block_range(self) -> list[int]:
        return _blocks_text(self.block_start, self.block_count)

    @property
    def blocks(self) -> range:
        return range(self.block_start, self.block_start + self.block_count)

    def is_expired(self, now_ms: int) -> bool:
        return self.status == "active" and now_ms >= self.expires_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "batch_id": self.batch_id,
            "layer_id": self.layer_id,
            "plan_version": self.plan_version,
            "block_range": self.block_range,
            "block_start": self.block_start,
            "block_count": self.block_count,
            "granted_ms": self.granted_ms,
            "expires_ms": self.expires_ms,
            "status": self.status,
            "released_ms": self.released_ms,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LeaseRequest:
    layer_id: str
    blocks: int
    preferred_start: int | None = None


class LeasePool:
    """A bounded block allocator.

    Allocation is first-fit over sorted active intervals.  A bundle is
    all-or-nothing, which prevents a batch from holding a partial layer set
    while waiting for the rest of its budget.  All state transitions use the
    caller supplied ``Clock`` and integer monotonic milliseconds.
    """

    def __init__(
        self,
        *,
        block_size_bytes: int,
        memory_budget_bytes: int,
        clock: Clock | None = None,
        id_prefix: str = "L",
    ) -> None:
        if block_size_bytes <= 0:
            raise ValueError("block_size_bytes must be positive")
        if memory_budget_bytes < block_size_bytes:
            raise ValueError("memory_budget_bytes must hold at least one block")
        self.block_size_bytes = int(block_size_bytes)
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.total_blocks = self.memory_budget_bytes // self.block_size_bytes
        self.clock = clock or Clock()
        self.id_prefix = id_prefix
        self._condition = threading.Condition(threading.RLock())
        self._leases: dict[str, Lease] = {}
        self._active: dict[str, Lease] = {}
        self._next_sequence = 1
        self._events: list[dict[str, Any]] = []
        self._reaper: threading.Thread | None = None
        self._stop_reaper = threading.Event()

    def _now(self) -> int:
        return int(self.clock.monotonic_ms())

    def now_ms(self) -> int:
        return self._now()

    def _active_intervals_locked(self) -> list[tuple[int, int, Lease]]:
        return sorted(
            (
                (lease.block_start, lease.block_start + lease.block_count, lease)
                for lease in self._active.values()
            ),
            key=lambda item: (item[0], item[1], item[2].lease_id),
        )

    def _is_free_locked(self, start: int, count: int) -> bool:
        if start < 0 or count <= 0 or start + count > self.total_blocks:
            return False
        end = start + count
        for active_start, active_end, _ in self._active_intervals_locked():
            if start < active_end and active_start < end:
                return False
        return True

    def _first_fit_locked(self, count: int) -> int | None:
        if count <= 0 or count > self.total_blocks:
            return None
        cursor = 0
        for start, end, _ in self._active_intervals_locked():
            if cursor + count <= start:
                return cursor
            cursor = max(cursor, end)
        if cursor + count <= self.total_blocks:
            return cursor
        return None

    @staticmethod
    def _request_counts(requests: Sequence[LeaseRequest]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for request in requests:
            if request.blocks <= 0:
                raise lease_error(
                    "lease request must contain a positive block count",
                    layer_id=request.layer_id,
                    field="blocks",
                )
            if request.layer_id in counts:
                raise lease_error(
                    "duplicate layer in lease bundle",
                    layer_id=request.layer_id,
                    field="layer_id",
                )
            counts[request.layer_id] = request.blocks
        return counts

    def _try_allocate_locked(
        self,
        batch_id: str,
        requests: Sequence[LeaseRequest],
        ttl_ms: int,
        plan_version: int,
        grant_ms: int,
    ) -> tuple[Lease, ...] | None:
        placements: list[tuple[LeaseRequest, int]] = []
        reserved: list[tuple[int, int]] = []
        for request in requests:
            start: int | None = None
            if request.preferred_start is not None:
                candidate = request.preferred_start
                if self._is_free_locked(candidate, request.blocks) and not any(
                    candidate < end and start_reserved < candidate + request.blocks
                    for start_reserved, end in reserved
                ):
                    start = candidate
            if start is None:
                # First-fit is repeated while treating already chosen bundle
                # ranges as occupied.
                occupied = self._active_intervals_locked()
                cursor = 0
                for active_start, active_end, _ in occupied:
                    if cursor + request.blocks <= active_start and not any(
                        cursor < end and start_reserved < cursor + request.blocks
                        for start_reserved, end in reserved
                    ):
                        start = cursor
                        break
                    cursor = max(cursor, active_end)
                if start is None and cursor + request.blocks <= self.total_blocks and not any(
                    cursor < end and start_reserved < cursor + request.blocks
                    for start_reserved, end in reserved
                ):
                    start = cursor
                if start is None:
                    # Dense scan handles holes introduced by earlier chosen
                    # bundle placements that first-fit adjacent-walk missed.
                    for candidate in range(self.total_blocks - request.blocks + 1):
                        if not self._is_free_locked(candidate, request.blocks):
                            continue
                        if any(
                            candidate < end and start_reserved < candidate + request.blocks
                            for start_reserved, end in reserved
                        ):
                            continue
                        start = candidate
                        break
            if start is None:
                return None
            placements.append((request, start))
            reserved.append((start, start + request.blocks))

        leases: list[Lease] = []
        for request, start in placements:
            sequence = self._next_sequence
            self._next_sequence += 1
            lease_id = (
                f"{self.id_prefix}{plan_version}-{batch_id}-{request.layer_id}-{sequence:05d}"
            )
            lease = Lease(
                lease_id=lease_id,
                batch_id=batch_id,
                layer_id=request.layer_id,
                plan_version=plan_version,
                block_start=start,
                block_count=request.blocks,
                granted_ms=grant_ms,
                expires_ms=grant_ms + ttl_ms,
            )
            self._leases[lease_id] = lease
            self._active[lease_id] = lease
            leases.append(lease)
            self._events.append(
                {
                    "event": "grant",
                    "lease_id": lease_id,
                    "batch_id": batch_id,
                    "layer_id": request.layer_id,
                    "block_range": lease.block_range,
                    "at_ms": grant_ms,
                    "expires_ms": lease.expires_ms,
                }
            )
        return tuple(leases)

    def acquire_bundle(
        self,
        batch_id: str,
        requests: Sequence[LeaseRequest],
        *,
        ttl_ms: int,
        plan_version: int = 1,
        timeout_ms: int | None = None,
    ) -> tuple[Lease, ...]:
        if not isinstance(ttl_ms, int) or isinstance(ttl_ms, bool) or ttl_ms <= 0:
            raise lease_error("ttl_ms must be a positive integer", field="lease_ttl_ms")
        if not batch_id:
            raise lease_error("batch_id must be non-empty", field="batch_id")
        if not requests:
            return ()
        counts = self._request_counts(requests)
        required = sum(counts.values())
        if required > self.total_blocks:
            raise budget_error(
                "lease bundle exceeds the block budget",
                required_blocks=required,
                budget_blocks=self.total_blocks,
            )
        deadline = None if timeout_ms is None else self._now() + timeout_ms
        with self._condition:
            while True:
                now = self._now()
                self._expire_due_locked(now)
                allocated = self._try_allocate_locked(
                    batch_id, list(requests), ttl_ms, plan_version, now
                )
                if allocated is not None:
                    self._condition.notify_all()
                    return allocated
                if deadline is not None and now >= deadline:
                    raise budget_error(
                        "timed out waiting for lease budget",
                        required_blocks=required,
                        active_blocks=self._used_blocks_locked(),
                        budget_blocks=self.total_blocks,
                    )
                active = self._active_intervals_locked()
                wait_seconds = 0.02
                if active:
                    next_expiry = min(lease.expires_ms for _, _, lease in active)
                    wait_seconds = max(0.001, min(wait_seconds, (next_expiry - now) / 1000.0))
                self._condition.wait(timeout=wait_seconds)

    def acquire(
        self,
        batch_id: str,
        layer_id: str,
        blocks: int,
        *,
        ttl_ms: int,
        plan_version: int = 1,
        preferred_start: int | None = None,
        timeout_ms: int | None = None,
    ) -> Lease:
        leases = self.acquire_bundle(
            batch_id,
            [LeaseRequest(layer_id, blocks, preferred_start)],
            ttl_ms=ttl_ms,
            plan_version=plan_version,
            timeout_ms=timeout_ms,
        )
        return leases[0]

    def _allocated_result(self, lease: Lease) -> dict[str, Any]:
        return {
            "ok": True,
            "lease_id": lease.lease_id,
            "batch_id": lease.batch_id,
            "layer_id": lease.layer_id,
            "status": lease.status,
            "released_ms": lease.released_ms,
            "released_block_range": lease.block_range,
            "early": lease.status == "released",
            "reason": lease.reason,
        }

    def release(
        self,
        lease_id: str,
        *,
        batch_id: str,
        layer_id: str | None = None,
    ) -> dict[str, Any]:
        with self._condition:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise lease_error(
                    "unknown lease identifier", field="lease_id", lease_id=lease_id
                )
            if lease.batch_id != batch_id:
                raise lease_error(
                    "lease ownership mismatch",
                    layer_id=lease.layer_id,
                    field="batch_id",
                    expected=lease.batch_id,
                    actual=batch_id,
                    lease_id=lease_id,
                )
            if layer_id is not None and lease.layer_id != layer_id:
                raise lease_error(
                    "lease ownership mismatch",
                    layer_id=lease.layer_id,
                    field="layer_id",
                    expected=lease.layer_id,
                    actual=layer_id,
                    lease_id=lease_id,
                )
            if lease.release_result is not None:
                return dict(lease.release_result)
            now = self._now()
            if lease.status == "active" and now >= lease.expires_ms:
                self._expire_lease_locked(lease, now)
            elif lease.status == "active":
                lease.status = "released"
                lease.released_ms = now
                lease.reason = "early_release"
                self._active.pop(lease.lease_id, None)
                lease.release_result = self._allocated_result(lease)
                self._events.append(
                    {
                        "event": "release",
                        "lease_id": lease_id,
                        "batch_id": batch_id,
                        "layer_id": lease.layer_id,
                        "block_range": lease.block_range,
                        "at_ms": now,
                        "reason": lease.reason,
                    }
                )
                self._condition.notify_all()
            elif lease.release_result is None:
                lease.release_result = self._allocated_result(lease)
            return dict(lease.release_result)

    def _expire_lease_locked(self, lease: Lease, now: int) -> None:
        if lease.status != "active":
            return
        lease.status = "expired"
        lease.released_ms = now
        lease.reason = "ttl_elapsed"
        self._active.pop(lease.lease_id, None)
        lease.release_result = self._allocated_result(lease)
        self._events.append(
            {
                "event": "expire",
                "lease_id": lease.lease_id,
                "batch_id": lease.batch_id,
                "layer_id": lease.layer_id,
                "block_range": lease.block_range,
                "at_ms": now,
                "reason": lease.reason,
            }
        )

    def _expire_due_locked(self, now: int) -> list[Lease]:
        expired = sorted(
            (lease for lease in self._active.values() if lease.is_expired(now)),
            key=lambda lease: (lease.expires_ms, lease.lease_id),
        )
        for lease in expired:
            self._expire_lease_locked(lease, now)
        if expired:
            self._condition.notify_all()
        return expired

    def expire_due(self) -> list[Lease]:
        with self._condition:
            return self._expire_due_locked(self._now())

    def assert_active(self, lease_id: str, *, batch_id: str) -> Lease:
        with self._condition:
            now = self._now()
            self._expire_due_locked(now)
            lease = self._leases.get(lease_id)
            if lease is None or lease.batch_id != batch_id:
                raise lease_error(
                    "lease ownership mismatch or unknown lease",
                    field="lease_id",
                    lease_id=lease_id,
                )
            if lease.status != "active":
                raise lease_error(
                    "lease is no longer active",
                    layer_id=lease.layer_id,
                    field="status",
                    lease_id=lease_id,
                    status=lease.status,
                )
            return lease

    def revoke(self, lease_id: str, *, reason: str) -> dict[str, Any]:
        with self._condition:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise lease_error("unknown lease identifier", lease_id=lease_id, field="lease_id")
            if lease.release_result is not None:
                return dict(lease.release_result)
            now = self._now()
            if lease.status != "active":
                lease.release_result = self._allocated_result(lease)
                return dict(lease.release_result)
            lease.status = "revoked"
            lease.released_ms = now
            lease.reason = reason
            self._active.pop(lease.lease_id, None)
            lease.release_result = self._allocated_result(lease)
            self._events.append(
                {
                    "event": "revoke",
                    "lease_id": lease.lease_id,
                    "batch_id": lease.batch_id,
                    "layer_id": lease.layer_id,
                    "block_range": lease.block_range,
                    "at_ms": now,
                    "reason": reason,
                }
            )
            self._condition.notify_all()
            return dict(lease.release_result)

    def revoke_plan(self, plan_version: int, *, reason: str) -> list[dict[str, Any]]:
        with self._condition:
            ids = sorted(
                lease.lease_id
                for lease in self._active.values()
                if lease.plan_version == plan_version
            )
        return [self.revoke(lease_id, reason=reason) for lease_id in ids]

    def _used_blocks_locked(self) -> int:
        return sum(lease.block_count for lease in self._active.values())

    def used_blocks(self) -> int:
        with self._condition:
            self._expire_due_locked(self._now())
            return self._used_blocks_locked()

    def free_ranges(self) -> list[list[int]]:
        with self._condition:
            self._expire_due_locked(self._now())
            ranges: list[list[int]] = []
            cursor = 0
            for start, end, _ in self._active_intervals_locked():
                if cursor < start:
                    ranges.append([cursor, start])
                cursor = max(cursor, end)
            if cursor < self.total_blocks:
                ranges.append([cursor, self.total_blocks])
            return ranges

    def all_leases(self) -> list[dict[str, Any]]:
        with self._condition:
            self._expire_due_locked(self._now())
            return [
                self._leases[key].to_dict()
                for key in sorted(self._leases)
            ]

    def active_leases(self) -> list[dict[str, Any]]:
        with self._condition:
            self._expire_due_locked(self._now())
            return [
                lease.to_dict()
                for lease in sorted(
                    self._active.values(), key=lambda item: (item.granted_ms, item.lease_id)
                )
            ]

    def current_view(self) -> dict[str, Any]:
        with self._condition:
            now = self._now()
            self._expire_due_locked(now)
            active = [
                lease.to_dict()
                for lease in sorted(
                    self._active.values(), key=lambda item: (item.block_start, item.lease_id)
                )
            ]
            used = self._used_blocks_locked()
            return {
                "now_ms": now,
                "block_size_bytes": self.block_size_bytes,
                "memory_budget_bytes": self.memory_budget_bytes,
                "total_blocks": self.total_blocks,
                "used_blocks": used,
                "used_bytes": used * self.block_size_bytes,
                "within_budget": used <= self.total_blocks,
                "free_ranges": self.free_ranges(),
                "active_leases": active,
            }

    def history(self) -> list[dict[str, Any]]:
        return self.all_leases()

    def drain_events(self) -> list[dict[str, Any]]:
        with self._condition:
            events = [dict(item) for item in self._events]
            self._events.clear()
            return events

    def verify_invariants(self) -> dict[str, Any]:
        with self._condition:
            intervals = sorted(
                (lease.block_start, lease.block_start + lease.block_count, lease.lease_id)
                for lease in self._active.values()
            )
            overlaps: list[list[str]] = []
            for left, right in zip(intervals, intervals[1:]):
                if left[1] > right[0]:
                    overlaps.append([left[2], right[2]])
            used = sum(end - start for start, end, _ in intervals)
            return {
                "ok": not overlaps and used <= self.total_blocks,
                "overlaps": overlaps,
                "used_blocks": used,
                "total_blocks": self.total_blocks,
                "within_budget": used <= self.total_blocks,
            }

    def start_reaper(self, interval_ms: int = 5) -> None:
        if self._reaper is not None and self._reaper.is_alive():
            return
        self._stop_reaper.clear()

        def reap() -> None:
            while not self._stop_reaper.wait(interval_ms / 1000.0):
                self.expire_due()

        self._reaper = threading.Thread(
            target=reap, name="mixedmem-lease-reaper", daemon=True
        )
        self._reaper.start()

    def stop_reaper(self) -> None:
        self._stop_reaper.set()
        if self._reaper is not None:
            self._reaper.join(timeout=1.0)
            self._reaper = None

    def __enter__(self) -> "LeasePool":
        self.start_reaper()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop_reaper()
