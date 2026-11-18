from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


class PoolOOM(RuntimeError):
    def __init__(self, event: dict[str, Any]):
        super().__init__(
            "OOM: need "
            f"{event.get('requested_blocks')} blocks, free {event.get('free_blocks_after')}"
        )
        self.event = event
        self.required_blocks = int(event.get("requested_blocks", 0))
        self.free_blocks = int(event.get("free_blocks_after", 0))
        self.largest_free_extent = int(event.get("largest_free_extent_after", 0))
        self.batch_id = event.get("batch_id")
        self.stage = event.get("stage")


@dataclass
class Allocation:
    allocation_id: str
    owner_id: str
    batch_id: str | None
    purpose: str
    stage: str | None
    blocks: tuple[int, ...]
    request_bytes: int
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "owner_id": self.owner_id,
            "batch_id": self.batch_id,
            "purpose": self.purpose,
            "stage": self.stage,
            "blocks": list(self.blocks),
            "request_bytes": self.request_bytes,
            "strategy": self.strategy,
        }


@dataclass
class PoolState:
    used_blocks: list[int] = field(default_factory=list)
    free_blocks: int = 0
    free_extents: list[tuple[int, int]] = field(default_factory=list)
    largest_free_extent: int = 0
    fragmentation_blocks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_blocks": list(self.used_blocks),
            "free_blocks": self.free_blocks,
            "free_extents": [[start, end] for start, end in self.free_extents],
            "largest_free_extent": self.largest_free_extent,
            "fragmentation_blocks": self.fragmentation_blocks,
        }


EventCallback = Callable[[dict[str, Any]], None]


class BlockPool:
    """Deterministic block allocator with a replayable allocation event log."""

    def __init__(
        self,
        total_blocks: int,
        block_size_bytes: int,
        *,
        origin_ns: int | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        on_event: EventCallback | None = None,
    ) -> None:
        if total_blocks < 1:
            raise ValueError("total_blocks must be positive")
        if block_size_bytes < 1:
            raise ValueError("block_size_bytes must be positive")
        self.total_blocks = total_blocks
        self.block_size_bytes = block_size_bytes
        self.origin_ns = origin_ns if origin_ns is not None else clock_ns()
        self.clock_ns = clock_ns
        self.on_event = on_event

        self._free_extents: list[list[int]] = [[0, total_blocks]]
        self._block_owner: dict[int, str] = {}
        self._allocations: dict[str, Allocation] = {}
        self._owner_allocations: dict[str, list[str]] = {}
        self.events: list[dict[str, Any]] = []
        self.peak_used_blocks = 0
        self.peak_fragmentation_blocks = 0
        self._allocation_counter = 0
        self._event_counter = 0

    @property
    def used_blocks(self) -> int:
        return len(self._block_owner)

    @property
    def free_blocks(self) -> int:
        return self.total_blocks - self.used_blocks

    def largest_free_extent(self) -> int:
        return max((end - start for start, end in self._free_extents), default=0)

    def fragmentation_blocks(self) -> int:
        free = self.free_blocks
        if free == 0:
            return 0
        return free - self.largest_free_extent()

    def snapshot(self) -> PoolState:
        free = self.free_blocks
        largest = self.largest_free_extent()
        return PoolState(
            used_blocks=sorted(self._block_owner),
            free_blocks=free,
            free_extents=[(start, end) for start, end in self._free_extents],
            largest_free_extent=largest,
            fragmentation_blocks=free - largest if free else 0,
        )

    def blocks_for_bytes(self, request_bytes: int) -> int:
        if request_bytes <= 0:
            return 0
        return math.ceil(request_bytes / self.block_size_bytes)

    def _next_allocation_id(self) -> str:
        self._allocation_counter += 1
        return f"A{self._allocation_counter:06d}"

    def _elapsed_ms(self, now_ns: int) -> float:
        return (now_ns - self.origin_ns) / 1_000_000.0

    def _emit(
        self,
        *,
        action: str,
        owner_id: str | None,
        allocation_id: str | None,
        batch_id: str | None,
        purpose: str,
        stage: str | None,
        requested_bytes: int,
        requested_blocks: int,
        blocks: Iterable[int] = (),
        strategy: str | None = None,
        message: str = "",
        now_ns: int | None = None,
    ) -> dict[str, Any]:
        timestamp = self.clock_ns() if now_ns is None else now_ns
        state = self.snapshot()
        self._event_counter += 1
        event = {
            "seq": self._event_counter,
            "action": action,
            "owner_id": owner_id,
            "allocation_id": allocation_id,
            "batch_id": batch_id,
            "purpose": purpose,
            "stage": stage,
            "strategy": strategy,
            "requested_bytes": requested_bytes,
            "requested_blocks": requested_blocks,
            "blocks": sorted(int(block) for block in blocks),
            "used_blocks_after": len(state.used_blocks),
            "used_block_ids_after": state.used_blocks,
            "free_blocks_after": state.free_blocks,
            "free_extents_after": [[start, end] for start, end in state.free_extents],
            "largest_free_extent_after": state.largest_free_extent,
            "fragmentation_blocks_after": state.fragmentation_blocks,
            "peak_used_blocks_after": self.peak_used_blocks,
            "t_mono_ns": timestamp,
            "elapsed_ms": self._elapsed_ms(timestamp),
            "message": message,
        }
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)
        return event

    def _take_blocks(self, count: int, *, contiguous: bool) -> list[int]:
        if count == 0:
            return []
        if self.free_blocks < count:
            raise ValueError("insufficient free blocks")
        if contiguous:
            for index, (start, end) in enumerate(self._free_extents):
                if end - start >= count:
                    blocks = list(range(start, start + count))
                    if start + count == end:
                        self._free_extents.pop(index)
                    else:
                        self._free_extents[index][0] = start + count
                    return blocks
            raise ValueError("no contiguous free extent")

        remaining = count
        blocks: list[int] = []
        for extent_index in range(len(self._free_extents) - 1, -1, -1):
            start, end = self._free_extents[extent_index]
            take = min(remaining, end - start)
            run = list(range(start, start + take))
            blocks.extend(run)
            if take == end - start:
                self._free_extents.pop(extent_index)
            else:
                self._free_extents[extent_index][0] = start + take
            remaining -= take
            if remaining == 0:
                break
        return sorted(blocks)

    def allocate(
        self,
        *,
        owner_id: str,
        batch_id: str | None,
        purpose: str,
        request_bytes: int,
        strategy: str,
        contiguous: bool,
        stage: str | None = None,
        requested_blocks: int | None = None,
    ) -> Allocation:
        required = self.blocks_for_bytes(request_bytes) if requested_blocks is None else requested_blocks
        if required < 0:
            raise ValueError("requested_blocks cannot be negative")
        if required == 0:
            allocation = Allocation(
                allocation_id=self._next_allocation_id(),
                owner_id=owner_id,
                batch_id=batch_id,
                purpose=purpose,
                stage=stage,
                blocks=(),
                request_bytes=request_bytes,
                strategy=strategy,
            )
            self._allocations[allocation.allocation_id] = allocation
            self._owner_allocations.setdefault(owner_id, []).append(allocation.allocation_id)
            return allocation

        possible = self.free_blocks >= required and (not contiguous or self.largest_free_extent() >= required)
        if not possible:
            event = self._emit(
                action="oom",
                owner_id=owner_id,
                allocation_id=None,
                batch_id=batch_id,
                purpose=purpose,
                stage=stage,
                strategy=strategy,
                requested_bytes=request_bytes,
                requested_blocks=required,
                message="allocation rejected by block budget or contiguous-fit rule",
            )
            raise PoolOOM(event)

        blocks = self._take_blocks(required, contiguous=contiguous)
        allocation_id = self._next_allocation_id()
        for block in blocks:
            if block in self._block_owner:
                raise RuntimeError(f"allocator invariant violated: block {block} already owned")
            self._block_owner[block] = allocation_id
        allocation = Allocation(
            allocation_id=allocation_id,
            owner_id=owner_id,
            batch_id=batch_id,
            purpose=purpose,
            stage=stage,
            blocks=tuple(blocks),
            request_bytes=request_bytes,
            strategy=strategy,
        )
        self._allocations[allocation_id] = allocation
        self._owner_allocations.setdefault(owner_id, []).append(allocation_id)
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_blocks)
        self.peak_fragmentation_blocks = max(self.peak_fragmentation_blocks, self.fragmentation_blocks())
        self._emit(
            action="allocate",
            owner_id=owner_id,
            allocation_id=allocation_id,
            batch_id=batch_id,
            purpose=purpose,
            stage=stage,
            strategy=strategy,
            requested_bytes=request_bytes,
            requested_blocks=required,
            blocks=blocks,
        )
        return allocation

    def allocate_single_block(
        self,
        *,
        owner_id: str,
        batch_id: str | None,
        purpose: str,
        request_bytes: int,
        strategy: str = "single",
        stage: str | None = None,
    ) -> Allocation:
        return self.allocate(
            owner_id=owner_id,
            batch_id=batch_id,
            purpose=purpose,
            request_bytes=request_bytes,
            strategy=strategy,
            contiguous=False,
            stage=stage,
            requested_blocks=1,
        )

    def release(self, owner_id: str, *, reason: str = "released") -> list[dict[str, Any]]:
        allocation_ids = self._owner_allocations.pop(owner_id, [])
        released_events: list[dict[str, Any]] = []
        for allocation_id in allocation_ids:
            allocation = self._allocations.pop(allocation_id, None)
            if allocation is None:
                continue
            if not allocation.blocks:
                continue
            for block in allocation.blocks:
                if self._block_owner.get(block) != allocation_id:
                    raise RuntimeError(f"allocator invariant violated while releasing block {block}")
                del self._block_owner[block]
            for start, end in _contiguous_runs(allocation.blocks):
                self._free_extents.append([start, end + 1])
            self._coalesce()
            event = self._emit(
                action="release",
                owner_id=allocation.owner_id,
                allocation_id=allocation.allocation_id,
                batch_id=allocation.batch_id,
                purpose=allocation.purpose,
                stage=allocation.stage,
                strategy=allocation.strategy,
                requested_bytes=allocation.request_bytes,
                requested_blocks=len(allocation.blocks),
                blocks=allocation.blocks,
                message=reason,
            )
            released_events.append(event)
        return released_events

    def release_batch(self, batch_id: str, *, reason: str = "batch complete") -> list[dict[str, Any]]:
        owner_ids = sorted(
            {
                allocation.owner_id
                for allocation in self._allocations.values()
                if allocation.batch_id == batch_id
            }
        )
        events: list[dict[str, Any]] = []
        for owner_id in owner_ids:
            events.extend(self.release(owner_id, reason=reason))
        return events

    def active_allocations(self) -> list[dict[str, Any]]:
        return [
            allocation.to_dict()
            for allocation in sorted(self._allocations.values(), key=lambda item: item.allocation_id)
        ]

    def note(
        self,
        *,
        batch_id: str | None,
        purpose: str,
        message: str,
        stage: str | None = None,
    ) -> dict[str, Any]:
        return self._emit(
            action="note",
            owner_id=None,
            allocation_id=None,
            batch_id=batch_id,
            purpose=purpose,
            stage=stage,
            requested_bytes=0,
            requested_blocks=0,
            message=message,
        )

    def _coalesce(self) -> None:
        self._free_extents.sort()
        merged: list[list[int]] = []
        for start, end in self._free_extents:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self._free_extents = merged

    def final_state(self) -> dict[str, Any]:
        state = self.snapshot()
        return {
            **state.to_dict(),
            "total_blocks": self.total_blocks,
            "block_size_bytes": self.block_size_bytes,
            "used_blocks_count": self.used_blocks,
            "peak_used_blocks": self.peak_used_blocks,
            "peak_fragmentation_blocks": self.peak_fragmentation_blocks,
            "active_allocations": self.active_allocations(),
            "events_count": len(self.events),
        }


def replay_events(events: list[dict[str, Any]], total_blocks: int) -> dict[str, Any]:
    """Replay an allocator log and verify ownership plus every recorded state."""

    used: set[int] = set()
    allocation_owner: dict[str, set[int]] = {}
    peak = 0
    errors: list[str] = []
    for expected_seq, event in enumerate(events, start=1):
        sequence = int(event.get("seq", -1))
        if sequence != expected_seq:
            errors.append(f"event {expected_seq}: sequence mismatch {sequence}")
        action = event.get("action")
        blocks = [int(block) for block in event.get("blocks", [])]
        if action == "allocate":
            allocation_id = event.get("allocation_id")
            if not allocation_id:
                errors.append(f"event {sequence}: allocate without allocation_id")
            for block in blocks:
                if not 0 <= block < total_blocks:
                    errors.append(f"event {sequence}: block {block} outside [0,{total_blocks})")
                if block in used:
                    errors.append(f"event {sequence}: block {block} allocated twice")
                used.add(block)
            if allocation_id:
                if allocation_id in allocation_owner:
                    errors.append(f"event {sequence}: duplicate allocation {allocation_id}")
                allocation_owner[allocation_id] = set(blocks)
        elif action == "release":
            allocation_id = event.get("allocation_id")
            owned = allocation_owner.pop(allocation_id, None) if allocation_id else None
            if owned is None:
                errors.append(f"event {sequence}: release of unknown allocation {allocation_id}")
            else:
                if owned != set(blocks):
                    errors.append(f"event {sequence}: release blocks do not match allocation")
                if not set(blocks).issubset(used):
                    errors.append(f"event {sequence}: release contains unallocated blocks")
                used.difference_update(blocks)
        elif action in {"oom", "note"}:
            pass
        else:
            errors.append(f"event {sequence}: unknown action {action!r}")

        actual_used = len(used)
        actual_free = total_blocks - actual_used
        peak = max(peak, actual_used)
        if int(event.get("used_blocks_after", -1)) != actual_used:
            errors.append(
                f"event {sequence}: used_blocks_after {event.get('used_blocks_after')} != {actual_used}"
            )
        if int(event.get("free_blocks_after", -1)) != actual_free:
            errors.append(
                f"event {sequence}: free_blocks_after {event.get('free_blocks_after')} != {actual_free}"
            )
        actual_extents = _free_extents_from_used(sorted(used), total_blocks)
        recorded_extents = [tuple(int(value) for value in extent) for extent in event.get("free_extents_after", [])]
        if actual_extents != recorded_extents:
            errors.append(
                f"event {sequence}: free_extents_after {recorded_extents} != {actual_extents}"
            )
        actual_largest = max((end - start for start, end in actual_extents), default=0)
        if int(event.get("largest_free_extent_after", -1)) != actual_largest:
            errors.append(
                f"event {sequence}: largest_free_extent_after {event.get('largest_free_extent_after')} != {actual_largest}"
            )
        actual_fragmented = actual_free - actual_largest if actual_free else 0
        if int(event.get("fragmentation_blocks_after", -1)) != actual_fragmented:
            errors.append(
                f"event {sequence}: fragmentation_blocks_after {event.get('fragmentation_blocks_after')} != {actual_fragmented}"
            )

    final_used = sorted(used)
    final_free_extents = _free_extents_from_used(final_used, total_blocks)
    final_largest = max((end - start for start, end in final_free_extents), default=0)
    final_fragmented = (total_blocks - len(used) - final_largest) if used else 0
    return {
        "verified": not errors,
        "errors": errors,
        "replayed_events": len(events),
        "peak_used_blocks": peak,
        "final_used_blocks": final_used,
        "final_free_blocks": total_blocks - len(used),
        "final_free_extents": [[start, end] for start, end in final_free_extents],
        "final_largest_free_extent": final_largest,
        "final_fragmentation_blocks": final_fragmented,
    }


def _free_extents_from_used(used: list[int], total_blocks: int) -> list[tuple[int, int]]:
    extents: list[tuple[int, int]] = []
    start: int | None = None
    used_set = set(used)
    for block in range(total_blocks + 1):
        is_free = block < total_blocks and block not in used_set
        if is_free and start is None:
            start = block
        elif not is_free and start is not None:
            extents.append((start, block))
            start = None
    return extents


def _contiguous_runs(blocks: Iterable[int]) -> list[tuple[int, int]]:
    sorted_blocks = sorted(int(block) for block in blocks)
    if not sorted_blocks:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = sorted_blocks[0]
    for block in sorted_blocks[1:]:
        if block == previous + 1:
            previous = block
            continue
        runs.append((start, previous))
        start = previous = block
    runs.append((start, previous))
    return runs
