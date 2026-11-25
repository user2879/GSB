from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .spec import TableRule


@dataclass(frozen=True, slots=True)
class BoundaryDiff:
    index: int | None
    old: int | None
    new: int | None
    offset: int


def first_boundary_diff(previous: TableRule | None, current: TableRule) -> BoundaryDiff:
    if previous is None:
        return BoundaryDiff(None, None, None, 0)
    old_boundaries = previous.boundaries
    new_boundaries = current.boundaries
    limit = min(len(old_boundaries), len(new_boundaries))
    for index in range(limit):
        if old_boundaries[index] != new_boundaries[index]:
            return BoundaryDiff(
                index,
                old_boundaries[index],
                new_boundaries[index],
                new_boundaries[index] - old_boundaries[index],
            )
    if len(old_boundaries) != len(new_boundaries):
        index = limit
        old_value = old_boundaries[index] if index < len(old_boundaries) else None
        new_value = new_boundaries[index] if index < len(new_boundaries) else None
        offset = (new_value - old_value) if old_value is not None and new_value is not None else 0
        return BoundaryDiff(index, old_value, new_value, offset)
    return BoundaryDiff(None, None, None, 0)


def compare_route(
    *,
    event_seq: int,
    db_id: str,
    table: str,
    key_column: str,
    route_key: int,
    previous_rule: TableRule | None,
    current_rule: TableRule,
) -> dict[str, Any]:
    current_segment = current_rule.segment_for_key(route_key)
    previous_segment = previous_rule.segment_for_key(route_key) if previous_rule else None
    current_shard = current_segment.shard_id if current_segment else None
    previous_shard = previous_segment.shard_id if previous_segment else None
    diff = first_boundary_diff(previous_rule, current_rule)

    if previous_rule is None:
        changed = False
        direction = "none"
        reason = f"table {table!r} has no previous route version"
    elif previous_segment is None and current_segment is not None:
        changed = True
        direction = "reroute"
        reason = (
            f"key {route_key} was outside the version {previous_rule.route_version} range and now routes "
            f"to {current_shard} under version {current_rule.route_version}"
        )
    elif previous_segment is not None and current_segment is None:
        changed = True
        direction = "reroute"
        reason = (
            f"key {route_key} routed to {previous_shard} under version {previous_rule.route_version} "
            f"and is outside the version {current_rule.route_version} range"
        )
    elif previous_shard != current_shard:
        old_index = previous_rule.segments.index(previous_segment) if previous_segment else -1
        new_index = current_rule.segments.index(current_segment) if current_segment else -1
        if new_index > old_index:
            direction = "right"
        elif new_index < old_index:
            direction = "left"
        else:
            direction = "reroute"
        if diff.index is None:
            boundary_note = "the shard identities changed without a scalar boundary offset"
        else:
            boundary_note = (
                f"boundary[{diff.index}] moved {diff.old}->{diff.new} "
                f"(offset {diff.offset:+d})"
            )
        changed = True
        reason = (
            f"key {route_key} moved {previous_shard}->{current_shard} under route table "
            f"{previous_rule.route_version}->{current_rule.route_version}; {boundary_note}"
        )
    else:
        changed = False
        direction = "none"
        reason = (
            f"key {route_key} remains in {current_shard} under versions "
            f"{previous_rule.route_version} and {current_rule.route_version}"
        )

    return {
        "event_seq": event_seq,
        "db_id": db_id,
        "table": table,
        "key_column": key_column,
        "route_key": route_key,
        "old_route_version": previous_rule.route_version if previous_rule else None,
        "new_route_version": current_rule.route_version,
        "old_shard": previous_shard,
        "new_shard": current_shard,
        "changed": changed,
        "drift_direction": direction,
        "boundary_index": diff.index,
        "boundary_old": diff.old,
        "boundary_new": diff.new,
        "boundary_offset": diff.offset,
        "reason": reason,
    }


def generate_sample_keys(
    *,
    seed: int,
    event_seq: int,
    route_key: int,
    rule: TableRule,
    sample_count: int,
) -> tuple[int, ...]:
    """Generate deterministic key samples, always including the event key.

    ``numpy.PCG64`` is used so the same integer seed and event sequence produce
    the same samples in a database child and in a parent-side prediction for
    quota reservation.  The sample range is the full declared rule range.
    """

    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")
    if sample_count == 1:
        return (route_key,)
    seed_sequence = np.random.SeedSequence(
        [seed, event_seq & 0xFFFFFFFF, route_key & 0xFFFFFFFF]
    )
    generator = np.random.Generator(np.random.PCG64(seed_sequence))
    low = rule.key_min
    high_exclusive = rule.key_max
    samples: list[int] = [route_key]
    low_64 = int(np.iinfo(np.int64).min)
    high_64 = int(np.iinfo(np.int64).max)
    width = high_exclusive - low
    if low >= low_64 and high_exclusive <= high_64 and width <= high_64:
        for raw_value in generator.integers(low, high_exclusive, size=sample_count - 1, dtype=np.int64):
            samples.append(int(raw_value))
    else:
        # Python's JSON integers are unbounded.  Use a 63-bit deterministic
        # offset and modulo projection for ranges outside numpy's int64 domain.
        for raw_value in generator.integers(
            0,
            high_64,
            size=sample_count - 1,
            dtype=np.int64,
        ):
            samples.append(low + (int(raw_value) % width))
    return tuple(samples)


def sample_hit_shards(rule: TableRule, keys: tuple[int, ...]) -> tuple[str, ...]:
    hits = {rule.shard_for_key(key) for key in keys}
    return tuple(sorted(hits))
