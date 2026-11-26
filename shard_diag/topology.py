from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from .errors import SpecError
from .models import ShardSpec


def stable_topological_order(shards: Sequence[ShardSpec] | Mapping[str, ShardSpec]) -> list[str]:
    """Kahn order with a deterministic ``(-priority, shard_id)`` ready heap."""

    if isinstance(shards, Mapping):
        by_id = dict(shards)
    else:
        by_id = {shard.shard_id: shard for shard in shards}

    indegree = {shard_id: 0 for shard_id in by_id}
    dependents: dict[str, list[str]] = defaultdict(list)
    for shard_id, shard in by_id.items():
        for dependency in shard.dependencies:
            if dependency not in by_id:
                raise SpecError(
                    f"dependency {dependency!r} does not exist",
                    object=f"$.shards[{shard_id}]",
                    field="dependencies",
                )
            indegree[shard_id] += 1
            dependents[dependency].append(shard_id)

    ready: list[tuple[int, str]] = []
    for shard_id, shard in by_id.items():
        if indegree[shard_id] == 0:
            heapq.heappush(ready, (-shard.priority, shard_id))

    order: list[str] = []
    while ready:
        _, shard_id = heapq.heappop(ready)
        order.append(shard_id)
        for dependent in sorted(dependents[shard_id]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, (-by_id[dependent].priority, dependent))

    if len(order) != len(by_id):
        remaining = sorted(shard_id for shard_id, degree in indegree.items() if degree > 0)
        first = remaining[0]
        raise SpecError(
            "dependency graph contains a cycle",
            object=f"$.shards[{first}]",
            field="dependencies",
            details={"remaining": remaining},
        )
    return order


def reverse_dependency_closure(
    shards: Mapping[str, ShardSpec], direct: Iterable[str]
) -> tuple[set[str], list[dict[str, str]]]:
    """Return direct ids plus all transitive dependents and ordered reasons."""

    dependents: dict[str, list[str]] = defaultdict(list)
    for shard in shards.values():
        for dependency in shard.dependencies:
            dependents[dependency].append(shard.shard_id)

    invalid: set[str] = set()
    steps: list[dict[str, str]] = []
    queue = list(dict.fromkeys(direct))
    for shard_id in queue:
        if shard_id not in shards:
            raise KeyError(shard_id)
        if shard_id not in invalid:
            invalid.add(shard_id)
            steps.append({"shard_id": shard_id, "kind": "direct", "basis": "declared change reference"})
        for dependent in sorted(dependents.get(shard_id, ())):
            if dependent not in invalid:
                invalid.add(dependent)
                steps.append(
                    {
                        "shard_id": dependent,
                        "kind": "propagated",
                        "basis": f"dependency:{shard_id}->{dependent}",
                    }
                )
                queue.append(dependent)
    return invalid, steps
