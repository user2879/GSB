from __future__ import annotations

import heapq
from typing import Any, Iterable

from .canonical import digest


def dependency_map(shards: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        str(shard["shard_id"]): [str(dep) for dep in shard.get("dependencies", [])]
        for shard in shards
    }


def reverse_dependency_map(shards: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    deps = dependency_map(shards)
    reverse: dict[str, set[str]] = {shard_id: set() for shard_id in deps}
    for shard_id, prerequisites in deps.items():
        for prerequisite in prerequisites:
            reverse.setdefault(prerequisite, set()).add(shard_id)
    return reverse


def stable_topological_order(shards: Iterable[dict[str, Any]]) -> list[str]:
    """Kahn order with larger priority first, then shard id ascending.

    The returned order is the explicit stable topology order used by all
    boundary tie-breaks.
    """

    shard_list = list(shards)
    deps = dependency_map(shard_list)
    by_id = {str(shard["shard_id"]): shard for shard in shard_list}
    indegree = {shard_id: len(prerequisites) for shard_id, prerequisites in deps.items()}
    children = reverse_dependency_map(shard_list)

    ready: list[tuple[int, str]] = []
    for shard_id, degree in indegree.items():
        if degree == 0:
            priority = int(by_id[shard_id].get("priority", 0))
            heapq.heappush(ready, (-priority, shard_id))

    order: list[str] = []
    while ready:
        _, shard_id = heapq.heappop(ready)
        order.append(shard_id)
        for child in sorted(children.get(shard_id, set())):
            indegree[child] -= 1
            if indegree[child] == 0:
                priority = int(by_id[child].get("priority", 0))
                heapq.heappush(ready, (-priority, child))

    if len(order) != len(deps):
        remaining = [shard_id for shard_id in deps if shard_id not in set(order)]
        cycle = _find_cycle(deps, remaining)
        error = ValueError(f"dependency graph contains a cycle: {' -> '.join(cycle)}")
        setattr(error, "cycle", cycle)
        raise error
    return order


def _find_cycle(deps: dict[str, list[str]], remaining: list[str]) -> list[str]:
    remaining_set = set(remaining)
    start = remaining[0]
    path: list[str] = []
    seen: set[str] = set()
    current = start
    while current not in seen:
        seen.add(current)
        path.append(current)
        candidates = [dep for dep in deps.get(current, []) if dep in remaining_set]
        if not candidates:
            break
        current = sorted(candidates)[0]
    if current in path:
        index = path.index(current)
        return path[index:] + [current]
    return path


def all_descendants(shard_ids: Iterable[str], reverse: dict[str, set[str]]) -> set[str]:
    result: set[str] = set()
    stack = list(shard_ids)
    while stack:
        current = stack.pop()
        for child in reverse.get(current, set()):
            if child not in result:
                result.add(child)
                stack.append(child)
    return result


def effective_node_id(shard: dict[str, Any], spec: dict[str, Any]) -> str:
    explicit = shard.get("node_id")
    if explicit:
        return str(explicit)
    # Stable placement for declarations without an explicit node reference.
    from .canonical import digest

    nodes = spec["nodes"]
    index = int(digest({"placement": shard["shard_id"]}), 16) % len(nodes)
    return nodes[index]["node_id"]


def propagation_basis(spec: dict[str, Any], shard_id: str) -> dict[str, Any]:
    """Deterministic graph/node provenance used by incremental/full checks."""

    shard = spec["shard_by_id"][shard_id]
    node_id = effective_node_id(shard, spec)
    node = spec["node_by_id"][node_id]
    node_basis = {
        "node_id": node_id,
        "seed": node["seed"],
        "capacity_version": node["capacity_version"],
        "fixed_scale": node["fixed_scale"],
        "group_count": node["group_count"],
    }
    dependency_basis = []
    for dependency in shard["dependencies"]:
        dep = spec["shard_by_id"][dependency]
        dependency_basis.append(
            {
                "shard_id": dependency,
                "rows": dep["rows"],
                "resource_units": dep["resource_units"],
                "priority": dep["priority"],
            }
        )
    basis = {
        "shard_id": shard_id,
        "dependencies": list(shard["dependencies"]),
        "dependency_basis": dependency_basis,
        "node_reference": node_basis,
        "rows": shard["rows"],
        "resource_units": shard["resource_units"],
        "priority": shard["priority"],
    }
    basis["basis_digest"] = digest(basis)
    return basis


def derive_invalidation(
    old_spec: dict[str, Any],
    new_spec: dict[str, Any],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Return invalid shards and stable propagation evidence."""

    old_shards = old_spec["shard_by_id"]
    new_shards = new_spec["shard_by_id"]
    old_nodes = old_spec["node_by_id"]
    new_nodes = new_spec["node_by_id"]
    reverse = reverse_dependency_map(new_spec["shards"])

    direct: dict[str, str] = {}
    scale_changed = old_spec.get("scale_factor") != new_spec.get("scale_factor")
    for shard_id, new_shard in new_shards.items():
        old_shard = old_shards.get(shard_id)
        if old_shard is None:
            direct[shard_id] = "added_shard"
            continue
        changed: list[str] = []
        if scale_changed:
            changed.append("global.scale_factor")
        for field in ("dependencies", "rows", "resource_units", "priority", "node_id", "group_count"):
            if old_shard.get(field) != new_shard.get(field):
                changed.append(field)
        old_node_id = effective_node_id(old_shard, old_spec)
        new_node_id = effective_node_id(new_shard, new_spec)
        if old_node_id != new_node_id:
            changed.append("node_reference")
        else:
            old_node = old_nodes.get(old_node_id, {})
            new_node = new_nodes.get(new_node_id, {})
            for field in ("seed", "capacity_version", "fixed_scale", "group_count"):
                if old_node.get(field) != new_node.get(field):
                    changed.append(f"node.{field}@{new_node_id}")
        if changed:
            direct[shard_id] = "changed:" + ",".join(changed)

    for shard_id in old_shards:
        if shard_id not in new_shards:
            # Deleted shards are reported by the caller's comparison layer;
            # descendants in the new graph are invalid through graph diff.
            for child in reverse.get(shard_id, set()):
                direct.setdefault(child, f"deleted_dependency:{shard_id}")

    invalid = set(direct)
    propagated = all_descendants(set(direct), reverse)
    invalid.update(propagated)

    steps: list[dict[str, Any]] = []
    topo = new_spec["topo_order"]
    for shard_id in topo:
        if shard_id in direct:
            steps.append(
                {
                    "shard_id": shard_id,
                    "kind": "direct",
                    "basis": direct[shard_id],
                    "path": [shard_id],
                }
            )
        elif shard_id in invalid:
            causes = [
                dep
                for dep in new_shards[shard_id]["dependencies"]
                if dep in invalid
            ]
            cause = sorted(causes, key=lambda item: new_spec["topo_index"].get(item, 10**9))[0]
            steps.append(
                {
                    "shard_id": shard_id,
                    "kind": "propagated",
                    "basis": f"upstream_invalid:{cause}",
                    "path": [cause, shard_id],
                }
            )
    return invalid, steps
