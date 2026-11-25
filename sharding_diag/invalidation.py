"""Invalidation propagation after capacity/quota/demand evolution.

Three trigger kinds are recognised:

* ``capacity_version`` - a node's capacity version advanced; every shard ever
  assigned to that node is directly invalidated, then all transitive
  dependents follow the dependency graph.
* ``quota`` - a node's concurrency or lease limit changed; the node's assigned
  shards are directly invalidated because scheduling ownership is rewritten,
  then dependents propagate.
* ``demand`` - a shard's resource demand changed; that shard is directly
  invalidated and its dependents propagate.

Output is a stable topologically ordered invalidation set with an explicit
``basis`` step list, so a reviewer can replay each propagation hop.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .model import Declaration, topological_order


@dataclass
class Change:
    kind: str            # capacity_version | quota | demand
    object_id: str       # node id or shard id
    field: str
    old_value: int
    new_value: int

    def as_dict(self):
        return {
            "kind": self.kind,
            "object_id": self.object_id,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }


def diff_declarations(old: Declaration, new: Declaration) -> List[Change]:
    changes: List[Change] = []
    old_nodes = old.node_map()
    new_nodes = new.node_map()
    for node_id in sorted(set(old_nodes) | set(new_nodes)):
        before, after = old_nodes.get(node_id), new_nodes.get(node_id)
        if before is None or after is None:
            changes.append(Change("quota", node_id, "node_presence",
                                  0 if before is None else 1,
                                  1 if after is None else 0))
            continue
        for fld in ("capacity_version", "concurrency", "lease_limit"):
            old_value, new_value = getattr(before, fld), getattr(after, fld)
            if old_value != new_value:
                kind = "capacity_version" if fld == "capacity_version" else "quota"
                changes.append(Change(kind, node_id, fld, old_value, new_value))
    old_shards, new_shards = old.shard_map(), new.shard_map()
    for shard_id in sorted(set(old_shards) | set(new_shards)):
        before, after = old_shards.get(shard_id), new_shards.get(shard_id)
        if before is None or after is None:
            changes.append(Change("demand", shard_id, "shard_presence",
                                  0 if before is None else 1,
                                  1 if after is None else 0))
            continue
        if before.demand != after.demand:
            changes.append(Change("demand", shard_id, "demand", before.demand, after.demand))
    return changes


def _reverse_dependencies(declaration: Declaration) -> Dict[str, List[str]]:
    dependents: Dict[str, List[str]] = {s.shard_id: [] for s in declaration.shards}
    for shard in declaration.shards:
        for dep in shard.depends_on:
            dependents[dep].append(shard.shard_id)
    return {sid: sorted(children) for sid, children in dependents.items()}


def propagate(
    declaration: Declaration,
    changes: List[Change],
    assignments: Dict[str, str],
) -> dict:
    """Compute the invalidation set and per-step propagation basis.

    ``assignments`` maps shard_id -> node_id from committed ledger ownership.
    """
    dependents = _reverse_dependencies(declaration)
    directly_invalid: Set[str] = set()
    basis: List[dict] = []

    node_to_shards: Dict[str, List[str]] = {}
    for shard_id, node_id in assignments.items():
        node_to_shards.setdefault(node_id, []).append(shard_id)

    for change in sorted(changes, key=lambda c: (c.kind, c.object_id, c.field)):
        if change.kind in ("capacity_version", "quota"):
            targets = sorted(node_to_shards.get(change.object_id, []))
            for shard_id in targets:
                if shard_id not in directly_invalid:
                    directly_invalid.add(shard_id)
                    basis.append({
                        "step": "direct",
                        "shard_id": shard_id,
                        "cause": change.kind,
                        "change": change.as_dict(),
                    })
        elif change.kind == "demand":
            if change.object_id in dependents and change.object_id not in directly_invalid:
                directly_invalid.add(change.object_id)
                basis.append({
                    "step": "direct",
                    "shard_id": change.object_id,
                    "cause": "demand",
                    "change": change.as_dict(),
                })

    invalidated: Set[str] = set(directly_invalid)
    frontier = sorted(directly_invalid)
    while frontier:
        current = frontier.pop(0)
        for child in dependents.get(current, []):
            if child not in invalidated:
                invalidated.add(child)
                frontier.append(child)
                basis.append({
                    "step": "propagate",
                    "shard_id": child,
                    "cause": "downstream_dependency",
                    "via": current,
                })
        frontier.sort()

    topo = topological_order(declaration.shards)
    ordered = [sid for sid in topo if sid in invalidated]
    untouched = sorted(sid for sid in topo if sid not in invalidated)
    return {
        "invalidated": ordered,
        "directly_invalidated": sorted(directly_invalid),
        "untouched": untouched,
        "basis": basis,
        "topological_order": topo,
    }
