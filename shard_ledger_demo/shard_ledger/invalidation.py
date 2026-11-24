"""Stable invalidation propagation across nodes and the shard DAG."""

from .spec import topo_sort


def propagate(manifest, completed_shards, revision) -> tuple:
    """Return (invalidated shards in stable topological order, propagation basis).

    Direct invalidation: completed shards that have a target node listed in the
    revision's affects set. Indirect invalidation: every transitive dependent
    of a directly invalidated shard, walking DAG edges in deterministic order.
    """
    dependents = manifest.dependents()
    direct = set()
    basis = []
    for shard_id in sorted(completed_shards):
        affected_targets = sorted(set(manifest.shards[shard_id].targets) & set(revision.affects))
        if affected_targets:
            direct.add(shard_id)
            basis.append(
                {
                    "step": "direct",
                    "shard_id": shard_id,
                    "from_revision": revision.version,
                    "nodes": affected_targets,
                    "reason": "target node fixed-point parameters changed",
                }
            )

    invalidated = set(direct)
    frontier = sorted(direct)
    while frontier:
        current = frontier.pop(0)
        for child in dependents.get(current, ()):
            if child not in invalidated:
                invalidated.add(child)
                frontier.append(child)
                basis.append(
                    {
                        "step": "dependency",
                        "shard_id": child,
                        "from_revision": revision.version,
                        "parent": current,
                        "reason": "upstream shard summary was invalidated",
                    }
                )
        frontier.sort()

    topo = topo_sort(manifest.shards)
    ordered = [sid for sid in topo if sid in invalidated]
    return ordered, basis
