"""Directed dependency graph from source elements to derived digests.

Node ids are stable strings:

  source nodes:
    part:<shard_id>          shard partition rule (tensor -> shard membership)
    param:<tensor>           parameter tensor
    momentum:<tensor>        optimizer momentum tensor
    rng                      RNG state
    cursor                   data cursor
  derived nodes:
    shard:<shard_id>         shard digest
    partition                partition derived digest
    optimizer                optimizer state derived digest
    content                  checkpoint content digest

Edges model "producer -> consumer", so invalidation flows forward along edges.
"""

from .model import tensor_names


def build_graph(partition_mode, layer_dims):
    names = tensor_names(layer_dims)
    if partition_mode == "unit":
        plans = [(f"shard{i:03d}", [name]) for i, name in enumerate(names)]
    else:
        plans = []
        idx = 0
        for layer in range(1, len(layer_dims)):
            plans.append((f"shard{idx:03d}", [f"W{layer}", f"b{layer}"]))
            idx += 1
    shard_ids = [shard_id for shard_id, _ in plans]

    edges = set()

    def add(src, dst):
        edges.add((src, dst))

    for shard_id, members in plans:
        add(f"part:{shard_id}", f"shard:{shard_id}")
        for name in members:
            add(f"param:{name}", f"shard:{shard_id}")
            add(f"momentum:{name}", f"shard:{shard_id}")
        add(f"shard:{shard_id}", "partition")
        add(f"shard:{shard_id}", "content")

    for name in names:
        add(f"momentum:{name}", "optimizer")
    add("optimizer", "content")
    add("partition", "content")
    add("rng", "content")
    add("cursor", "content")

    return {
        "edges": sorted(edges),
        "plans": plans,
        "shard_ids": shard_ids,
        "tensor_names": names,
    }


def adjacency(edges):
    forward = {}
    reverse = {}
    for src, dst in edges:
        forward.setdefault(src, set()).add(dst)
        reverse.setdefault(dst, set()).add(src)
    return forward, reverse


def all_source_nodes(graph):
    nodes = set()
    for src, _dst in graph["edges"]:
        nodes.add(src)
    return sorted(nodes)


def validate_references(graph):
    """Ensure every edge endpoint is a known node (reference completeness)."""
    known = set()
    for src, dst in graph["edges"]:
        known.add(src)
        known.add(dst)
    violations = []
    forward, reverse = adjacency(graph["edges"])
    for node in known:
        if node.startswith(("shard:", "partition", "optimizer", "content")):
            parents = reverse.get(node, set())
            if not parents:
                violations.append(f"node {node} has no producer edges")
    return violations


def propagate(graph, changed_sources):
    """Forward-propagate changed source nodes through the DAG.

    Returns (affected, order, evidence). ``order`` is a stable topological
    ordering; ``evidence`` lists, per newly affected node, the changed producer
    that propagated into it.
    """
    forward, reverse = adjacency(graph["edges"])
    affected = set(changed_sources)
    evidence = {node: ["explicit change set"] for node in changed_sources}

    # Kahn topological order restricted to reachable nodes, with tie-break by id.
    indegree = {}
    reachable = set()
    stack = sorted(affected)
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        for nxt in forward.get(node, ()):
            stack.append(nxt)
    for src, dst in graph["edges"]:
        if src in reachable and dst in reachable:
            indegree[dst] = indegree.get(dst, 0) + 1

    ready = sorted(node for node in reachable if indegree.get(node, 0) == 0)
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for nxt in sorted(forward.get(node, ())):
            if nxt not in reachable:
                continue
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort()

    for node in order:
        if node in affected:
            continue
        producers = sorted(reverse.get(node, set()))
        triggers = [parent for parent in producers if parent in affected]
        if triggers:
            affected.add(node)
            evidence[node] = [
                f"{trigger} -> {node}" for trigger in triggers
            ]

    affected_shards = sorted(
        node.split(":", 1)[1]
        for node in affected
        if node.startswith("shard:")
    )
    return {
        "affected": sorted(affected),
        "order": order,
        "evidence": evidence,
        "affected_shards": affected_shards,
        "partition_affected": "partition" in affected,
        "optimizer_affected": "optimizer" in affected,
        "content_affected": "content" in affected,
    }


def shard_producer_sources(graph, shard_id):
    """Source nodes that feed one shard digest (partition + members)."""
    _forward, reverse = adjacency(graph["edges"])
    producers = sorted(reverse.get(f"shard:{shard_id}", set()))
    return producers
