"""Shared helpers: canonical serialization, hashing, monotonic clock."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding used for byte-identical artifacts and digests.

    Sort keys, no whitespace surprises, preserve non-ASCII as explicit escapes.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def now_ms() -> int:
    """Monotonic clock in integer milliseconds (queueing / scheduling ledger)."""
    return int(time.monotonic() * 1000)


def wall_ms() -> int:
    """Wall clock in integer milliseconds (structured log timestamps)."""
    return int(time.time() * 1000)


def stable_topological(nodes: set[str], edges: dict[str, set[str]]) -> list[str]:
    """Kahn topo sort with lexicographic tie breaking.

    ``edges[x]`` contains the direct dependencies (predecessors) of ``x``.
    Raises ValueError with the remaining cycle nodes when a cycle exists.
    """
    indegree = {n: 0 for n in nodes}
    dependents: dict[str, list[str]] = {n: [] for n in nodes}
    for node in nodes:
        for dep in edges.get(node, ()):
            if dep in nodes:
                indegree[node] += 1
                dependents[dep].append(node)
    ready = sorted(n for n, deg in indegree.items() if deg == 0)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for dependent in sorted(dependents[current]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort()
    if len(order) != len(nodes):
        remaining = sorted(n for n, deg in indegree.items() if deg > 0)
        raise ValueError(f"cycle detected among: {remaining}")
    return order
