from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from .errors import cycle_error


def _adjacency(refs_by_key: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {key: [] for key in refs_by_key}
    for key in sorted(refs_by_key):
        for parent in refs_by_key[key]:
            children.setdefault(parent, []).append(key)
    for parent in children:
        children[parent] = sorted(set(children[parent]))
    return children


def _find_cycle(refs_by_key: Mapping[str, Sequence[str]], remaining: Iterable[str]) -> list[str]:
    remaining_set = set(remaining)
    state: dict[str, int] = {key: 0 for key in remaining_set}
    stack: list[str] = []
    stack_index: dict[str, int] = {}

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack_index[node] = len(stack)
        stack.append(node)
        for parent in sorted(refs_by_key[node]):
            if parent not in remaining_set:
                continue
            if state[parent] == 0:
                found = visit(parent)
                if found:
                    return found
            elif state[parent] == 1:
                return stack[stack_index[parent] :] + [parent]
        state[node] = 2
        stack.pop()
        stack_index.pop(node, None)
        return None

    for node in sorted(remaining_set):
        if state[node] == 0:
            found = visit(node)
            if found:
                return found
    return sorted(remaining_set)


def validate_acyclic(refs_by_key: Mapping[str, Sequence[str]]) -> None:
    """Validate all references and reject cycles with a stable first key.

    Kahn's algorithm is used for normal validation; the DFS fallback runs only
    on the deterministic residual graph and is used to produce a printable
    cycle.  Both the reported key and the cycle path are independent of dict
    insertion and hash traversal order.
    """

    for key in sorted(refs_by_key):
        for ref in refs_by_key[key]:
            if ref not in refs_by_key:
                raise cycle_error(
                    f"派生引用目标不存在: {ref}",
                    key=key,
                    cycle=[key, ref],
                )

    indegree = {key: len(set(refs_by_key[key])) for key in refs_by_key}
    children = _adjacency(refs_by_key)
    ready = [key for key, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    visited: list[str] = []
    while ready:
        node = heapq.heappop(ready)
        visited.append(node)
        for child in children.get(node, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)

    if len(visited) == len(refs_by_key):
        return

    remaining = [key for key in refs_by_key if indegree[key] > 0]
    cycle = _find_cycle(refs_by_key, remaining)
    if cycle and cycle[0] == cycle[-1] and len(cycle) > 1:
        body = cycle[:-1]
    else:
        body = cycle
    if body:
        start = min(body)
        index = body.index(start)
        body = body[index:] + body[:index]
        cycle = body + [start]
    first_key = body[0] if body else (sorted(remaining)[0] if remaining else None)
    raise cycle_error(
        "派生依赖图存在环",
        key=first_key,
        cycle=cycle,
    )


def topological_order(
    refs_by_key: Mapping[str, Sequence[str]],
    subset: Iterable[str] | None = None,
) -> list[str]:
    selected = set(refs_by_key if subset is None else subset)
    indegree = {
        key: sum(1 for ref in set(refs_by_key[key]) if ref in selected)
        for key in selected
    }
    children: dict[str, list[str]] = defaultdict(list)
    for key in selected:
        for ref in set(refs_by_key[key]):
            if ref in selected:
                children[ref].append(key)
    for ref in children:
        children[ref] = sorted(set(children[ref]))
    ready = [key for key, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        key = heapq.heappop(ready)
        ordered.append(key)
        for child in children.get(key, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(ordered) != len(selected):
        validate_acyclic(refs_by_key)
        raise cycle_error("无法生成稳定拓扑序", key=min(selected), cycle=[])
    return ordered


def descendants(changed: Iterable[str], refs_by_key: Mapping[str, Sequence[str]]) -> list[str]:
    children = _adjacency(refs_by_key)
    seen: set[str] = set()
    frontier = sorted(set(changed))
    for key in frontier:
        seen.add(key)
    while frontier:
        current = heapq.heappop(frontier)
        for child in children.get(current, []):
            if child not in seen:
                seen.add(child)
                heapq.heappush(frontier, child)
    return topological_order(refs_by_key, seen)


def build_propagation(
    refs_by_key: Mapping[str, Sequence[str]],
    changed: Iterable[str],
    direct_fields: Mapping[str, Sequence[str]] | None = None,
    invalid_status: str = "recomputed",
) -> tuple[list[str], list[dict[str, object]]]:
    """Return affected keys in stable topological order and per-step evidence."""

    changed_set = set(changed)
    affected = descendants(changed_set, refs_by_key)
    direct_fields = direct_fields or {}
    propagation: list[dict[str, object]] = []
    for index, key in enumerate(affected, start=1):
        if key in changed_set:
            propagation.append(
                {
                    "step": index,
                    "key": key,
                    "basis": "direct",
                    "via": None,
                    "edge": None,
                    "fields": sorted(direct_fields.get(key, ())),
                    "action": invalid_status,
                }
            )
            continue
        candidates = sorted(set(refs_by_key[key]) & changed_set)
        if not candidates:
            candidates = sorted(set(refs_by_key[key]) & set(affected[: index - 1]))
        via = candidates[0] if candidates else None
        propagation.append(
            {
                "step": index,
                "key": key,
                "basis": "dependency",
                "via": via,
                "edge": f"{via}->{key}" if via else None,
                "fields": ["value", "basis", "digest", "version"],
                "action": invalid_status,
            }
        )
    return affected, propagation


def graph_layers(refs_by_key: Mapping[str, Sequence[str]]) -> list[list[str]]:
    """Lay out the dependency graph deterministically for the GUI canvas."""

    order = topological_order(refs_by_key)
    layer: dict[str, int] = {}
    for key in order:
        parents = list(refs_by_key[key])
        layer[key] = 0 if not parents else 1 + max(layer[parent] for parent in parents)
    layers: dict[int, list[str]] = defaultdict(list)
    for key in sorted(refs_by_key):
        layers[layer[key]].append(key)
    return [sorted(layers[index]) for index in sorted(layers)]
