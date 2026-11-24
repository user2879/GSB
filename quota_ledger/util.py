"""Small deterministic and clock helpers shared by the component."""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def monotonic_ms() -> int:
    """Monotonic wall-independent milliseconds."""
    return time.perf_counter_ns() // 1_000_000


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_pos_int(value: Any) -> bool:
    return is_int(value) and value > 0


def is_nonneg_int(value: Any) -> bool:
    return is_int(value) and value >= 0


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_obj(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + f".tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, target)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, canonical_json(value) + "\n")


def deep_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def stable_topological_order(nodes: Iterable[str], dependencies: Mapping[str, Sequence[str]]) -> list[str]:
    """Lexicographically stable Kahn topological order.

    Ties are resolved by shard id so the result is independent of dict/hash order.
    """
    node_set = set(nodes)
    indegree = {node: 0 for node in node_set}
    children: dict[str, list[str]] = {node: [] for node in node_set}
    for node in sorted(node_set):
        for dep in sorted(set(dependencies.get(node, ()))):
            if dep in node_set:
                indegree[node] += 1
                children[dep].append(node)
    heap = [node for node, count in indegree.items() if count == 0]
    heapq.heapify(heap)
    ordered: list[str] = []
    while heap:
        node = heapq.heappop(heap)
        ordered.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, child)
    if len(ordered) != len(node_set):
        return ordered
    return ordered


def find_cycle(nodes: Iterable[str], dependencies: Mapping[str, Sequence[str]]) -> list[str]:
    """Return a deterministic cycle, or an empty list."""
    node_set = set(nodes)
    color: dict[str, int] = {node: 0 for node in node_set}
    stack: list[str] = []
    stack_pos: dict[str, int] = {}

    def visit(node: str) -> list[str]:
        color[node] = 1
        stack_pos[node] = len(stack)
        stack.append(node)
        for dep in sorted(dependencies.get(node, ())):
            if dep not in node_set:
                continue
            if color[dep] == 0:
                found = visit(dep)
                if found:
                    return found
            elif color[dep] == 1:
                return stack[stack_pos[dep]:] + [dep]
        stack.pop()
        stack_pos.pop(node, None)
        color[node] = 2
        return []

    for node in sorted(node_set):
        if color[node] == 0:
            found = visit(node)
            if found:
                return found
    return []


def deterministic_samples(node_seed: int, shard_seed_offset: int, count: int, groups: int) -> list[list[int]]:
    """Create fixed integer (group, value) samples without external downloads."""
    # Linear congruential arithmetic is used explicitly so this also remains stable
    # if NumPy's random implementation ever changes.
    state = (int(node_seed) * 1_103_515_245 + int(shard_seed_offset) * 12_345 + 0x9E3779B9) & 0x7FFFFFFF
    samples: list[list[int]] = []
    for _ in range(count):
        state = (1_103_515_245 * state + 12_345) & 0x7FFFFFFF
        key = state % groups
        state = (1_103_515_245 * state + 12_345) & 0x7FFFFFFF
        value = (state % 20_001) - 10_000
        samples.append([int(key), int(value)])
    return samples
