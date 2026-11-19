"""确定性失效传播算法。

触发集合由显式失效、过期回收、命名空间刷新三类来源合并得到；随后在
注释依赖图上求可达闭包，并用最小堆做 Kahn 拓扑排序。图在加载期已
保证无环，因此结果只取决于图本身，与提交顺序、线程调度无关。
"""

from __future__ import annotations

import heapq
from collections import defaultdict

from .model import EntryID, Graph


def reachable_closure(graph: Graph,
                      seeds: set[EntryID],
                      alive: set[EntryID] | None = None) -> list[EntryID]:
    """返回 seeds 在注释边上的全部可达存活节点（含 seeds），稳定拓扑顺序。

    已失效条目不参与新一轮传播：它们不会再次出现在受影响序列中，也不会
    再次递增注释版本。alive 为 None 时视为全部存活，便于纯图算法测试。
    """
    alive = alive if alive is not None else set(graph.entries)
    reachable: set[EntryID] = set()
    stack = sorted(seeds, reverse=True)
    while stack:
        node = stack.pop()
        if node in reachable or node not in alive:
            continue
        reachable.add(node)
        stack.extend(sorted(
            (target for target in graph.annotation_edges[node]
             if target in alive),
            reverse=True))

    indegree: dict[EntryID, int] = {node: 0 for node in reachable}
    adjacency: dict[EntryID, list[EntryID]] = defaultdict(list)
    for node in reachable:
        for target in graph.annotation_edges[node]:
            if target in reachable:
                adjacency[node].append(target)
                indegree[target] += 1

    heap: list[EntryID] = [
        node for node in reachable if indegree[node] == 0
    ]
    heapq.heapify(heap)
    ordered: list[EntryID] = []
    while heap:
        node = heapq.heappop(heap)
        ordered.append(node)
        for target in sorted(adjacency[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(heap, target)

    if len(ordered) != len(reachable):
        # 加载阶段已做无环校验，这里属于防御性断言而非业务输入错误。
        raise RuntimeError("注释依赖图传播时检测到环，图加载校验失效")
    return ordered


def namespace_seed_entries(graph: Graph,
                           namespaces: list[str]) -> list[EntryID]:
    """命名空间刷新触发：沿继承链展开下游命名空间中的存活条目。"""
    expanded: set[str] = set()
    for root in namespaces:
        expanded.update(graph.downstream_namespaces(root))
    seeds = [
        entry_id for entry_id in graph.entry_ids_sorted()
        if entry_id.namespace in expanded
    ]
    return seeds


def expired_entries(graph: Graph, now_ms: int,
                    alive: set[EntryID]) -> list[EntryID]:
    return [
        entry_id for entry_id in graph.entry_ids_sorted()
        if entry_id in alive
        and graph.entries[entry_id].expire_ms <= now_ms
    ]
