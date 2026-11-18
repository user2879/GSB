"""派生依赖图：稳定拓扑序、环检测、增量失效传播。"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from .errors import CycleError, ReferenceError


class DependencyGraph:
    """以 adjacency 表达的 DAG。

    edges[k] 保存 k 直接依赖的上游键（k 由这些键派生）；
    反向索引 ``reverse[k]`` 保存直接下游，用于失效传播。
    """

    def __init__(self, edges: Dict[str, Tuple[str, ...]] | None = None):
        self.edges: Dict[str, Tuple[str, ...]] = dict(edges or {})
        self.reverse: Dict[str, Set[str]] = {k: set() for k in self.edges}
        for node, ups in self.edges.items():
            for up in ups:
                self.reverse.setdefault(up, set()).add(node)
                self.reverse.setdefault(node, self.reverse.get(node, set()))

    # ------------------------------------------------------------------ build
    @staticmethod
    def from_items(item_map: Dict[str, "object"]) -> "DependencyGraph":  # noqa: F821
        edges: Dict[str, Tuple[str, ...]] = {}
        for key, item in item_map.items():
            refs = item.derivation.ref_keys if item.derivation else ()
            missing = [r for r in refs if r not in item_map]
            if missing:
                raise ReferenceError(
                    f"配置项 {key} 引用了不存在的上游 {missing}",
                    {"key": key, "missing": missing},
                )
            edges[key] = tuple(refs)
        graph = DependencyGraph(edges)
        graph.assert_acyclic()
        return graph

    def assert_acyclic(self) -> None:
        """Kahn 算法检测环；有环时给出环上节点。"""
        indeg = {k: 0 for k in self.edges}
        for node, ups in self.edges.items():
            for up in ups:
                indeg[node] = indeg.get(node, 0) + 1
        queue = sorted(k for k, d in indeg.items() if d == 0)
        visited: List[str] = []
        work = dict(indeg)
        while queue:
            node = queue.pop(0)
            visited.append(node)
            for down in sorted(self.reverse.get(node, ())):
                work[down] -= 1
                if work[down] == 0:
                    queue.append(down)
            queue.sort()
        if len(visited) != len(self.edges):
            cyclic = sorted(set(self.edges) - set(visited))
            raise CycleError(
                f"派生依赖存在环，环上节点: {cyclic}",
                {"cyclic_nodes": cyclic},
            )

    def topological_order(self) -> List[str]:
        """稳定拓扑序：同层按键名字典序，保证可重放。"""
        indeg = {k: len(v) for k, v in self.edges.items()}
        queue = sorted(k for k, d in indeg.items() if d == 0)
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for down in sorted(self.reverse.get(node, ())):
                indeg[down] -= 1
                if indeg[down] == 0:
                    queue.append(down)
            queue.sort()
        if len(order) != len(self.edges):
            raise CycleError("拓扑排序失败：依赖图存在环", {})
        return order

    def downstream_closure(self, changed: Iterable[str]) -> List[str]:
        """变更集合的全部直接/间接下游，按稳定拓扑序返回（不含 changed 自身）。"""
        changed_set = set(changed)
        affected: Set[str] = set()
        stack = list(changed_set)
        while stack:
            node = stack.pop()
            for down in self.reverse.get(node, ()):
                if down not in affected and down not in changed_set:
                    affected.add(down)
                    stack.append(down)
        order = self.topological_order()
        return [k for k in order if k in affected]

    def reachability_matrix(self) -> Tuple[List[str], np.ndarray]:
        """NumPy 布尔可达矩阵（间接依赖的显式矩阵口径，便于审计解释）。"""
        keys = sorted(self.edges)
        idx = {k: i for i, k in enumerate(keys)}
        n = len(keys)
        reach = np.eye(n, dtype=bool)
        for node, ups in self.edges.items():
            for up in ups:
                reach[idx[node], idx[up]] = True
        # Warshall 闭包
        for k in range(n):
            reach |= reach[:, [k]] & reach[[k], :]
        return keys, reach

    def referrers(self, target: str) -> List[str]:
        """所有（直接）引用 target 的键，删除前检查用。"""
        return sorted(
            node for node, ups in self.edges.items() if target in ups
        )

    def assert_deletable(self, removed: Sequence[str], item_map: Dict[str, object]) -> None:
        refs = {key: self.referrers(key) for key in removed}
        blocked = {k: v for k, v in refs.items() if v}
        if blocked:
            raise ReferenceError(
                "删除的配置项仍被其他配置项引用",
                {"blocked": blocked},
            )
