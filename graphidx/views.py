"""可达性派生视图：依赖图、失效传播与基于 NumPy 的重算。

- 视图内容 = 源节点集合的可达闭包 ∪ 上游视图可达集合；
- 依赖成环以 E_DEPEND 拒绝并输出环上视图序列；
- 删除仍被引用的视图以 E_DEPEND 拒绝并列出引用者；
- 图结构或边权重变化时，沿依赖图推导直接与间接受影响视图，
  并按稳定拓扑顺序失效。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .canon import digest as canonical_digest
from .errors import DependencyError
from .graph import ChangeRecord, PropertyGraph


class ViewDef:
    def __init__(self, name: str, sources: list[int], depends_on: list[str]):
        self.name = name
        self.sources = sorted(set(sources))
        self.depends_on = list(depends_on)

    @staticmethod
    def from_dict(data: Any) -> "ViewDef":
        if not isinstance(data, dict):
            raise DependencyError(
                "view definition must be an object",
                [{"element": "view_def", "field": "type", "message": "expected object"}],
            )
        errors = []
        name = data.get("name")
        sources = data.get("sources", [])
        depends_on = data.get("depends_on", [])
        if not isinstance(name, str) or not name:
            errors.append({"element": "view_def", "field": "name", "message": "name must be a non-empty string"})
        if not isinstance(sources, list) or any(
            not isinstance(s, int) or isinstance(s, bool) for s in sources
        ):
            errors.append({"element": name or "view_def", "field": "sources", "message": "sources must be a list of integer node ids"})
        if not isinstance(depends_on, list) or any(not isinstance(d, str) for d in depends_on):
            errors.append({"element": name or "view_def", "field": "depends_on", "message": "depends_on must be a list of view names"})
        if errors:
            raise DependencyError("invalid view definition", errors)
        return ViewDef(name, sources, depends_on)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "sources": list(self.sources), "depends_on": list(self.depends_on)}


class ViewState:
    __slots__ = ("name", "version", "status", "content", "digest")

    def __init__(self, name: str, version: int, status: str, content: Any, digest: str):
        self.name = name
        self.version = version
        self.status = status  # "valid" | "invalid"
        self.content = content
        self.digest = digest

    @classmethod
    def fresh(cls, name: str) -> "ViewState":
        return cls(name, 0, "invalid", None, "")

    def refresh(self, content: Any) -> None:
        self.version += 1
        self.content = content
        self.digest = canonical_digest(content)
        self.status = "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.name,
            "version": self.version,
            "status": self.status,
            "content": self.content,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ViewState":
        return cls(data["view"], data["version"], data["status"], data["content"], data["digest"])


class ViewStore:
    def __init__(self, defs: list[ViewDef]):
        self.defs: dict[str, ViewDef] = {}
        for d in defs:
            if d.name in self.defs:
                raise DependencyError(
                    "duplicate view definition name",
                    [{"element": d.name, "field": "name", "message": "duplicate view name"}],
                )
            self.defs[d.name] = d
        for d in defs:
            for up in d.depends_on:
                if up not in self.defs:
                    raise DependencyError(
                        "view depends on unknown view",
                        [{"element": d.name, "field": "depends_on", "message": f"unknown view {up!r}"}],
                    )
        self.states: dict[str, ViewState] = {name: ViewState.fresh(name) for name in self.defs}
        self.order = self._topo_order()

    # ------------------------------------------------------------------
    # 依赖图
    # ------------------------------------------------------------------
    def _dependents(self) -> dict[str, list[str]]:
        dependents: dict[str, list[str]] = {name: [] for name in self.defs}
        for d in self.defs.values():
            for up in d.depends_on:
                dependents[up].append(d.name)
        return dependents

    def _topo_order(self) -> list[str]:
        """稳定拓扑排序；发现环时以 E_DEPEND 输出环上视图序列。"""
        indegree = {name: len(d.depends_on) for name, d in self.defs.items()}
        ready = sorted(n for n, deg in indegree.items() if deg == 0)
        order: list[str] = []
        dependents = self._dependents()
        while ready:
            name = ready.pop(0)
            order.append(name)
            for down in sorted(dependents[name]):
                indegree[down] -= 1
                if indegree[down] == 0:
                    ready.append(down)
            ready.sort()
        if len(order) != len(self.defs):
            cycle = self._find_cycle()
            raise DependencyError(
                "view dependency cycle detected",
                [{"element": "->".join(cycle), "field": "depends_on", "message": "dependency cycle"}],
            )
        return order

    def _find_cycle(self) -> list[str]:
        color = {name: 0 for name in self.defs}  # 0=white 1=gray 2=black
        stack: list[str] = []
        cycle: list[str] = []

        def dfs(node: str) -> bool:
            color[node] = 1
            stack.append(node)
            for up in sorted(self.defs[node].depends_on):
                if color[up] == 1:
                    start = stack.index(up)
                    cycle.extend(stack[start:] + [up])
                    return True
                if color[up] == 0 and dfs(up):
                    return True
            stack.pop()
            color[node] = 2
            return False

        for name in sorted(self.defs):
            if color[name] == 0 and dfs(name):
                break
        return cycle

    def validate_sources(self, graph: PropertyGraph) -> None:
        errors = []
        for name in sorted(self.defs):
            for src in self.defs[name].sources:
                if src not in graph.nodes:
                    errors.append(
                        {"element": name, "field": "sources", "message": f"source node {src} does not exist"}
                    )
        if errors:
            raise DependencyError("view sources reference missing nodes", errors)

    def remove_def(self, name: str) -> None:
        """删除视图定义；仍被引用时以 E_DEPEND 拒绝并列出引用者。"""
        if name not in self.defs:
            raise DependencyError(
                "unknown view",
                [{"element": name, "field": "view", "message": "view does not exist"}],
            )
        referrers = sorted(d.name for d in self.defs.values() if name in d.depends_on)
        if referrers:
            raise DependencyError(
                f"view {name!r} is still referenced",
                [
                    {"element": name, "field": "depends_on", "message": f"referenced by {ref}"}
                    for ref in referrers
                ],
            )
        del self.defs[name]
        del self.states[name]
        self.order = self._topo_order()

    # ------------------------------------------------------------------
    # 可达性计算（NumPy 布尔矩阵闭包）
    # ------------------------------------------------------------------
    def compute(self, name: str, graph: PropertyGraph) -> dict[str, Any]:
        view_def = self.defs[name]
        ids = sorted(graph.nodes)
        index = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)
        adj = np.zeros((n, n), dtype=bool)
        for edge in graph.edges.values():
            adj[index[edge.src], index[edge.dst]] = True
        reach = np.zeros(n, dtype=bool)
        for src in view_def.sources:
            if src in index:
                reach[index[src]] = True
        for up in view_def.depends_on:
            upstream = self.states[up]
            if upstream.content is None:
                raise DependencyError(
                    "upstream view not computed",
                    [{"element": name, "field": "depends_on", "message": f"upstream view {up!r} has no content"}],
                )
            for nid in upstream.content["reachable"]:
                if nid in index:
                    reach[index[nid]] = True
        if n:
            while True:
                nxt = reach | (reach @ adj)
                if np.array_equal(nxt, reach):
                    break
                reach = nxt
        return {"view": name, "reachable": [ids[i] for i in range(n) if reach[i]]}

    def compute_all(self, graph: PropertyGraph) -> dict[str, str]:
        for name in self.order:
            self.recompute(name, graph)
        return {name: self.states[name].digest for name in self.order}

    def recompute(self, name: str, graph: PropertyGraph) -> dict[str, Any]:
        content = self.compute(name, graph)
        state = self.states[name]
        state.refresh(content)
        return {"view": name, "new_version": state.version, "digest": state.digest}

    # ------------------------------------------------------------------
    # 失效传播：直接影响 + 沿依赖图的间接影响，稳定拓扑顺序输出
    # ------------------------------------------------------------------
    def invalidate(self, change: ChangeRecord) -> list[str]:
        if not change.structural:
            return []
        changed = set(change.structural_nodes)
        direct: set[str] = set()
        for name in self.order:
            state = self.states[name]
            if state.content is None:
                direct.add(name)
                continue
            region = set(self.defs[name].sources) | set(state.content["reachable"])
            if changed & region:
                direct.add(name)
        affected = set(direct)
        dependents = self._dependents()
        queue = list(direct)
        while queue:
            current = queue.pop()
            for down in dependents[current]:
                if down not in affected:
                    affected.add(down)
                    queue.append(down)
        ordered = [name for name in self.order if name in affected]
        for name in ordered:
            self.states[name].status = "invalid"
        return ordered
