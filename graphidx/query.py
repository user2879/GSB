"""遍历查询：邻接遍历、按标签过滤遍历、视图可达性查询。

查询绑定所使用的索引条目版本与视图版本；被引用条目或视图处于失效
状态时以 E_STALE 拒绝并列出失效引用，绝不返回旧缓存结论；索引同步
后同一查询重复执行得到逐字节一致的遍历序列。
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .errors import DependencyError, GraphError, IndexIntegrityError, StaleError
from .indexes import LABEL_INVERTED, WEIGHT_ADJACENCY


class QueryEngine:
    def __init__(self, session: Any):
        self.session = session

    def _index_name(self, type_: str) -> str:
        names = sorted(
            n for n, d in self.session.indexes.defs.items() if d.type == type_
        )
        if not names:
            raise IndexIntegrityError(
                "required index type not defined",
                [{"element": type_, "field": "type", "message": "no index of this type"}],
            )
        return names[0]

    # ------------------------------------------------------------------
    # 邻接遍历：按 (weight, dst, edge_id) 升序的确定性 BFS
    # ------------------------------------------------------------------
    def traverse_adjacency(self, start: int, max_depth: int | None = None) -> dict[str, Any]:
        graph = self.session.graph
        if start not in graph.nodes:
            raise GraphError(
                "start node does not exist",
                [{"element": str(start), "field": "start", "message": "unknown start node"}],
            )
        adj_name = self._index_name(WEIGHT_ADJACENCY)
        entries = self.session.indexes.entries[adj_name]
        bindings: dict[str, int] = {}
        stale: list[str] = []
        sequence = [start]
        visited = {start}
        queue: deque[tuple[int, int]] = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            entry = entries.get(str(node))
            if entry is None:
                continue  # 无出边
            bindings[entry.key] = entry.entry_version
            if entry.status != "valid":
                stale.append(f"index:{adj_name}:{entry.key}")
                continue
            for dst, _weight, _elabel, _eid in entry.content["edges"]:
                if dst not in visited:
                    visited.add(dst)
                    sequence.append(dst)
                    queue.append((dst, depth + 1))
        if stale:
            raise StaleError(
                "query references invalid index entries",
                [{"element": ref, "field": "status", "message": "entry is invalid"} for ref in stale],
            )
        return {
            "query": "traverse_adjacency",
            "start": start,
            "sequence": sequence,
            "bindings": {"indexes": {adj_name: bindings}, "views": {}},
        }

    # ------------------------------------------------------------------
    # 按标签过滤遍历：仅访问具有指定标签的节点
    # ------------------------------------------------------------------
    def traverse_by_label(self, start: int, label: str) -> dict[str, Any]:
        graph = self.session.graph
        if start not in graph.nodes:
            raise GraphError(
                "start node does not exist",
                [{"element": str(start), "field": "start", "message": "unknown start node"}],
            )
        label_name = self._index_name(LABEL_INVERTED)
        adj_name = self._index_name(WEIGHT_ADJACENCY)
        label_entry = self.session.indexes.entries[label_name].get(label)
        stale: list[str] = []
        label_bindings: dict[str, int] = {}
        if label_entry is None:
            allowed: set[int] = set()
        else:
            label_bindings[label_entry.key] = label_entry.entry_version
            if label_entry.status != "valid":
                stale.append(f"index:{label_name}:{label_entry.key}")
            allowed = set(label_entry.content["nodes"])
        adj_bindings: dict[str, int] = {}
        sequence: list[int] = []
        if start in allowed:
            sequence = [start]
            visited = {start}
            queue: deque[int] = deque([start])
            adj_entries = self.session.indexes.entries[adj_name]
            while queue:
                node = queue.popleft()
                entry = adj_entries.get(str(node))
                if entry is None:
                    continue
                adj_bindings[entry.key] = entry.entry_version
                if entry.status != "valid":
                    stale.append(f"index:{adj_name}:{entry.key}")
                    continue
                for dst, _weight, _elabel, _eid in entry.content["edges"]:
                    if dst in allowed and dst not in visited:
                        visited.add(dst)
                        sequence.append(dst)
                        queue.append(dst)
        if stale:
            raise StaleError(
                "query references invalid index entries",
                [{"element": ref, "field": "status", "message": "entry is invalid"} for ref in stale],
            )
        return {
            "query": "traverse_by_label",
            "start": start,
            "label": label,
            "sequence": sequence,
            "bindings": {
                "indexes": {label_name: label_bindings, adj_name: adj_bindings},
                "views": {},
            },
        }

    # ------------------------------------------------------------------
    # 视图可达性查询
    # ------------------------------------------------------------------
    def view_reachability(self, view_name: str) -> dict[str, Any]:
        views = self.session.views
        if view_name not in views.defs:
            raise DependencyError(
                "unknown view",
                [{"element": view_name, "field": "view", "message": "view does not exist"}],
            )
        upstream: set[str] = set()
        stack = [view_name]
        while stack:
            current = stack.pop()
            for up in views.defs[current].depends_on:
                if up not in upstream:
                    upstream.add(up)
                    stack.append(up)
        chain = [n for n in views.order if n in upstream] + [view_name]
        stale = [n for n in chain if views.states[n].status != "valid"]
        if stale:
            raise StaleError(
                "query references invalid views",
                [{"element": f"view:{n}", "field": "status", "message": "view is invalid"} for n in stale],
            )
        state = views.states[view_name]
        return {
            "query": "view_reachability",
            "view": view_name,
            "reachable": list(state.content["reachable"]),
            "bindings": {"indexes": {}, "views": {n: views.states[n].version for n in chain}},
        }
