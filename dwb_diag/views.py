"""派生视图定义、依赖传播、失效重建与一致性校验。

视图两类(要求 5/6):
  prefix_count : 按键前缀计数, 依赖该前缀键所在页集合
  value_range  : 按值区间的有序视图, 依赖值落在 [lo, hi] 的键所在页集合
  rollup       : 视图对视图的聚合(用于制造直接+间接传播链与成环检测)
页被覆盖/修复时, 沿 page -> view -> view 依赖推导全部受影响视图,
按稳定拓扑顺序输出失效序列并递增视图版本; 成环抛 E_DEPEND。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DependCycleError, ViewError
from .keyspace import Keyspace
from .util import canonical_json, digest_of_json, write_bytes_atomic


@dataclass(frozen=True)
class ViewDef:
    name: str
    kind: str                      # prefix_count / value_range / rollup
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "params": self.params}

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "ViewDef":
        return ViewDef(str(obj["name"]), str(obj["kind"]), dict(obj.get("params", {})))


@dataclass
class ViewState:
    definition: ViewDef
    version: int = 0
    digest: str | None = None
    content: Any = None
    stale: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.definition.name,
            "kind": self.definition.kind,
            "version": self.version,
            "digest": self.digest,
            "stale": self.stale,
        }


class ViewCatalog:
    """视图定义集合(持久化 views.json), 名称唯一。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.defs: dict[str, ViewDef] = {}

    @classmethod
    def load(cls, path: Path) -> "ViewCatalog":
        cat = cls(path)
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            for item in obj.get("views", []):
                d = ViewDef.from_dict(item)
                cat.defs[d.name] = d
        return cat

    def save(self) -> None:
        payload = {
            "v": 1,
            "views": [self.defs[k].to_dict() for k in sorted(self.defs)],
        }
        write_bytes_atomic(self.path, canonical_json(payload).encode("utf-8"))

    def add(self, definition: ViewDef) -> None:
        self.defs[definition.name] = definition

    def names_sorted(self) -> list[str]:
        return sorted(self.defs)


class ViewManager:
    """构建依赖图、传播失效并基于已恢复主文件重建视图。"""

    def __init__(self, catalog: ViewCatalog, keyspace: Keyspace) -> None:
        self.catalog = catalog
        self.keyspace = keyspace
        self.states: dict[str, ViewState] = {
            name: ViewState(catalog.defs[name]) for name in catalog.names_sorted()
        }
        # 依赖图: 视图 -> 它直接依赖的页号集合 / 视图名集合
        self._page_deps: dict[str, set[int]] = {}
        self._view_deps: dict[str, set[str]] = {}
        self.rebuild_graph()

    # ---------- 依赖图 ----------
    def rebuild_graph(self) -> None:
        self._page_deps = {}
        self._view_deps = {}
        entries_by_page = self.keyspace.all_page_entries()
        for name in self.catalog.names_sorted():
            d = self.catalog.defs[name]
            if d.kind == "prefix_count":
                prefix = str(d.params["prefix"])
                pages = {
                    pno
                    for pno, entries in entries_by_page.items()
                    if any(k.startswith(prefix) for k in entries)
                }
                self._page_deps[name] = pages
                self._view_deps[name] = set()
            elif d.kind == "value_range":
                lo = int(d.params["lo"])
                hi = int(d.params["hi"])
                pages = {
                    pno
                    for pno, entries in entries_by_page.items()
                    if any(lo <= v <= hi for v in entries.values())
                }
                self._page_deps[name] = pages
                self._view_deps[name] = set()
            elif d.kind == "rollup":
                sources = [str(s) for s in d.params.get("sources", [])]
                self._page_deps[name] = set()
                self._view_deps[name] = set(sources)
            else:
                raise ValueError(f"unknown view kind {d.kind}")

    def direct_page_deps(self, name: str) -> set[int]:
        return set(self._page_deps.get(name, set()))

    def direct_view_deps(self, name: str) -> set[str]:
        return set(self._view_deps.get(name, set()))

    def detect_cycle(self) -> list[str]:
        """返回视图依赖环(节点名序列); 无环返回空列表。"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in self.catalog.names_sorted()}
        stack: list[str] = []

        def visit(node: str) -> list[str] | None:
            color[node] = GRAY
            stack.append(node)
            for nxt in sorted(self._view_deps.get(node, set())):
                if nxt not in color:
                    continue
                if color[nxt] == GRAY:
                    idx = stack.index(nxt)
                    return stack[idx:] + [nxt]
                if color[nxt] == WHITE:
                    found = visit(nxt)
                    if found is not None:
                        return found
            stack.pop()
            color[node] = BLACK
            return None

        for name in self.catalog.names_sorted():
            if color[name] == WHITE:
                found = visit(name)
                if found is not None:
                    return found
        return []

    def topological_views(self) -> list[str]:
        """Kahn 拓扑排序(被依赖者在前), 名称排序保证稳定。"""
        names = self.catalog.names_sorted()
        indeg = {n: 0 for n in names}
        adj: dict[str, list[str]] = {n: [] for n in names}
        for name in names:
            for src in sorted(self._view_deps.get(name, set())):
                if src in indeg:
                    indeg[name] += 1
                    adj[src].append(name)
        ready = sorted([n for n, d in indeg.items() if d == 0])
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for nxt in sorted(adj[node]):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
                    ready.sort()
        if len(order) != len(names):
            cycle = self.detect_cycle()
            raise DependCycleError(
                "view dependency graph contains a cycle", cycle=cycle
            )
        return order

    # ---------- 失效传播 ----------
    def propagate_invalidation(self, changed_pages: set[int]) -> list[str]:
        """从变更页集合推导全部直接+间接失效视图, 返回稳定拓扑序列。"""
        cycle = self.detect_cycle()
        if cycle:
            raise DependCycleError(
                "view dependency graph contains a cycle", cycle=cycle
            )
        changed = set(changed_pages)
        affected: set[str] = set()
        # 页 -> 直接依赖视图(与遍历顺序无关: 先收集到集合)
        for name in self.catalog.names_sorted():
            if self._page_deps.get(name) & changed:
                affected.add(name)
        # 沿 view -> view 闭包传播
        grown = True
        while grown:
            grown = False
            for name in self.catalog.names_sorted():
                if name in affected:
                    continue
                if self._view_deps.get(name) & affected:
                    affected.add(name)
                    grown = True
        order = self.topological_views()
        sequence = [n for n in order if n in affected]
        for name in sequence:
            state = self.states[name]
            state.version += 1
            state.stale = True
        return sequence

    # ---------- 重建与校验 ----------
    def _compute(self, name: str) -> Any:
        d = self.catalog.defs[name]
        page_entries = self.keyspace.all_page_entries()
        if d.kind == "prefix_count":
            prefix = str(d.params["prefix"])
            count = sum(
                1
                for entries in page_entries.values()
                for k in entries
                if k.startswith(prefix)
            )
            return {"kind": "prefix_count", "prefix": prefix, "count": count}
        if d.kind == "value_range":
            lo, hi = int(d.params["lo"]), int(d.params["hi"])
            pairs = [
                (k, v)
                for entries in page_entries.values()
                for k, v in entries.items()
                if lo <= v <= hi
            ]
            pairs.sort(key=lambda kv: (kv[1], kv[0]))
            return {"kind": "value_range", "lo": lo, "hi": hi, "items": pairs}
        if d.kind == "rollup":
            sources = [str(s) for s in d.params.get("sources", [])]
            total = 0
            for src in sources:
                content = self._compute(src)
                if content.get("kind") == "prefix_count":
                    total += int(content["count"])
                elif content.get("kind") == "value_range":
                    total += len(content["items"])
            return {"kind": "rollup", "sources": sorted(sources), "total": total}
        raise ValueError(f"unknown view kind {d.kind}")

    def rebuild_one(self, name: str) -> ViewState:
        """仅依据当前主文件已恢复页内容重建, 不复用失效前条目。"""
        state = self.states[name]
        content = self._compute(name)
        state.content = content
        state.digest = digest_of_json(content)
        state.stale = False
        return state

    def rebuild_all(self) -> list[str]:
        """按拓扑序基于当前主文件重算全部视图。

        返回因失效(stale 且版本已递增)而重建的视图名; 其余视图仅刷新内容,
        保证任何作答都来自当前已恢复页而非旧缓存。
        """
        rebuilt: list[str] = []
        for name in self.topological_views():
            state = self.states[name]
            invalidated = state.stale and state.version > 0
            self.rebuild_one(name)
            if invalidated:
                rebuilt.append(name)
        return rebuilt

    def verify(self) -> list[dict[str, Any]]:
        """要求 6: 比较重建前后摘要与版本; 异常以 E_VIEW 报首个视图名与字段。"""
        snapshot = {
            name: (self.states[name].version, self.states[name].digest)
            for name in self.catalog.names_sorted()
        }
        order = self.topological_views()
        for name in order:
            state = self.states[name]
            before_version, before_digest = snapshot[name]
            # 从未参与失效传播的视图(版本 0 且未标记 stale)不在校验范围;
            # 版本 0 却 stale 属于"失效后未递增", 仍须以 E_VIEW 报告
            if before_version == 0 and not state.stale:
                continue
            content = self._compute(name)
            recomputed = digest_of_json(content)
            if state.version <= 0 or state.version != before_version:
                raise ViewError(
                    "view version not incremented as required",
                    view=name,
                    field="version",
                )
            if state.stale:
                raise ViewError(
                    "view still stale before verify", view=name, field="stale"
                )
            if state.digest is None or state.digest != recomputed:
                raise ViewError(
                    "view digest mismatch on recompute",
                    view=name,
                    field="digest",
                )
        return [
            {
                "name": name,
                "version": self.states[name].version,
                "digest": self.states[name].digest,
            }
            for name in order
        ]

    def digest_manifest(self) -> dict[str, Any]:
        return {
            name: {
                "version": self.states[name].version,
                "digest": self.states[name].digest,
            }
            for name in self.catalog.names_sorted()
        }

    def replace_states(self, versions: dict[str, int]) -> None:
        """快照整体替换视图状态(标记为待重建, 不做增量合并)。"""
        keep = {name: ViewState(self.catalog.defs[name]) for name in self.catalog.names_sorted()}
        for name, version in versions.items():
            if name in keep:
                keep[name].version = int(version)
                keep[name].stale = True
        self.states = keep
