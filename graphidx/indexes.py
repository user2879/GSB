"""节点倒排索引与按边权重的有序邻接索引。

条目内容的序列化顺序由显式规则确定：
- label_inverted 条目内容为按节点标识升序的节点列表；
- weight_adjacency 条目内容按 (weight, dst, edge_id) 升序排列。
同一图内容在任意构建顺序下得到逐字节一致的索引摘要。
"""

from __future__ import annotations

from typing import Any

from .canon import digest as canonical_digest
from .errors import IndexIntegrityError
from .graph import ChangeRecord, PropertyGraph

LABEL_INVERTED = "label_inverted"
WEIGHT_ADJACENCY = "weight_adjacency"
INDEX_TYPES = (LABEL_INVERTED, WEIGHT_ADJACENCY)


class IndexDef:
    def __init__(self, name: str, type: str, version: int):
        self.name = name
        self.type = type
        self.version = version

    @staticmethod
    def from_dict(data: Any) -> "IndexDef":
        if not isinstance(data, dict):
            raise IndexIntegrityError(
                "index definition must be an object",
                [{"element": "index_def", "field": "type", "message": "expected object"}],
            )
        errors = []
        name = data.get("name")
        type_ = data.get("type")
        version = data.get("version")
        if not isinstance(name, str) or not name:
            errors.append({"element": "index_def", "field": "name", "message": "name must be a non-empty string"})
        if type_ not in INDEX_TYPES:
            errors.append(
                {"element": name or "index_def", "field": "type", "message": f"type must be one of {INDEX_TYPES}"}
            )
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            errors.append(
                {"element": name or "index_def", "field": "version", "message": "version must be a positive integer"}
            )
        if errors:
            raise IndexIntegrityError("invalid index definition", errors)
        return IndexDef(name, type_, version)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "version": self.version}


class IndexEntry:
    __slots__ = ("index", "key", "def_version", "entry_version", "status", "content", "digest")

    def __init__(self, index: str, key: str, def_version: int, entry_version: int,
                 status: str, content: Any, digest: str):
        self.index = index
        self.key = key
        self.def_version = def_version
        self.entry_version = entry_version
        self.status = status  # "valid" | "invalid"
        self.content = content
        self.digest = digest

    @classmethod
    def create(cls, index: str, key: str, def_version: int, entry_version: int,
               status: str, content: Any) -> "IndexEntry":
        return cls(index, key, def_version, entry_version, status, content, canonical_digest(content))

    def refresh(self, content: Any, def_version: int) -> None:
        self.content = content
        self.def_version = def_version
        self.digest = canonical_digest(content)
        self.status = "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "key": self.key,
            "def_version": self.def_version,
            "entry_version": self.entry_version,
            "status": self.status,
            "content": self.content,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexEntry":
        return cls(
            index=data["index"],
            key=data["key"],
            def_version=data["def_version"],
            entry_version=data["entry_version"],
            status=data["status"],
            content=data["content"],
            digest=data["digest"],
        )


def _key_sort(index_type: str, key: str) -> tuple:
    """条目键的稳定排序规则：邻接索引按键的整数值升序，倒排索引按字典序。"""
    if index_type == WEIGHT_ADJACENCY:
        return (0, int(key))
    return (1, key)


class IndexStore:
    def __init__(self, defs: list[IndexDef]):
        self.defs: dict[str, IndexDef] = {}
        for d in defs:
            if d.name in self.defs:
                raise IndexIntegrityError(
                    "duplicate index definition name",
                    [{"element": d.name, "field": "name", "message": "duplicate index name"}],
                )
            self.defs[d.name] = d
        self.entries: dict[str, dict[str, IndexEntry]] = {d.name: {} for d in defs}

    # ------------------------------------------------------------------
    # 定义版本
    # ------------------------------------------------------------------
    def defs_digest(self) -> str:
        return canonical_digest([self.defs[n].to_dict() for n in sorted(self.defs)])

    def _names_by_type(self, type_: str) -> list[str]:
        return sorted(n for n, d in self.defs.items() if d.type == type_)

    # ------------------------------------------------------------------
    # 条目内容计算（显式排序规则）
    # ------------------------------------------------------------------
    def compute_content(self, index_name: str, key: str, graph: PropertyGraph) -> Any | None:
        """计算条目内容；内容为空时返回 None（条目应被移除）。"""
        index_def = self.defs[index_name]
        if index_def.type == LABEL_INVERTED:
            nodes = sorted(nid for nid, label in graph.nodes.items() if label == key)
            if not nodes:
                return None
            return {"label": key, "nodes": nodes}
        src = int(key)
        outs = [e for e in graph.edges.values() if e.src == src]
        if not outs:
            return None
        rows = sorted(
            ([e.dst, e.weight, e.label, e.id] for e in outs),
            key=lambda row: (row[1], row[0], row[3]),
        )
        return {"src": src, "edges": rows}

    # ------------------------------------------------------------------
    # 全量构建
    # ------------------------------------------------------------------
    def build_full(self, graph: PropertyGraph) -> dict[str, Any]:
        for name in sorted(self.defs):
            index_def = self.defs[name]
            self.entries[name] = {}
            if index_def.type == LABEL_INVERTED:
                keys = sorted({label for label in graph.nodes.values()})
            else:
                srcs = {e.src for e in graph.edges.values()}
                keys = [str(nid) for nid in sorted(srcs)]
            for key in keys:
                content = self.compute_content(name, key, graph)
                if content is None:
                    continue
                self.entries[name][key] = IndexEntry.create(
                    name, key, index_def.version, 1, "valid", content
                )
        return self.build_report(graph)

    def index_digest(self, name: str) -> str:
        index_def = self.defs[name]
        pairs = [
            [key, self.entries[name][key].digest]
            for key in sorted(self.entries[name], key=lambda k: _key_sort(index_def.type, k))
        ]
        return canonical_digest(pairs)

    def build_report(self, graph: PropertyGraph) -> dict[str, Any]:
        return {
            "graph_version": graph.version,
            "graph_digest": graph.content_digest(),
            "index_def_version": self.defs_digest(),
            "indexes": {name: self.index_digest(name) for name in sorted(self.defs)},
            "entry_counts": {name: len(self.entries[name]) for name in sorted(self.defs)},
        }

    # ------------------------------------------------------------------
    # 受影响条目推导（只包含与变更相关的条目，输出顺序稳定）
    # ------------------------------------------------------------------
    def derive_affected(self, change: ChangeRecord) -> list[tuple[str, str]]:
        affected: set[tuple[str, str]] = set()
        label_indexes = self._names_by_type(LABEL_INVERTED)
        adj_indexes = self._names_by_type(WEIGHT_ADJACENCY)
        for op in change.ops_detail:
            kind = op["op"]
            if kind == "add_node":
                for name in label_indexes:
                    affected.add((name, op["label"]))
            elif kind == "remove_node":
                for name in label_indexes:
                    affected.add((name, op["old_label"]))
            elif kind == "set_node_label":
                for name in label_indexes:
                    affected.add((name, op["old_label"]))
                    affected.add((name, op["new_label"]))
            elif kind in ("add_edge", "remove_edge", "set_edge_weight"):
                for name in adj_indexes:
                    affected.add((name, str(op["src"])))
        return sorted(affected, key=lambda ik: (ik[0], _key_sort(self.defs[ik[0]].type, ik[1])))

    def mark_invalid(self, name: str, key: str) -> None:
        entry = self.entries[name].get(key)
        if entry is not None:
            entry.status = "invalid"

    # ------------------------------------------------------------------
    # 单条目重算：版本递增、摘要刷新；空内容条目被移除
    # ------------------------------------------------------------------
    def recompute(self, name: str, key: str, graph: PropertyGraph) -> dict[str, Any]:
        content = self.compute_content(name, key, graph)
        entry = self.entries[name].get(key)
        if content is None:
            if entry is not None:
                old_version = entry.entry_version
                del self.entries[name][key]
                return {"index": name, "key": key, "action": "removed", "old_version": old_version}
            return {"index": name, "key": key, "action": "noop"}
        if entry is None:
            entry = IndexEntry.create(name, key, self.defs[name].version, 1, "valid", content)
            self.entries[name][key] = entry
            return {"index": name, "key": key, "action": "created", "new_version": 1, "digest": entry.digest}
        entry.entry_version += 1
        entry.refresh(content, self.defs[name].version)
        return {
            "index": name,
            "key": key,
            "action": "updated",
            "new_version": entry.entry_version,
            "digest": entry.digest,
        }
