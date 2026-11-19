"""领域模型：条目 ID、条目、命名空间与注释依赖图。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoundRecord:
    """失效账本中的一轮传播记录，也是快照轨迹的可复现单元。"""

    round_no: int
    clock_ms: int
    trigger_reason: str
    explicit_triggers: tuple[str, ...]
    expired_triggers: tuple[str, ...]
    refresh_triggers: tuple[str, ...]
    affected: tuple[str, ...]
    annotation_versions: dict[str, int]
    format_version: int

    def to_dict(self) -> dict:
        return {
            "round_no": self.round_no,
            "clock_ms": self.clock_ms,
            "trigger_reason": self.trigger_reason,
            "explicit_triggers": list(self.explicit_triggers),
            "expired_triggers": list(self.expired_triggers),
            "refresh_triggers": list(self.refresh_triggers),
            "affected": list(self.affected),
            "annotation_versions": dict(
                sorted(self.annotation_versions.items())),
            "format_version": self.format_version,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RoundRecord":
        return cls(
            round_no=int(payload["round_no"]),
            clock_ms=int(payload["clock_ms"]),
            trigger_reason=payload["trigger_reason"],
            explicit_triggers=tuple(payload.get("explicit_triggers", [])),
            expired_triggers=tuple(payload.get("expired_triggers", [])),
            refresh_triggers=tuple(payload.get("refresh_triggers", [])),
            affected=tuple(payload["affected"]),
            annotation_versions={
                str(key): int(value)
                for key, value in payload["annotation_versions"].items()
            },
            format_version=int(payload["format_version"]),
        )


@dataclass(frozen=True, order=True)
class EntryID:
    """条目的稳定标识。条目键只在命名空间内唯一，因此用二元组标识。"""

    namespace: str
    key: int

    @classmethod
    def parse(cls, text: str) -> "EntryID":
        if "/" not in text:
            raise ValueError(f"条目标识必须形如 namespace/key: {text!r}")
        namespace, raw_key = text.rsplit("/", 1)
        if not namespace:
            raise ValueError(f"条目标识缺少命名空间: {text!r}")
        return cls(namespace, int(raw_key))

    def to_str(self) -> str:
        return f"{self.namespace}/{self.key}"

    def __str__(self) -> str:
        return self.to_str()


@dataclass(frozen=True)
class Entry:
    id: EntryID
    expire_ms: int
    digest: str
    digest_algorithm: str = "none"
    annotation_version: int = 0
    alive: bool = True


@dataclass(frozen=True)
class Namespace:
    name: str
    parents: tuple[str, ...] = ()


@dataclass
class Graph:
    """注释依赖图与命名空间继承链的内存表示。

    annotation_edges: 条目 -> 它所注释的下游条目集合（前者失效，后者连带失效）。
    namespace_parents: 命名空间 -> 直接上游命名空间（有序、去重）。
    """

    namespaces: dict[str, Namespace]
    entries: dict[EntryID, Entry]
    annotation_edges: dict[EntryID, set[EntryID]] = field(default_factory=dict)
    namespace_parents: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for eid in self.entries:
            self.annotation_edges.setdefault(eid, set())
        if not self.namespace_parents:
            self.namespace_parents = {
                name: ns.parents for name, ns in self.namespaces.items()
            }

    def entry_ids_sorted(self) -> list[EntryID]:
        return sorted(self.entries)

    def downstream_namespaces(self, root: str) -> list[str]:
        """返回继承链上 root 的全部下游命名空间（含 root），稳定排序。"""
        children: dict[str, list[str]] = {name: [] for name in self.namespaces}
        for name, parents in self.namespace_parents.items():
            for parent in parents:
                children.setdefault(parent, []).append(name)
                children.setdefault(name, children.get(name, []))
        seen: set[str] = set()
        stack = [root]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(sorted(children.get(current, ())))
        return sorted(seen)

    def digest_algorithms(self) -> dict[str, str]:
        return {
            entry.id.to_str(): entry.digest_algorithm
            for entry in self.entries.values()
        }
