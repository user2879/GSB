"""版本化路由缓存与整体替换视图。

缓存条目记录所属配置/环/依赖图版本；命中时版本不一致视同未命中并记录原因。
版本演进时调用 :meth:`RouteCache.replace_view` 整体替换当前视图（而非逐条修补），
旧条目保留在历史区并标记 stale，不参与当前展示与汇总。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .determinism import content_digest


@dataclass(frozen=True)
class CacheEntry:
    entry_key: str
    unit_id: str
    route_key: str
    config_version: int
    ring_version: int
    dependency_version: int
    targets: tuple[str, ...]
    summary: str
    plan_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_key": self.entry_key,
            "unit_id": self.unit_id,
            "route_key": self.route_key,
            "config_version": self.config_version,
            "ring_version": self.ring_version,
            "dependency_version": self.dependency_version,
            "targets": list(self.targets),
            "summary": self.summary,
            "plan_digest": self.plan_digest,
        }


@dataclass
class CacheView:
    config_version: int
    ring_version: int
    dependency_version: int
    entries: dict[str, CacheEntry] = field(default_factory=dict)

    def digest(self) -> str:
        return content_digest(
            {
                "config_version": self.config_version,
                "ring_version": self.ring_version,
                "dependency_version": self.dependency_version,
                "entries": [
                    entry.to_dict()
                    for entry_key, entry in sorted(self.entries.items())
                ],
            }
        )


@dataclass
class ReplacementReport:
    changed_rules: list[str]
    affected_targets: list[str]
    cleared_entry_count: int
    old_versions: dict[str, int]
    new_versions: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_rules": list(self.changed_rules),
            "affected_targets": list(self.affected_targets),
            "cleared_entry_count": self.cleared_entry_count,
            "old_versions": dict(self.old_versions),
            "new_versions": dict(self.new_versions),
        }


class RouteCache:
    def __init__(self) -> None:
        self._view: CacheView | None = None
        self._stale: list[CacheEntry] = []
        self._last_report: ReplacementReport | None = None

    @property
    def view(self) -> CacheView | None:
        return self._view

    @property
    def stale_entries(self) -> list[CacheEntry]:
        return list(self._stale)

    def initialize(
        self,
        config_version: int,
        ring_version: int,
        dependency_version: int,
    ) -> CacheView:
        previous = self._view
        cleared = 0 if previous is None else len(previous.entries)
        if previous is not None:
            self._stale.extend(previous.entries.values())
        self._view = CacheView(
            config_version=config_version,
            ring_version=ring_version,
            dependency_version=dependency_version,
        )
        self._last_report = ReplacementReport(
            changed_rules=[],
            affected_targets=[],
            cleared_entry_count=cleared,
            old_versions={} if previous is None else self._versions_of(previous),
            new_versions={
                "config": config_version,
                "ring": ring_version,
                "dependency": dependency_version,
            },
        )
        return self._view

    def replace_view(
        self,
        *,
        config_version: int,
        ring_version: int,
        dependency_version: int,
        changed_rules: list[str],
        affected_targets: list[str],
    ) -> ReplacementReport:
        previous = self._view
        old_versions = {} if previous is None else self._versions_of(previous)
        cleared = 0 if previous is None else len(previous.entries)
        if previous is not None:
            self._stale.extend(previous.entries.values())
        self._view = CacheView(
            config_version=config_version,
            ring_version=ring_version,
            dependency_version=dependency_version,
        )
        report = ReplacementReport(
            changed_rules=sorted(set(changed_rules)),
            affected_targets=sorted(set(affected_targets)),
            cleared_entry_count=cleared,
            old_versions=old_versions,
            new_versions={
                "config": config_version,
                "ring": ring_version,
                "dependency": dependency_version,
            },
        )
        self._last_report = report
        return report

    def put(self, entry: CacheEntry) -> None:
        if self._view is None:
            raise RuntimeError("缓存视图尚未初始化")
        self._check_version(entry)
        self._view.entries[entry.entry_key] = entry

    def get(
        self, entry_key: str, *, current_versions: tuple[int, int, int]
    ) -> tuple[CacheEntry | None, str | None]:
        if self._view is None:
            return None, "缓存视图未初始化"
        entry = self._view.entries.get(entry_key)
        if entry is None:
            return None, "条目不存在"
        entry_versions = (
            entry.config_version,
            entry.ring_version,
            entry.dependency_version,
        )
        if entry_versions != current_versions:
            return None, (
                "版本不一致视同未命中："
                f"entry={entry_versions} current={current_versions}"
            )
        return entry, None

    def _check_version(self, entry: CacheEntry) -> None:
        current = (
            self._view.config_version,
            self._view.ring_version,
            self._view.dependency_version,
        )
        entry_versions = (
            entry.config_version,
            entry.ring_version,
            entry.dependency_version,
        )
        if current != entry_versions:
            raise ValueError(
                f"拒绝写入跨版本缓存条目 entry={entry_versions} current={current}"
            )

    @staticmethod
    def _versions_of(view: CacheView) -> dict[str, int]:
        return {
            "config": view.config_version,
            "ring": view.ring_version,
            "dependency": view.dependency_version,
        }

    def current_entries_sorted(self) -> list[CacheEntry]:
        if self._view is None:
            return []
        return [self._view.entries[key] for key in sorted(self._view.entries)]

    def last_report(self) -> ReplacementReport | None:
        return self._last_report
