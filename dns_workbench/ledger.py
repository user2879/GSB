"""缓存账本：版本识别、整体替换与陈旧结论防护。

触发整体替换的条件：
- 缓存记录格式版本集合变化（出现新的 format version）；
- 区版本 zone_version 单调递增。
旧账本整体标记 stale=True 并保留在历史中，但不参与当前展示与统计。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .cache import TtlCache
from .records import CacheEntry


@dataclass
class ReplacementDiff:
    added: list[tuple[str, int]] = field(default_factory=list)
    removed: list[tuple[str, int]] = field(default_factory=list)
    changed: list[dict] = field(default_factory=list)
    stale: bool = False

    def to_dict(self) -> dict:
        return {
            "added": [list(item) for item in self.added],
            "removed": [list(item) for item in self.removed],
            "changed": list(self.changed),
            "stale": self.stale,
        }


@dataclass
class LedgerGeneration:
    generation: int
    zone_version: int
    format_versions: tuple[str, ...]
    cache: TtlCache
    stale: bool = False


class CacheLedger:
    def __init__(self, *, capacity: int = 256, negative_ttl_cap: int = 300, clock=None) -> None:
        self._clock = clock
        self._capacity = capacity
        self._negative_ttl_cap = negative_ttl_cap
        self._generation = 0
        self._current: LedgerGeneration | None = None
        self.history: list[LedgerGeneration] = []
        self.last_diff: ReplacementDiff | None = None

    @property
    def generation(self) -> int:
        return self._generation

    def current(self) -> LedgerGeneration:
        if self._current is None:
            raise RuntimeError("账本尚未初始化，请先调用 replace/initialize")
        return self._current

    def cache(self) -> TtlCache:
        return self.current().cache

    def initialize(self, *, zone_version: int, entries: list[CacheEntry]) -> ReplacementDiff:
        return self.replace(zone_version=zone_version, entries=entries, force=True)

    def replace(
        self,
        *,
        zone_version: int,
        entries: list[CacheEntry],
        force: bool = False,
    ) -> ReplacementDiff:
        new_versions = tuple(sorted({entry.source_version for entry in entries}))
        old_snapshot: dict[tuple[str, int], CacheEntry] = {}
        old_zone_version = -1
        old_versions: tuple[str, ...] = ()
        if self._current is not None:
            old_zone_version = self._current.zone_version
            old_versions = self._current.format_versions
            for stored in self._current.cache.view():
                old_snapshot[stored.entry.key()] = stored.entry

        version_bump = self._current is None or zone_version > old_zone_version
        format_changed = new_versions != old_versions
        if not force and not version_bump and not format_changed:
            return ReplacementDiff()

        new_cache = TtlCache(
            capacity=self._capacity,
            negative_ttl_cap=self._negative_ttl_cap,
            clock=self._clock,
        )
        for entry in entries:
            new_cache.put(entry)

        new_keys = {entry.key() for entry in entries}
        old_keys = set(old_snapshot)
        diff = ReplacementDiff(
            added=sorted(new_keys - old_keys),
            removed=sorted(old_keys - new_keys),
        )
        new_by_key = {entry.key(): entry for entry in entries}
        for key in sorted(new_keys & old_keys):
            before = old_snapshot[key]
            after = new_by_key[key]
            if before.content_digest != after.content_digest or before.source_version != after.source_version:
                diff.changed.append(
                    {
                        "key": list(key),
                        "old_version": before.source_version,
                        "new_version": after.source_version,
                        "old_digest": before.content_digest,
                        "new_digest": after.content_digest,
                    }
                )

        if self._current is not None:
            self._current.stale = True
            self.history.append(self._current)
        diff.stale = self._current is not None
        self._generation += 1
        self._current = LedgerGeneration(
            generation=self._generation,
            zone_version=zone_version,
            format_versions=new_versions,
            cache=new_cache,
        )
        self.last_diff = diff
        return diff

    def ensure_version(self, zone_version: int, format_versions: tuple[str, ...]) -> bool:
        """返回是否发生了整体替换（版本号单调递增或格式集合变化）。"""
        if self._current is None:
            return True
        if zone_version < self._current.zone_version:
            from .errors import VersionError

            raise VersionError(
                "区版本不得倒退",
                fields=["zone_version"],
                context={"given": zone_version, "current": self._current.zone_version},
            )
        return zone_version > self._current.zone_version or tuple(sorted(format_versions)) != self._current.format_versions
