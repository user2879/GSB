"""分阶段内容寻址缓存：缓存键、上游引用集合、精确失效与复用判定。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hashing import digest

STAGES = ("load", "coarse", "fine", "intra", "merge")


@dataclass
class CacheEntry:
    stage: str
    sample_id: str
    cache_key: str
    upstream: list[str]
    payload: dict[str, Any]
    fingerprints: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "sample_id": self.sample_id,
            "cache_key": self.cache_key,
            "upstream": self.upstream,
            "payload": self.payload,
            "fingerprints": self.fingerprints,
        }


@dataclass
class InvalidationReport:
    invalidated: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)


class StageCache:
    """以 (sample_id, stage) -> CacheEntry 保存的内存缓存，可整体序列化进快照。"""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], CacheEntry] = {}
        self.last_report = InvalidationReport()

    def put(self, entry: CacheEntry) -> None:
        self._entries[(entry.sample_id, entry.stage)] = entry

    def get(self, sample_id: str, stage: str) -> CacheEntry | None:
        return self._entries.get((sample_id, stage))

    def remove_sample(self, sample_id: str) -> list[str]:
        removed = [key for key in self._entries if key[0] == sample_id]
        for key in removed:
            self._entries.pop(key)
        return sorted(f"{sample_id}:{stage}" for _, stage in removed)

    def reconcile(self, previous: dict[tuple[str, str], CacheEntry],
                  expected: dict[tuple[str, str], CacheEntry] | None = None
                  ) -> InvalidationReport:
        """按新缓存键与上游引用集合做精确失效；键未变即复用。"""
        report = InvalidationReport()
        if expected is None:  # 兼容旧调用：以当前缓存为期望集合
            expected = self._entries
        old_keys = set(previous)
        new_keys = set(expected)
        for key in sorted(old_keys - new_keys):
            label = f"{key[0]}:{key[1]}"
            report.invalidated.append(label)
            report.reasons[label] = "sample_removed_or_stage_no_longer_present"
            self._entries.pop(key, None)
        for key in sorted(new_keys):
            label = f"{key[0]}:{key[1]}"
            wanted = expected[key]
            prior = previous.get(key)
            if prior is None:
                report.invalidated.append(label)
                report.reasons[label] = "first_build"
                self._entries[key] = wanted
                continue
            if prior.cache_key != wanted.cache_key:
                report.invalidated.append(label)
                changed = sorted(
                    name for name, fp in wanted.fingerprints.items()
                    if prior.fingerprints.get(name) != fp)
                report.reasons[label] = "fingerprint_changed:" + ",".join(changed)
                self._entries[key] = wanted
            else:
                report.reused.append(label)
                report.reasons[label] = "cache_key_unchanged"
        self.last_report = report
        return report

    def all_entries(self) -> list[CacheEntry]:
        return [self._entries[key] for key in sorted(self._entries)]

    def load_entries(self, entries: list[dict]) -> None:
        self._entries = {}
        for item in entries:
            entry = CacheEntry(
                stage=item["stage"], sample_id=item["sample_id"],
                cache_key=item["cache_key"], upstream=item["upstream"],
                payload=item["payload"], fingerprints=item["fingerprints"])
            self._entries[(entry.sample_id, entry.stage)] = entry

    def state_digest(self) -> str:
        return digest([entry.to_dict() for entry in self.all_entries()])
