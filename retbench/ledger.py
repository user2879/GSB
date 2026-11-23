"""阶段级内容指纹账本：精确失效、复用判定与等价性校验。

阶段 DAG（下游失效随上游传播）：
    visual_features -> alignment -> recall -> diagnosis
    text_features  --^

缓存键由"上游内容指纹 + 参数指纹 + 量化档位"派生；账本同时记录每个结果
引用的分块与断言集合，供审计与逐查询归因使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fingerprint import digest_many, digest_obj

STAGES = ("visual", "text", "alignment", "recall", "diagnosis")
DOWNSTREAM: dict[str, tuple[str, ...]] = {
    "visual": ("alignment", "recall", "diagnosis"),
    "text": ("alignment", "recall", "diagnosis"),
    "alignment": ("recall", "diagnosis"),
    "recall": ("diagnosis"),
    "diagnosis": (),
}


@dataclass
class LedgerEntry:
    stage: str
    sample: str
    cache_key: str
    upstream_keys: list[str]
    referenced_tiles: list[tuple[int, int]] = field(default_factory=list)
    referenced_assertions: list[int] = field(default_factory=list)
    payload_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "sample": self.sample,
            "cache_key": self.cache_key,
            "upstream_keys": list(self.upstream_keys),
            "referenced_tiles": [list(t) for t in self.referenced_tiles],
            "referenced_assertions": list(self.referenced_assertions),
            "payload_fingerprint": self.payload_fingerprint,
        }


class Ledger:
    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], LedgerEntry] = {}

    def upsert(self, entry: LedgerEntry) -> None:
        self.entries[(entry.stage, entry.sample)] = entry

    def get(self, stage: str, sample: str) -> LedgerEntry | None:
        return self.entries.get((stage, sample))

    def stage_keys(self, stage: str) -> dict[str, str]:
        return {sample: entry.cache_key
                for (st, sample), entry in self.entries.items() if st == stage}

    def invalidate(self, stage: str, samples: set[str] | None = None) -> set[str]:
        """失效指定阶段（可限定样本集合）并传播下游，返回全部失效 (stage, sample)。"""

        affected: set[tuple[str, str]] = set()

        def walk(st: str, scope: set[str]) -> None:
            targets = {(s, name) for (s, name) in self.entries
                       if s == st and (not scope or name in scope)}
            new = targets - affected
            if not new:
                return
            affected.update(new)
            next_scope = {name for (_, name) in new}
            for downstream in DOWNSTREAM[st]:
                walk(downstream, next_scope)

        walk(stage, samples or set())
        for key in affected:
            self.entries.pop(key, None)
        return affected

    def changed_scopes(self, desired: dict[tuple[str, str], str]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
        """对照期望缓存键，返回 (重算集合, 复用集合)。"""

        recompute: set[tuple[str, str]] = set()
        reuse: set[tuple[str, str]] = set()
        for key, desired_cache_key in desired.items():
            current = self.entries.get(key)
            if current is None or current.cache_key != desired_cache_key:
                recompute.add(key)
            else:
                reuse.add(key)
        return recompute, reuse

    def diff_payload(self, fresh_entries: dict[tuple[str, str], LedgerEntry]) -> list[str]:
        """全量重建结果与账本复用结果的逐键等价性校验，返回不一致描述列表。"""

        mismatches: list[str] = []
        for key, fresh in fresh_entries.items():
            old = self.entries.get(key)
            if old is None:
                mismatches.append(f"missing:{key[0]}/{key[1]}")
                continue
            if old.cache_key != fresh.cache_key:
                mismatches.append(f"cache_key:{key[0]}/{key[1]}")
            if old.payload_fingerprint != fresh.payload_fingerprint:
                mismatches.append(f"payload:{key[0]}/{key[1]}")
        extra = set(self.entries) - set(fresh_entries)
        for key in sorted(extra):
            mismatches.append(f"extra:{key[0]}/{key[1]}")
        return sorted(mismatches)

    def export(self) -> list[dict[str, Any]]:
        return [self.entries[key].to_dict() for key in sorted(self.entries)]


def make_cache_key(stage: str, sample: str, upstream: list[str],
                   param_fingerprint: str, quant_key: str,
                   payload: Any) -> str:
    return digest_many(
        stage, sample, digest_obj(upstream), param_fingerprint,
        quant_key, digest_obj(payload)
    )
