"""基于内容指纹的阶段产物缓存与规则引用索引。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .fingerprint import content_fingerprint

STAGE_SEGMENT = "segment"
STAGE_DETECT = "detect"
STAGE_SCORE = "score"
STAGE_AGGREGATE = "aggregate"
STAGE_DECISION = "decision"
STAGES = (STAGE_SEGMENT, STAGE_DETECT, STAGE_SCORE, STAGE_AGGREGATE, STAGE_DECISION)


@dataclass
class CacheEntry:
    key: str
    stage: str
    sample_id: str
    payload: Any
    upstream_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "stage": self.stage,
            "sample_id": self.sample_id,
            "payload": self.payload,
            "upstream_keys": list(self.upstream_keys),
        }

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "CacheEntry":
        return cls(
            key=obj["key"],
            stage=obj["stage"],
            sample_id=obj["sample_id"],
            payload=obj["payload"],
            upstream_keys=list(obj.get("upstream_keys", [])),
        )


class ContentCache:
    """内存缓存 + 可选 JSON 持久化，键完全由内容与参数指纹派生。"""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self._entries: Dict[Tuple[str, str, str], CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        self.reused: List[str] = []
        self.recomputed: List[str] = []
        self.rule_usage: Dict[str, set] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for raw in data.get("entries", []):
            entry = CacheEntry.from_dict(raw)
            self._entries[(entry.sample_id, entry.stage, entry.key)] = entry
        self.rule_usage = {k: set(v) for k, v in data.get("rule_usage", {}).items()}

    def persist(self) -> None:
        if self.path is None:
            return
        data = {
            "entries": [entry.to_dict() for entry in
                        sorted(self._entries.values(),
                               key=lambda e: (e.sample_id, e.stage, e.key))],
            "rule_usage": {k: sorted(v) for k, v in sorted(self.rule_usage.items())},
            "counters": {"hits": self.hits, "misses": self.misses},
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, sample_id: str, stage: str, key: str) -> Optional[CacheEntry]:
        entry = self._entries.get((sample_id, stage, key))
        if entry is None:
            self.misses += 1
            self.recomputed.append(f"{sample_id}:{stage}:{key}")
            return None
        self.hits += 1
        self.reused.append(f"{sample_id}:{stage}:{key}")
        return entry

    def put(self, sample_id: str, stage: str, key: str, payload: Any,
            upstream_keys: Optional[List[str]] = None) -> CacheEntry:
        entry = CacheEntry(key=key, stage=stage, sample_id=sample_id,
                           payload=payload, upstream_keys=list(upstream_keys or []))
        self._entries[(sample_id, stage, key)] = entry
        return entry

    def invalidate_sample(self, sample_id: str) -> List[tuple]:
        """失效单条样本全部阶段产物，返回被失效的完整缓存坐标。"""
        doomed = [k for k in self._entries if k[0] == sample_id]
        for key in doomed:
            self._entries.pop(key, None)
        return doomed

    def invalidate_rule(self, rule_id: str) -> List[tuple]:
        """失效引用该规则的样本的全部下游产物。"""
        sample_ids = self.rule_usage.get(rule_id, set())
        invalidated: List[tuple] = []
        for sample_id in list(sample_ids):
            invalidated.extend(self.invalidate_sample(sample_id))
        return invalidated


    def invalidate_stage(self, stage: str,
                        sample_ids: Optional[List[str]] = None) -> List[tuple]:
        """仅失效指定阶段（及其下游由调用方另行处理）的缓存坐标。"""
        targets = set(sample_ids) if sample_ids else None
        doomed = [key for key in self._entries
                  if key[1] == stage and (targets is None or key[0] in targets)]
        for key in doomed:
            self._entries.pop(key, None)
        return doomed

    def invalidate_stages(self, stages: List[str],
                          sample_ids: Optional[List[str]] = None) -> List[tuple]:
        coords: List[tuple] = []
        for stage in stages:
            coords.extend(self.invalidate_stage(stage, sample_ids))
        return coords

    def record_rule_usage(self, rule_id: str, sample_id: str) -> None:
        self.rule_usage.setdefault(rule_id, set()).add(sample_id)

    def samples_using_rule(self, rule_id: str) -> List[str]:
        return sorted(self.rule_usage.get(rule_id, set()))

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0
        self.reused = []
        self.recomputed = []

    def counters(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "reused": sorted(self.reused),
            "recomputed": sorted(self.recomputed),
            "reused_count": len(self.reused),
            "recomputed_count": len(self.recomputed),
        }
