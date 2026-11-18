"""Content-fingerprint artifact cache with precise invalidation.

Each stage artifact is stored under a derived cache key combining the upstream
sample fingerprint, config fingerprint, caliber version and scope. Every group
artifact also records the set of request IDs (event set) it was computed from,
so that a new batch or a single-event correction only invalidates the affected
groups while untouched artifacts are reused byte-for-byte.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .fingerprint import CALIBER_VERSION, derive_cache_key, fingerprint_json

STAGE_BUCKETING = "bucketing"
STAGE_STATS = "stats"
STAGE_ESTIMATION = "estimation"
STAGE_ATTRIBUTION = "attribution"
STAGES = (STAGE_BUCKETING, STAGE_STATS, STAGE_ESTIMATION, STAGE_ATTRIBUTION)


@dataclass
class CacheEntry:
    key: str
    stage: str
    scope: str
    payload: Any
    event_ids: List[str] = field(default_factory=list)
    upstream: str = ""
    config_fp: str = ""
    caliber: str = CALIBER_VERSION

    def to_dict(self):
        return {
            "key": self.key,
            "stage": self.stage,
            "scope": self.scope,
            "payload": self.payload,
            "event_ids": sorted(self.event_ids),
            "upstream": self.upstream,
            "config_fp": self.config_fp,
            "caliber": self.caliber,
        }


class ArtifactCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, "artifact_cache.json")
        self.entries: Dict[str, CacheEntry] = {}
        self.reused: List[str] = []
        self.invalidated: List[str] = []
        self.computed: List[str] = []
        self._load()
        self.reused = []
        self.invalidated = []
        self.computed = []

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for item in raw.get("entries", []):
            self.entries[item["key"]] = CacheEntry(
                key=item["key"],
                stage=item["stage"],
                scope=item["scope"],
                payload=item["payload"],
                event_ids=list(item.get("event_ids", [])),
                upstream=item.get("upstream", ""),
                config_fp=item.get("config_fp", ""),
                caliber=item.get("caliber", CALIBER_VERSION),
            )

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(
                {"entries": [e.to_dict() for _k, e in sorted(self.entries.items())]},
                handle,
                sort_keys=True,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp, self.path)

    def make_key(
        self, stage: str, upstream: str, config_fp: str, scope: str = ""
    ) -> str:
        return derive_cache_key(stage, upstream, config_fp, CALIBER_VERSION, scope)

    def get(self, key: str) -> Optional[CacheEntry]:
        entry = self.entries.get(key)
        if entry is not None:
            self.reused.append(key)
        return entry

    def put(
        self,
        stage: str,
        scope: str,
        upstream: str,
        config_fp: str,
        payload: Any,
        event_ids: Optional[Set[str]] = None,
    ) -> str:
        key = self.make_key(stage, upstream, config_fp, scope)
        if key in self.entries:
            self.reused.append(key)
            return key
        self.entries[key] = CacheEntry(
            key=key,
            stage=stage,
            scope=scope,
            payload=payload,
            event_ids=sorted(event_ids or []),
            upstream=upstream,
            config_fp=config_fp,
        )
        self.computed.append(key)
        return key

    def invalidate_groups(self, affected_requests: Set[str], config_fp: str):
        """Drop entries whose recorded event set intersects the edited requests."""
        removed: List[str] = []
        for key, entry in list(self.entries.items()):
            if entry.stage == STAGE_BUCKETING:
                continue
            if entry.config_fp != config_fp:
                removed.append(key)
                del self.entries[key]
                continue
            if entry.event_ids and set(entry.event_ids) & affected_requests:
                removed.append(key)
                del self.entries[key]
        self.invalidated.extend(removed)
        return sorted(removed)

    def invalidate_on_config_change(self, new_config_fp: str):
        removed: List[str] = []
        for key, entry in list(self.entries.items()):
            if entry.config_fp != new_config_fp:
                removed.append(key)
                del self.entries[key]
        self.invalidated.extend(removed)
        return sorted(removed)

    def clear_all(self):
        removed = sorted(self.entries)
        self.entries.clear()
        self.invalidated.extend(removed)
        return removed

    def reset_run_log(self):
        self.reused = []
        self.computed = []
        self.invalidated = []

    def change_report(self) -> Dict[str, List[str]]:
        return {
            "reused_keys": sorted(set(self.reused)),
            "computed_keys": sorted(set(self.computed)),
            "invalidated_keys": sorted(set(self.invalidated)),
        }

    def prune_stage_scope(self, stage: str, scopes, valid_keys) -> List[str]:
        """Invalidate entries of a stage/scope set whose key is no longer referenced."""
        valid = set(valid_keys)
        scopes = set(scopes)
        removed: List[str] = []
        for key, entry in list(self.entries.items()):
            if entry.stage != stage or entry.scope not in scopes:
                continue
            if key not in valid:
                removed.append(key)
                del self.entries[key]
        self.invalidated.extend(removed)
        return sorted(removed)

    def fingerprint(self) -> str:
        return fingerprint_json(
            [e.to_dict() for _k, e in sorted(self.entries.items())]
        )
