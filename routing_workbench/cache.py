"""Content-addressed stage-product cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .util import fingerprint, jsonable


STAGES = ("validation", "domain", "intent", "slots", "recalibration")


@dataclass
class CacheEntry:
    stage: str
    key: str
    product: Any
    references: dict[str, Any] = field(default_factory=dict)
    utterance_ids: list[str] = field(default_factory=list)
    digest: str = ""

    def finalize(self) -> "CacheEntry":
        if not self.digest:
            self.digest = fingerprint({"stage": self.stage, "key": self.key, "product": self.product, "references": self.references})
        self.utterance_ids = sorted(set(self.utterance_ids))
        return self

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "stage": self.stage,
            "key": self.key,
            "product": jsonable(self.product),
            "references": jsonable(self.references),
            "utterance_ids": list(self.utterance_ids),
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CacheEntry":
        entry = cls(
            stage=str(raw["stage"]),
            key=str(raw["key"]),
            product=raw.get("product"),
            references=raw.get("references", {}),
            utterance_ids=[str(item) for item in raw.get("utterance_ids", [])],
            digest=str(raw.get("digest", "")),
        )
        return entry.finalize()


class StageCache:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, CacheEntry]] = {stage: {} for stage in STAGES}
        self.reused: dict[str, set[str]] = {stage: set() for stage in STAGES}
        self.recomputed: dict[str, set[str]] = {stage: set() for stage in STAGES}
        self.pending: dict[str, set[str]] = {stage: set() for stage in STAGES}

    def clear_stats(self) -> None:
        self.reused = {stage: set() for stage in STAGES}
        self.recomputed = {stage: set() for stage in STAGES}
        self.pending = {stage: set() for stage in STAGES}

    def get(self, stage: str, key: str) -> CacheEntry | None:
        return self.entries.get(stage, {}).get(key)

    def put(self, stage: str, key: str, product: Any, references: dict[str, Any] | None = None, utterance_ids: list[str] | None = None) -> CacheEntry:
        entry = CacheEntry(
            stage=stage,
            key=key,
            product=jsonable(product),
            references=jsonable(references or {}),
            utterance_ids=list(utterance_ids or []),
        ).finalize()
        self.entries.setdefault(stage, {})[key] = entry
        return entry

    def mark_reuse(self, stage: str, utterance_id: str) -> None:
        self.reused.setdefault(stage, set()).add(utterance_id)

    def mark_recompute(self, stage: str, utterance_id: str) -> None:
        self.recomputed.setdefault(stage, set()).add(utterance_id)

    def mark_pending(self, stage: str, utterance_id: str) -> None:
        self.pending.setdefault(stage, set()).add(utterance_id)

    def stage_versions(self, previous: dict[str, Any] | None = None) -> dict[str, Any]:
        versions: dict[str, Any] = {}
        previous = previous or {}
        for stage in STAGES:
            entries = self.entries.get(stage, {})
            digest = fingerprint(
                [
                    {"key": entry.key, "digest": entry.digest}
                    for entry in sorted(entries.values(), key=lambda item: item.key)
                ]
            )
            old = previous.get(stage, {})
            revision = int(old.get("revision", 0))
            if old.get("digest") != digest or not old:
                revision += 1
            versions[stage] = {
                "digest": digest,
                "revision": revision,
                "entries": len(entries),
            }
        return versions

    def to_stage_payload(self) -> dict[str, list[dict[str, Any]]]:
        return {
            stage: [entry.to_dict() for entry in sorted(self.entries.get(stage, {}).values(), key=lambda item: item.key)]
            for stage in STAGES
        }

    @classmethod
    def from_stage_payload(cls, payload: dict[str, Any]) -> "StageCache":
        cache = cls()
        for stage in STAGES:
            for raw_entry in payload.get(stage, []):
                try:
                    entry = CacheEntry.from_dict(raw_entry)
                except (KeyError, TypeError, ValueError):
                    continue
                cache.entries[stage][entry.key] = entry
        return cache

    def reuse_recompute(self, only_ids: set[str] | None = None) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        summary: dict[str, Any] = {}
        for stage in STAGES:
            ids = set(self.reused.get(stage, set())) | set(self.recomputed.get(stage, set())) | set(self.pending.get(stage, set()))
            if only_ids is not None:
                ids &= only_ids
            stage_rows = 0
            for utterance_id in sorted(ids):
                if utterance_id in self.pending.get(stage, set()):
                    status = "pending"
                elif utterance_id in self.recomputed.get(stage, set()):
                    status = "recompute"
                else:
                    status = "reuse"
                rows.append({"utterance_id": utterance_id, "stage": stage, "status": status})
                stage_rows += 1
            summary[stage] = {
                "reused": len(self.reused.get(stage, set())),
                "recomputed": len(self.recomputed.get(stage, set())),
                "pending": len(self.pending.get(stage, set())),
            }
        rows.sort(key=lambda item: (item["utterance_id"], STAGES.index(item["stage"]), item["status"]))
        return {"rows": rows, "summary": summary}

    def keys_for(self, utterance_id: str, result_stage_keys: dict[str, str] | None = None) -> dict[str, str]:
        if result_stage_keys:
            return dict(sorted(result_stage_keys.items()))
        keys: dict[str, str] = {}
        for stage in STAGES:
            for entry in self.entries.get(stage, {}).values():
                if utterance_id in entry.utterance_ids:
                    keys[stage] = entry.key
                    break
        return keys

