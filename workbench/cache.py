"""Content-fingerprint stage cache keys and intent-level precise invalidation."""
from __future__ import annotations

from dataclasses import dataclass, field

from .io_utils import fingerprint


def stage_cache_key(utterance_id: str, stage: str, parts: list) -> str:
    payload = {"utterance_id": utterance_id, "stage": stage, "parts": parts}
    return fingerprint(payload)


def task_sort_key(utterance_id: str, stage: str) -> str:
    return fingerprint({"task": utterance_id, "stage": stage})


@dataclass
class StageCache:
    """Maps cache key -> (value, owning utterance).

    Keeping the owning utterance alongside each value lets invalidation drop
    entries by identity even after a config switch changes the derived keys.
    """

    entries: dict = field(default_factory=dict)

    def get(self, key: str):
        entry = self.entries.get(key)
        return None if entry is None else entry["value"]

    def put(self, key: str, value, utterance_id: str) -> None:
        self.entries[key] = {"value": value, "utterance_id": utterance_id,
                             "key": key}

    def invalidate_keys(self, keys) -> None:
        for key in keys:
            self.entries.pop(key, None)

    def invalidate_utterance(self, utterance_id: str) -> None:
        dead = [key for key, entry in self.entries.items()
                if entry["utterance_id"] == utterance_id]
        for key in dead:
            self.entries.pop(key, None)

    def keys_for(self, utterance_id: str):
        return [key for key, entry in self.entries.items()
                if entry["utterance_id"] == utterance_id]

    def clear(self) -> None:
        self.entries.clear()


@dataclass
class CacheLedger:
    """Holds stage caches plus the reverse index needed for precise edits."""

    domain_cache: StageCache = field(default_factory=StageCache)
    intent_cache: StageCache = field(default_factory=StageCache)
    slot_cache: StageCache = field(default_factory=StageCache)
    recal_cache: StageCache = field(default_factory=StageCache)
    intent_index: dict = field(default_factory=dict)

    def stage(self, name: str) -> StageCache:
        return getattr(self, f"{name}_cache")

    def record_reference(self, intent: str, utterance_id: str, entries: list,
                         domain: str) -> None:
        bucket = self.intent_index.setdefault(intent, {
            "utterances": set(), "trigger_entries": [], "domains": set()})
        bucket["utterances"].add(utterance_id)
        bucket["domains"].add(domain)
        for entry in entries:
            record = {"trigger": entry["trigger"], "negator": entry["negator"]}
            if record not in bucket["trigger_entries"]:
                bucket["trigger_entries"].append(record)

    def referenced_utterances(self, intents) -> set:
        affected = set()
        for intent in intents:
            bucket = self.intent_index.get(intent)
            if bucket:
                affected.update(bucket["utterances"])
        return affected

    def drop_intent(self, intent: str) -> None:
        self.intent_index.pop(intent, None)

    def snapshot_index(self) -> dict:
        return {
            intent: {
                "utterances": sorted(bucket["utterances"]),
                "trigger_entries": bucket["trigger_entries"],
                "domains": sorted(bucket["domains"]),
            }
            for intent, bucket in sorted(self.intent_index.items())
        }
