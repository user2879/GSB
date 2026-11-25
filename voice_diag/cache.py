"""Content-addressed artifact cache and deterministic reuse audit."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .fingerprint import canonical_json, sha256_bytes, sha256_json, write_json_atomic

CACHE_SCHEMA = 1


class ContentCache:
    def __init__(self, root: Path, *, read_enabled: bool = True):
        self.root = root
        self.entries_dir = root / "entries"
        self.read_enabled = read_enabled
        self._memory: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        if read_enabled:
            self.entries_dir.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, key: str) -> Path:
        return self.entries_dir / f"{sha256_text_key(key)}.json"

    def get(self, stage: str, key: str, *, references: list[str] | None = None) -> Any | None:
        if not self.read_enabled:
            self._record(stage, key, "miss", "full_rebuild_or_cache_read_disabled", references)
            return None
        if key in self._memory:
            self._record(stage, key, "reuse", "memory_content_key_hit", references)
            return self._memory[key]
        path = self._entry_path(key)
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("cache_schema") == CACHE_SCHEMA and payload.get("cache_key") == key:
                    self._memory[key] = payload["value"]
                    self._record(stage, key, "reuse", "persistent_content_key_hit", references)
                    return payload["value"]
            except (OSError, json.JSONDecodeError):
                pass
        self._record(stage, key, "miss", "content_key_absent_or_unreadable", references)
        return None

    def put(self, stage: str, key: str, value: Any, *, references: list[str] | None = None) -> Any:
        self._memory[key] = value
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self._entry_path(key),
            {"cache_schema": CACHE_SCHEMA, "cache_key": key, "value": value},
            indent=None,
        )
        self._record(stage, key, "computed", "content_key_materialized", references)
        return value

    def _record(self, stage: str, key: str, status: str, basis: str, references: list[str] | None) -> None:
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "stage": stage,
                "cache_key": key,
                "status": status,
                "basis": basis,
                "references": sorted(references or []),
            }
        )

    def clear(self) -> None:
        if self.entries_dir.exists():
            shutil.rmtree(self.entries_dir)
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self._memory.clear()


def sha256_text_key(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def summarize_cache_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    reuse = sum(1 for event in events if event["status"] == "reuse")
    misses = sum(1 for event in events if event["status"] == "miss")
    computed = sum(1 for event in events if event["status"] == "computed")
    by_stage: dict[str, dict[str, int]] = {}
    for event in events:
        bucket = by_stage.setdefault(event["stage"], {"reuse": 0, "miss": 0, "computed": 0})
        bucket[event["status"]] += 1
    return {
        "event_count": len(events),
        "reuse_hit_count": reuse,
        "miss_count": misses,
        "computed_count": computed,
        "by_stage": [
            {"stage": stage, **by_stage[stage]} for stage in sorted(by_stage)
        ],
    }
