"""Content-addressed stage cache."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Any

from .hashing import sha256_json

STAGE_SCHEMA = "pose3d-stage-cache/1"


def make_stage_key(stage: str, previous_key: str | None, material: dict[str, Any]) -> str:
    return sha256_json(
        {
            "schema": STAGE_SCHEMA,
            "stage": stage,
            "previous_key": previous_key,
            "material": material,
        }
    )


@dataclass
class StageCache:
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    used_keys: set[str] = field(default_factory=set)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def from_serialized(cls, payload: Any) -> "StageCache":
        if not isinstance(payload, dict):
            return cls()
        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            return cls()
        clean = {
            str(key): value
            for key, value in entries.items()
            if isinstance(value, dict)
            and value.get("schema") == STAGE_SCHEMA
            and isinstance(value.get("payload"), dict)
        }
        return cls(entries=clean)

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self.entries.get(key)
            if entry is None:
                return None
            self.used_keys.add(key)
            return copy.deepcopy(entry["payload"])

    def put(self, key: str, stage: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self.used_keys.add(key)
            self.entries[key] = {
                "schema": STAGE_SCHEMA,
                "stage": stage,
                "payload": copy.deepcopy(payload),
            }

    def contains(self, key: str) -> bool:
        return key in self.entries

    def serialize(self) -> dict[str, Any]:
        # Only entries reachable from the current input snapshot are persisted.
        # This makes a cached snapshot byte-comparable with a fresh full build.
        with self._lock:
            keys = sorted(self.used_keys)
            return {
                "schema": STAGE_SCHEMA,
                "entries": {key: copy.deepcopy(self.entries[key]) for key in keys if key in self.entries},
            }
