from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .fingerprint import sha256_value
from .snapshot import atomic_write_json


class CacheStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.by_column: dict[str, dict[str, str]] = {}

    def record(self, stage: str, column: str, hit: bool) -> None:
        with self._lock:
            if hit:
                self.hits += 1
            else:
                self.misses += 1
            self.by_column.setdefault(column, {})[stage] = "hit" if hit else "miss"

    def record_stage_reuse(self, stage: str, columns: list[str]) -> None:
        with self._lock:
            for column in columns:
                self.hits += 1
                self.by_column.setdefault(column, {})[stage] = "hit"

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "by_column": {
                    column: {stage: state for stage, state in sorted(stages.items())}
                    for column, stages in sorted(self.by_column.items())
                },
            }


class ArtifactCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "cache"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, stage: str, key: str) -> Path:
        return self.root / stage / key[:2] / f"{key}.json"

    def load(self, stage: str, key: str) -> dict[str, Any] | None:
        path = self._path(stage, key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                wrapper = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(wrapper, dict):
            return None
        if wrapper.get("stage") != stage or wrapper.get("key") != key:
            return None
        payload = wrapper.get("payload")
        if wrapper.get("payload_sha256") != sha256_value(payload):
            return None
        return payload

    def store(self, stage: str, key: str, payload: dict[str, Any]) -> None:
        path = self._path(stage, key)
        wrapper = {
            "format_version": 1,
            "stage": stage,
            "key": key,
            "payload_sha256": sha256_value(payload),
            "payload": payload,
        }
        atomic_write_json(path, wrapper, indent=None)
