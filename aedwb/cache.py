"""Content-addressed packet-stage object cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .util import read_json, write_json_atomic


class ObjectCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not key or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("cache key must be a lowercase hexadecimal digest")
        return self.root / key[:2] / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or value.get("cache_key") != key:
            return None
        payload = value.get("payload")
        return payload if isinstance(payload, dict) else None

    def store(self, key: str, stage: str, payload: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            path,
            {"schema_version": "aed-cache-object/1", "cache_key": key, "stage": stage, "payload": payload},
        )
