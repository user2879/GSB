"""Content-addressed JSON artifact cache with committed-inventory gating."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, file_sha256


class CacheStore:
    def __init__(self, root: Path):
        self.root = root
        self.objects_root = root / "objects"
        self.objects_root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        return self.objects_root / namespace / f"{key}.json"

    def get(
        self,
        namespace: str,
        key: str,
        committed: set[tuple[str, str]] | dict[tuple[str, str], str],
    ) -> dict[str, Any] | None:
        if (namespace, key) not in committed:
            return None
        path = self._path(namespace, key)
        if not path.exists():
            return None
        expected_digest = committed.get((namespace, key)) if isinstance(committed, dict) else None
        if expected_digest and file_sha256(path) != expected_digest:
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def put(self, namespace: str, key: str, value: dict[str, Any]) -> dict[str, str]:
        path = self._path(namespace, key)
        atomic_write_json(path, value)
        return {
            "namespace": namespace,
            "key": key,
            "sha256": file_sha256(path),
        }

    def has_committed(
        self,
        namespace: str,
        key: str,
        committed: set[tuple[str, str]] | dict[tuple[str, str], str],
    ) -> bool:
        return (namespace, key) in committed and self._path(namespace, key).exists()
