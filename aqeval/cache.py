"""Content-addressed artifact cache with per-file dependency metadata."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

from .util import atomic_write_json, canonical_json_bytes, read_json, sha256_bytes


CACHE_SCHEMA = "aqeval.cache.v1"


class ArtifactCache:
    def __init__(self, root: str | Path, *, clear: bool = False) -> None:
        self.root = Path(root)
        self.path = self.root / "cache" / "artifacts.json"
        self.blob_root = self.root / "cache" / "blobs"
        if clear and self.root.exists():
            shutil.rmtree(self.root)
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {
            "schema": CACHE_SCHEMA,
            "features": {},
            "dimensions": {},
            "totals": {},
            "serialized": {},
            "meta": {"file_state": {}, "fingerprints": {}},
        }
        if self.path.exists():
            try:
                loaded = read_json(self.path)
                if isinstance(loaded, dict):
                    for section in ("features", "dimensions", "totals", "serialized"):
                        if isinstance(loaded.get(section), dict):
                            self.data[section] = loaded[section]
                    if isinstance(loaded.get("meta"), dict):
                        self.data["meta"].update(loaded["meta"])
            except (OSError, ValueError):
                pass
        self.previous = copy.deepcopy(self.data)

    def get(self, section: str, key: str) -> Any | None:
        value = self.data[section].get(key)
        if value is None:
            return None
        value = copy.deepcopy(value)
        if isinstance(value, dict):
            value = {name: item for name, item in value.items() if not name.startswith("_")}
        return value

    def put(self, section: str, key: str, value: Any) -> None:
        self.data[section][key] = copy.deepcopy(value)

    def previous_file_state(self, file_id: str) -> dict[str, Any]:
        state = self.previous.get("meta", {}).get("file_state", {}).get(file_id, {})
        return state if isinstance(state, dict) else {}

    def set_file_state(self, file_id: str, value: dict[str, Any]) -> None:
        self.data["meta"].setdefault("file_state", {})[file_id] = value

    def set_run_fingerprints(self, value: dict[str, Any]) -> None:
        self.data["meta"]["fingerprints"] = value

    def previous_fingerprints(self) -> dict[str, Any]:
        value = self.previous.get("meta", {}).get("fingerprints", {})
        return value if isinstance(value, dict) else {}

    def store_blob(self, data: bytes) -> str:
        digest = sha256_bytes(data)
        path = self.blob_root / digest
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return digest

    def load_blob(self, digest: str) -> bytes:
        return (self.blob_root / digest).read_bytes()

    def save(self) -> None:
        atomic_write_json(self.path, self.data)

    def has_previous_entry(self, section: str, key: str) -> bool:
        return key in self.previous.get(section, {})

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.data))
