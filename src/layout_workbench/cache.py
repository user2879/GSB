from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, digest, read_json

PIPELINE_CACHE_VERSION = "layout-pipeline-2026-11-20.1"


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}


@dataclass
class CacheStore:
    root: Path
    memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)

    def path_for(self, key: str) -> Path:
        return self.root / "objects" / key[:2] / f"{key}.json"

    def exists(self, key: str) -> bool:
        return key in self.memory or self.path_for(key).is_file()

    def get(self, key: str) -> dict[str, Any] | None:
        if key in self.memory:
            self.stats.hits += 1
            return self.memory[key]
        path = self.path_for(key)
        if not path.is_file():
            self.stats.misses += 1
            return None
        try:
            artifact = read_json(path)
        except (OSError, ValueError):
            self.stats.misses += 1
            return None
        if not isinstance(artifact, dict) or artifact.get("cache_key") != key:
            self.stats.misses += 1
            return None
        if artifact.get("payload_digest") != digest(artifact.get("payload")):
            self.stats.misses += 1
            return None
        self.memory[key] = artifact
        self.stats.hits += 1
        return artifact

    def put(
        self,
        *,
        stage: str,
        key: str,
        image: str,
        dependencies: dict[str, Any],
        payload: dict[str, Any],
        font_version: str,
    ) -> dict[str, Any]:
        artifact = {
            "schema": "layout-cache-v2",
            "cache_key": key,
            "stage": stage,
            "image": image,
            "dependencies": dependencies,
            "font_version": font_version,
            "pipeline_cache_version": PIPELINE_CACHE_VERSION,
            "payload_digest": digest(payload),
            "payload": payload,
        }
        atomic_write_json(self.path_for(key), artifact)
        self.memory[key] = artifact
        self.stats.writes += 1
        return artifact


def cache_key(stage: str, dependency_values: dict[str, Any]) -> str:
    return digest(
        {
            "stage": stage,
            "pipeline_cache_version": PIPELINE_CACHE_VERSION,
            "dependencies": dependency_values,
        }
    )
