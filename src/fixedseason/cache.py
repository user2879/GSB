"""Persistent content-addressed stage cache."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fingerprints import CACHE_STAGES
from .jsonutil import fingerprint, read_json, write_json_atomic

_KEY_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    value: Any
    reason: str


class CacheStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, stage: str, key: str) -> Path:
        if stage not in CACHE_STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        if not _KEY_RE.fullmatch(key):
            raise ValueError(f"invalid cache key {key!r}")
        return self.root / stage / f"{key}.json"

    def get(self, stage: str, key: str) -> CacheLookup:
        path = self._path(stage, key)
        if not path.exists():
            return CacheLookup(False, None, "cache_absent")
        try:
            payload = read_json(path)
            if payload.get("schema_version") != 1:
                return CacheLookup(False, None, "cache_schema_mismatch")
            if payload.get("stage") != stage or payload.get("key") != key:
                return CacheLookup(False, None, "cache_identity_mismatch")
            value = payload.get("value")
            if fingerprint(value) != payload.get("value_fingerprint"):
                return CacheLookup(False, None, "cache_checksum_mismatch")
            return CacheLookup(True, value, "cache_hit")
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return CacheLookup(False, None, "cache_unreadable_or_corrupt")

    def put(self, stage: str, key: str, value: Any) -> str:
        path = self._path(stage, key)
        payload = {
            "schema_version": 1,
            "stage": stage,
            "key": key,
            "value_fingerprint": fingerprint(value),
            "value": value,
        }
        write_json_atomic(path, payload)
        return payload["value_fingerprint"]

    def delete(self, stage: str, key: str) -> bool:
        path = self._path(stage, key)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError:
            return False

    def all_stage_hits(self, keys: dict[str, str]) -> tuple[bool, dict[str, CacheLookup]]:
        lookups = {stage: self.get(stage, keys[stage]) for stage in CACHE_STAGES}
        return all(lookup.hit for lookup in lookups.values()), lookups
