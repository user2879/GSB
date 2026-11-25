from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .config import canonical_hash
from .ledger import atomic_write_json


def derive_key(stage: str, *parts: Any) -> str:
    return f"{stage}:" + canonical_hash({"stage": stage, "parts": parts})[:24]


def array_region_fingerprint(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
    region = gray[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    h = hashlib.sha256()
    h.update(str(region.shape).encode())
    h.update(region.tobytes())
    return h.hexdigest()


class ContentCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if self.path.exists():
            raw = __import__("json").loads(self.path.read_text(encoding="utf-8"))
        else:
            raw = {"entries": {}, "scopes": {}}
        self.entries: dict[str, dict[str, Any]] = raw.get("entries", {})
        self.scopes: dict[str, str] = raw.get("scopes", {})
        self.events: list[dict[str, Any]] = []

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self.entries.get(key)
        return entry["product"] if entry else None

    def hit(self, stage: str, scope: str, key: str, basis: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {"status": "hit", "stage": stage, "scope": scope, "cache_key": key, "basis": basis or {}}
        self.events.append(event)
        return event

    def check_scope_invalidation(self, scope: str, key: str, reason: str, basis: dict[str, Any]) -> None:
        old_key = self.scopes.get(scope)
        if old_key and old_key != key:
            self.events.append({
                "status": "invalidated",
                "stage": basis.get("stage", "unknown"),
                "scope": scope,
                "old_cache_key": old_key,
                "new_cache_key": key,
                "reason": reason,
                "basis": basis,
            })

    def put(
        self,
        stage: str,
        scope: str,
        key: str,
        product: Any,
        reason: str = "new_or_changed_dependency",
        basis: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            old_key = self.scopes.get(scope)
            invalidated = bool(old_key and old_key != key)
            self.entries[key] = {"stage": stage, "scope": scope, "product": product}
            self.scopes[scope] = key
            event = {
                "status": "recomputed",
                "stage": stage,
                "scope": scope,
                "cache_key": key,
                "reason": reason if not invalidated else "dependency_changed",
                "invalidated_previous": invalidated,
                "old_cache_key": old_key if invalidated else None,
                "basis": basis or {},
            }
            self.events.append(event)
            atomic_write_json(self.path, {"entries": self.entries, "scopes": self.scopes})

    def save(self) -> None:
        with self._lock:
            atomic_write_json(self.path, {"entries": self.entries, "scopes": self.scopes})

    def counts(self) -> dict[str, int]:
        return {
            "hit": sum(1 for e in self.events if e["status"] == "hit"),
            "recomputed": sum(1 for e in self.events if e["status"] == "recomputed"),
            "invalidated": sum(1 for e in self.events if e["status"] == "invalidated"),
        }
