from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .util import atomic_write_json, is_safe_id


@dataclass(frozen=True)
class CacheLookup:
    value: Any
    reused: bool
    key: str


class ContentCache:
    """Content-addressed cache with deterministic in-process behavior.

    The cache is deliberately a performance artifact, never a semantic input.
    A cache miss and a cache hit must produce the same value for the same key.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).resolve() if root is not None else None
        self._memory: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()

    def _path(self, namespace: str, key: str) -> Path:
        if not is_safe_id(namespace) or not all(char in "0123456789abcdef" for char in key) or len(key) != 64:
            raise ValueError("invalid cache namespace or key")
        return self.root / namespace / key[:2] / f"{key}.json"

    def get_or_compute(self, namespace: str, key: str, factory: Callable[[], Any]) -> CacheLookup:
        mem_key = (namespace, key)
        with self._lock:
            if mem_key in self._memory:
                return CacheLookup(self._memory[mem_key], True, key)
        if self.root is not None:
            path = self._path(namespace, key)
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if payload.get("namespace") == namespace and payload.get("key") == key:
                        value = payload["value"]
                        with self._lock:
                            self._memory[mem_key] = value
                        return CacheLookup(value, True, key)
                except (OSError, ValueError, KeyError, TypeError):
                    pass
        value = factory()
        with self._lock:
            self._memory[mem_key] = value
        if self.root is not None:
            atomic_write_json(
                self._path(namespace, key),
                {"namespace": namespace, "key": key, "value": value},
            )
        return CacheLookup(value, False, key)

    def put(self, namespace: str, key: str, value: Any) -> None:
        with self._lock:
            self._memory[(namespace, key)] = value
        if self.root is not None:
            atomic_write_json(
                self._path(namespace, key),
                {"namespace": namespace, "key": key, "value": value},
            )

