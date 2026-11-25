"""Content-fingerprinted persistent product cache.

Each stage product is stored under a cache key derived from its upstream
content fingerprints, the parameter fingerprint and the quantization format.
The cache is a JSON document on disk, so reuse hits survive process restarts and
can be audited independently of the in-memory engine.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional

from .fingerprint import digest


class ProductCache:
    """Persistent key -> serializable product mapping with atomic writes."""

    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, Any] = {}
        self._loaded = False

    def load(self) -> "ProductCache":
        if os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8") as handle:
                self._data = json.load(handle)
        else:
            self._data = {}
        self._loaded = True
        return self

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def has(self, key: str) -> bool:
        return key in self._data

    def put(self, key: str, product: Dict[str, Any]) -> None:
        self._data[key] = product

    def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    def keys(self):
        return list(self._data.keys())

    def flush(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".cache-", suffix=".json",
                                   dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def fingerprint(self) -> str:
        return digest(["cache", self._data])

    def __len__(self) -> int:
        return len(self._data)
