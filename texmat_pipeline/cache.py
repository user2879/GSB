"""Content-fingerprint cache and precise invalidation planning.

Cache keys are derived from upstream content fingerprints plus a parameter
fingerprint (see :mod:`texmat_pipeline.hashing`).  The store keeps an atomic
JSON index and one ``.npy`` payload per key.  Lineage edges recorded alongside
every entry allow an edit (texture replacement, layer parameter change, blend
mode or bit-width switch) to invalidate exactly the affected subgraph and its
downstream nodes while everything else is reused.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CacheEntry:
    key: str
    kind: str
    upstream: list[str]
    params: dict[str, Any]
    payload: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "upstream": list(self.upstream),
            "params": self.params,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        return cls(data["key"], data["kind"], list(data["upstream"]), data["params"], data["payload"])


class CacheStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, CacheEntry] = {}
        if self.index_path.exists():
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._index = {k: CacheEntry.from_dict(v) for k, v in raw.items()}

    def has(self, key: str) -> bool:
        entry = self._index.get(key)
        return entry is not None and (self.root / entry.payload).exists()

    def load_array(self, key: str) -> np.ndarray:
        entry = self._index[key]
        return np.load(self.root / entry.payload)

    def load_meta(self, key: str) -> CacheEntry:
        return self._index[key]

    def all_keys(self) -> set[str]:
        return {k for k in self._index if self.has(k)}

    def lineage(self) -> dict[str, list[str]]:
        return {k: list(e.upstream) for k, e in self._index.items() if self.has(k)}

    def store_array(self, key: str, arr: np.ndarray, kind: str,
                    upstream: list[str], params: dict[str, Any]) -> None:
        rel = f"data/{key}.npy"
        payload = self.root / rel
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".npy", dir=str(self.data_dir))
        os.close(tmp_fd)
        try:
            with open(tmp_path, "wb") as fh:
                np.save(fh, np.ascontiguousarray(arr))
            os.replace(tmp_path, payload)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        self._index[key] = CacheEntry(key, kind, list(upstream), params, rel)
        self._flush_index()

    def _flush_index(self) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(self.root))
        os.close(tmp_fd)
        try:
            payload = {k: e.to_dict() for k, e in sorted(self._index.items())}
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, sort_keys=True, indent=1)
            os.replace(tmp_path, self.index_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def downstream(edges: dict[str, list[str]], roots: set[str]) -> set[str]:
    """Return every node reachable from ``roots`` following upstream->node edges."""
    children: dict[str, list[str]] = {}
    for node, ups in edges.items():
        for up in ups:
            children.setdefault(up, []).append(node)
    affected: set[str] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        for child in children.get(node, []):
            if child not in affected:
                affected.add(child)
                stack.append(child)
    return affected


def plan_reuse(desired: set[str], store: CacheStore) -> tuple[list[str], list[str]]:
    """Split desired cache keys into (reusable, recompute) lists, sorted."""
    reusable = sorted(k for k in desired if store.has(k))
    recompute = sorted(k for k in desired if k not in reusable)
    return reusable, recompute
