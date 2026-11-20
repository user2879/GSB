"""Content-fingerprint cache with exact invalidation.

Every stage artifact is stored under a cache key derived from upstream
content fingerprints plus the parameter fingerprint. Changing a method
hyperparameter, the preprocess method, or a single sample therefore
invalidates exactly the affected candidates and their downstream
evaluations; untouched candidates hit the cache and are never re-iterated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .fingerprint import fingerprint, _normalize


def preprocess_key(data_fp: str, params_fp: str) -> str:
    return fingerprint("stage:preprocess", data_fp, params_fp)


def cluster_key(pre_key: str, method: str, params: dict) -> str:
    return fingerprint("stage:cluster", pre_key, method, params)


def evaluate_key(cluster_k: str, metrics: list, reg_covar: float) -> str:
    return fingerprint("stage:evaluate", cluster_k, list(metrics),
                       float(reg_covar))


def diagnostics_key(eval_keys: list[str], extra: dict) -> str:
    return fingerprint("stage:diagnostics", sorted(eval_keys), extra)


class ArtifactCache:
    """File-backed cache: one JSON document per key, atomically replaced."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        with open(p, "r", encoding="utf-8") as fh:
            self.hits += 1
            return json.load(fh)

    def put(self, key: str, value: Any) -> None:
        p = self._path(key)
        tmp = self.root / f".{key}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_normalize(value), fh, sort_keys=True, ensure_ascii=True)
        os.replace(tmp, p)  # atomic whole-file replace

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses}
