"""Content-addressed cache enabling exact, dependency-driven invalidation.

Entries are keyed by the explicit cache keys produced at each layer:
window fingerprints, spectrum position keys, per-detector score keys and
verdict keys.  A single changed window only changes spectrum keys whose
ordered window dependency sets contain it, so untouched positions are reused
and never recomputed.
"""

import json
import os
from pathlib import Path


class ContentCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._memo = {}
        self.hits = 0
        self.misses = 0
        self.computed = {"window": 0, "spectrum": 0, "detector": 0,
                         "verdict": 0}

    def _path(self, layer, key):
        return self.root / layer / f"{key}.json"

    def get(self, layer, key):
        cache_key = (layer, key)
        if cache_key in self._memo:
            self.hits += 1
            return self._memo[cache_key]
        path = self._path(layer, key)
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                value = json.load(fh)
            self._memo[cache_key] = value
            self.hits += 1
            return value
        self.misses += 1
        return None

    def put(self, layer, key, value):
        self._memo[(layer, key)] = value
        path = self._path(layer, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(value, fh, sort_keys=True, separators=(",", ":"))
        os.replace(tmp, path)

    def compute(self, layer, key, producer):
        cached = self.get(layer, key)
        if cached is not None:
            return cached, True
        value = producer()
        self.computed[layer] = self.computed.get(layer, 0) + 1
        self.put(layer, key, value)
        return value, False

    def stats(self):
        return {"hits": self.hits, "misses": self.misses,
                "computed": dict(self.computed)}
