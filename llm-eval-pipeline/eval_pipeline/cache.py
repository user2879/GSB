"""Content-fingerprint cache with exact invalidation and reuse accounting.

Each artifact key is derived from its upstream content fingerprint, a
parameter fingerprint and the scale-tier identifier, so changing any of
those inputs changes the key and leaves every unaffected entry reusable.
"""

from __future__ import annotations

import os

from .util import atomic_write_json, keyed_fingerprint, read_json


class FingerprintCache:
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.reuse_count = 0
        self.recompute_count = 0
        self.invalidated_count = 0
        self.reused_keys = []
        self.computed_keys = []
        self._known_keys = set()
        self._scan()

    def _entry_path(self, key: str) -> str:
        return os.path.join(self.directory, f"{key}.json")

    def _scan(self) -> None:
        if not os.path.isdir(self.directory):
            return
        for name in os.listdir(self.directory):
            if name.endswith(".json"):
                self._known_keys.add(name[:-5])

    @staticmethod
    def derive_key(stage: str, content_fp: str, params_fp: str, tier_id: str) -> str:
        return f"{stage}_{keyed_fingerprint([content_fp, params_fp, tier_id])}"

    def get_or_compute(self, stage: str, content_fp: str, params_fp: str, tier_id: str, compute):
        key = self.derive_key(stage, content_fp, params_fp, tier_id)
        path = self._entry_path(key)
        if key in self._known_keys and os.path.exists(path):
            payload = read_json(path)
            self.reuse_count += 1
            self.reused_keys.append(key)
            return payload["artifact"], True
        artifact = compute()
        atomic_write_json(path, {"key": key, "stage": stage, "artifact": artifact})
        self._known_keys.add(key)
        self.recompute_count += 1
        self.computed_keys.append(key)
        return artifact, False

    def invalidate_prefixes(self, prefixes: list) -> int:
        """Delete entries whose stage prefix is listed (downstream invalidation)."""
        removed = 0
        for key in list(self._known_keys):
            if any(key.startswith(prefix + "_") for prefix in prefixes):
                path = self._entry_path(key)
                if os.path.exists(path):
                    os.unlink(path)
                self._known_keys.discard(key)
                removed += 1
        self.invalidated_count += removed
        return removed

    def stats(self) -> dict:
        return {
            "reuse_count": self.reuse_count,
            "recompute_count": self.recompute_count,
            "invalidated_count": self.invalidated_count,
            "reused_keys": sorted(self.reused_keys),
            "computed_keys": sorted(self.computed_keys),
            "entries_on_disk": sorted(self._known_keys),
        }
