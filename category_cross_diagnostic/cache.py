"""Persistent content-addressed stage cache."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .hashing import canonical_json


class CacheStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.load()

    def load(self) -> None:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.entries = dict(payload.get("entries", {}))

    def has(self, cache_key: str) -> bool:
        return cache_key in self.entries

    def get(self, cache_key: str) -> Any:
        if cache_key not in self.entries:
            raise KeyError(cache_key)
        self.hits += 1
        return self.entries[cache_key]["value"]

    def envelope(self, cache_key: str) -> dict[str, Any] | None:
        return self.entries.get(cache_key)

    def put(
        self,
        cache_key: str,
        value: Any,
        *,
        stage: str,
        combo_id: str,
        combo_fingerprint: str,
        base_fingerprint: str,
        parameter_fingerprint: str,
        width_gear_fingerprint: str,
        column_fingerprints: dict[str, str],
    ) -> None:
        self.entries[cache_key] = {
            "cache_key": cache_key,
            "stage": stage,
            "combo_id": combo_id,
            "combo_fingerprint": combo_fingerprint,
            "base_fingerprint": base_fingerprint,
            "parameter_fingerprint": parameter_fingerprint,
            "width_gear_fingerprint": width_gear_fingerprint,
            "column_fingerprints": column_fingerprints,
            "value": value,
        }
        self.misses += 1

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {"version": 1, "entries": self.entries},
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


def assert_canonically_equal(actual: Any, expected: Any) -> None:
    if canonical_json(actual) != canonical_json(expected):
        raise AssertionError("canonical field mismatch")
