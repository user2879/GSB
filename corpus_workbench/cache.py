"""Content-addressed signature and group-decision caches with audit counters."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .models import SignatureRecord
from .util import read_json, write_json_atomic


class SignatureCache:
    """One JSON file per content/parameter key.

    The cache deliberately excludes the Jaccard threshold from its key. Changing
    only the threshold therefore reuses every signature while forcing grouping
    and representative decisions to be recomputed.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root) / "signatures"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._index: dict[str, dict[str, Any]] = {}
        self._by_document: dict[tuple[str, str], list[str]] = {}
        self.reset_stats()
        self._rebuild_index()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def _rebuild_index(self) -> None:
        for path in sorted(self.root.glob("*/*.json")):
            try:
                value = read_json(path)
                key = str(value["cache_key"])
                if key != path.stem:
                    continue
                record = SignatureRecord.from_dict(value)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            self._index[key] = record.to_dict()
            self._by_document.setdefault((record.source, record.doc_id), []).append(key)
        for keys in self._by_document.values():
            keys.sort()

    def reset_stats(self) -> None:
        with self._lock:
            self._stats: dict[str, Any] = {
                "hits": 0,
                "misses": 0,
                "invalidated_by_params": 0,
                "invalidated_by_content": 0,
                "hit_sources": {},
                "miss_sources": {},
                "hit_documents": [],
                "computed_documents": [],
            }

    def lookup(
        self,
        key: str,
        *,
        source: str,
        doc_id: str,
        fingerprint: str,
        shingle_length: int,
        signature_dim: int,
    ) -> SignatureRecord | None:
        with self._lock:
            value = self._index.get(key)
            if value is None:
                path = self._path(key)
                if path.exists():
                    try:
                        value = read_json(path)
                        record = SignatureRecord.from_dict(value)
                        self._index[key] = value
                        self._by_document.setdefault((record.source, record.doc_id), []).append(key)
                        self._by_document[(record.source, record.doc_id)].sort()
                    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                        value = None
            if value is not None:
                try:
                    record = SignatureRecord.from_dict(value)
                except (KeyError, TypeError, ValueError):
                    record = None
                if (
                    record is not None
                    and record.cache_key == key
                    and record.source == source
                    and record.doc_id == doc_id
                    and record.fingerprint == fingerprint
                    and record.shingle_length == shingle_length
                    and record.signature_dim == signature_dim
                ):
                    self._stats["hits"] += 1
                    sources = self._stats["hit_sources"]
                    sources[source] = sources.get(source, 0) + 1
                    self._stats["hit_documents"].append(f"{source}\x1f{doc_id}")
                    return SignatureRecord(
                        source=record.source,
                        doc_id=record.doc_id,
                        fingerprint=record.fingerprint,
                        shingle_length=record.shingle_length,
                        signature_dim=record.signature_dim,
                        point_cost=record.point_cost,
                        signature=record.signature,
                        cache_key=record.cache_key,
                        cache_hit=True,
                    )

            self._stats["misses"] += 1
            sources = self._stats["miss_sources"]
            sources[source] = sources.get(source, 0) + 1
            invalidated_content = False
            invalidated_params = False
            for old_key in self._by_document.get((source, doc_id), []):
                old_value = self._index.get(old_key, {})
                if old_value.get("fingerprint") != fingerprint:
                    invalidated_content = True
                elif old_key != key:
                    invalidated_params = True
            if invalidated_content:
                self._stats["invalidated_by_content"] += 1
            if invalidated_params:
                self._stats["invalidated_by_params"] += 1
            return None

    def put(self, record: SignatureRecord) -> None:
        value = record.to_dict()
        with self._lock:
            write_json_atomic(self._path(record.cache_key), value)
            self._index[record.cache_key] = value
            keys = self._by_document.setdefault((record.source, record.doc_id), [])
            if record.cache_key not in keys:
                keys.append(record.cache_key)
                keys.sort()
            self._stats["computed_documents"].append(f"{record.source}\x1f{record.doc_id}")

    def stats(self) -> dict[str, Any]:
        with self._lock:
            hit_sources = dict(sorted(self._stats["hit_sources"].items()))
            miss_sources = dict(sorted(self._stats["miss_sources"].items()))
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "invalidated_by_params": self._stats["invalidated_by_params"],
                "invalidated_by_content": self._stats["invalidated_by_content"],
                "hit_sources": hit_sources,
                "miss_sources": miss_sources,
                "reused_sources": sorted(source for source, count in hit_sources.items() if count > 0),
                "recomputed_sources": sorted(source for source, count in miss_sources.items() if count > 0),
                "hit_documents": sorted(self._stats["hit_documents"]),
                "computed_documents": sorted(self._stats["computed_documents"]),
            }


class GroupDecisionCache:
    """Cache exact representative decisions for an already-formed group."""

    def __init__(self, root: str | Path):
        self.root = Path(root) / "group_decisions"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._hit_groups: list[str] = []
        self._miss_groups: list[str] = []

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        with self._lock:
            if not path.exists():
                self._misses += 1
                self._miss_groups.append(key)
                return None
            try:
                value = read_json(path)
            except (OSError, json.JSONDecodeError):
                self._misses += 1
                self._miss_groups.append(key)
                return None
            self._hits += 1
            self._hit_groups.append(key)
            return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            write_json_atomic(self._path(key), value)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_group_keys": sorted(self._hit_groups),
                "miss_group_keys": sorted(self._miss_groups),
            }
