"""Atomic snapshot store and incremental result cache.

Each tier snapshot is written to a temporary file and ``os.replace``d into
place, so a hard kill can only leave a complete previous snapshot or a
complete new one. Recovery validates the configuration fingerprint and
tier identifier before accepting a snapshot; mismatch raises
``E_CONFIG_MISMATCH``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .config import fingerprint
from .errors import ConfigMismatchError
from .pipeline import QueryResult

SNAPSHOT_VERSION = 3


def scope_fingerprint(vocab_fingerprint: str, strategy_fingerprint: str,
                      tier_id: str) -> str:
    """Stable scope identifier shared by every query in one tier.

    Per-query content is deliberately excluded: a replaced sample must only
    invalidate its own record, while unchanged queries keep reusing theirs.
    """
    return fingerprint({
        "vocab": vocab_fingerprint,
        "strategy": strategy_fingerprint,
        "tier": tier_id,
    })


def cache_key(query_id: str, scope: str) -> str:
    return fingerprint({"query_id": query_id, "scope": scope})


@dataclass
class CacheCounters:
    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    reused_snapshots: int = 0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "invalidations": self.invalidations,
            "reused_snapshots": self.reused_snapshots,
        }


class SnapshotStore:
    def __init__(self, workdir: str, config_fingerprint: str):
        self.workdir = workdir
        self.config_fingerprint = config_fingerprint
        os.makedirs(workdir, exist_ok=True)
        self.counters = CacheCounters()

    def _path(self, tier_id: str) -> str:
        return os.path.join(self.workdir, f"snapshot_{tier_id}.json")

    def load(self, tier_id: str) -> Optional[dict]:
        path = self._path(tier_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        meta = payload.get("metadata", {})
        if meta.get("config_fingerprint") != self.config_fingerprint:
            raise ConfigMismatchError(
                f"snapshot {tier_id} config fingerprint mismatch; "
                "refusing to resume",
                stage="snapshot_restore")
        if meta.get("tier_id") != tier_id:
            raise ConfigMismatchError(
                f"snapshot tier identifier mismatch for {tier_id}",
                stage="snapshot_restore")
        if meta.get("snapshot_version") != SNAPSHOT_VERSION:
            raise ConfigMismatchError(
                f"snapshot version mismatch for {tier_id}",
                stage="snapshot_restore")
        self.counters.reused_snapshots += 1
        return payload

    def save(self, tier_id: str, payload: dict) -> None:
        payload = dict(payload)
        payload["metadata"] = {
            "tier_id": tier_id,
            "config_fingerprint": self.config_fingerprint,
            "snapshot_version": SNAPSHOT_VERSION,
        }
        path = self._path(tier_id)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".snapshot_{tier_id}.", suffix=".tmp", dir=self.workdir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def reuse_results(self, previous: Optional[dict],
                      current_keys: Sequence[str],
                      tier_id: str, scope: str,
                      content_by_key: Dict[str, str]) -> Dict[str, dict]:
        """Return still-valid per-query records keyed by cache key."""
        if not previous:
            return {}
        prev_meta = previous.get("metadata", {})
        if prev_meta.get("tier_id") != tier_id:
            return {}
        if previous.get("scope_fingerprint") != scope:
            self.counters.invalidations += len(previous.get("records", {}))
            return {}
        key_set = set(current_keys)
        records = previous.get("records", {})
        reusable = {}
        invalid = 0
        for key, value in records.items():
            if key not in key_set:
                invalid += 1
                continue
            # Content fingerprint guards against replaced sample text while
            # leaving every unchanged record reusable.
            if value.get("query_fingerprint") == content_by_key.get(key):
                reusable[key] = value
            else:
                invalid += 1
        self.counters.invalidations += invalid
        return reusable

    def count_lookup(self, hit: bool) -> None:
        if hit:
            self.counters.hits += 1
        else:
            self.counters.misses += 1

    def build_records(self, results: Sequence[QueryResult],
                      key_by_query: Dict[str, str]) -> Dict[str, dict]:
        return {
            key_by_query[result.query_id]: result.to_dict()
            for result in results
        }
