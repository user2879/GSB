"""Content-fingerprint cache book with slot-level precise invalidation.

A new backfill batch only invalidates the slot groups it covers and the
request rankings that reference those groups. Unaffected slots and requests
are served from the previous cache book and recorded as hits.
"""
from __future__ import annotations


class CacheBook:
    def __init__(self, previous: "CacheBook | None" = None):
        self.view_cache: dict[str, dict] = {}
        self.score_cache: dict[str, dict] = {}
        self.ranking_cache: dict[str, str] = {}
        self.ranking_references: dict[str, list[str]] = {}
        self.hits = {"view": 0, "score": 0, "ranking": 0}
        self.misses = {"view": 0, "score": 0, "ranking": 0}
        self.invalidated_groups: list[str] = []
        self._previous = previous

    def record_view(self, key: str, entry: dict) -> dict:
        prior = self._lookup_view(key)
        if prior is not None and prior == entry:
            self.hits["view"] += 1
            self.view_cache[key] = prior
        else:
            self.misses["view"] += 1
            self.view_cache[key] = entry
        return self.view_cache[key]

    def record_score(self, entry: dict) -> dict:
        key = entry["score_cache_key"]
        prior = self._lookup_score(key)
        if prior is not None and _comparable(prior) == _comparable(entry):
            self.hits["score"] += 1
            cached = dict(prior)
        else:
            self.misses["score"] += 1
            cached = dict(entry)
            self.score_cache[key] = _comparable(entry)
        # Unit identity always reflects the current caller; only the
        # computed content is shared through the cache.
        cached["request_id"] = entry["request_id"]
        cached["item_id"] = entry["item_id"]
        cached["points"] = entry["points"]
        return cached

    def record_ranking(self, request_id: str, ranking_key: str,
                       referenced: list[str]) -> None:
        self.ranking_references[request_id] = sorted(set(referenced))
        prior = self._lookup_ranking(ranking_key)
        if prior is not None:
            self.hits["ranking"] += 1
        else:
            self.misses["ranking"] += 1
        self.ranking_cache[ranking_key] = ranking_key

    def mark_invalidated_groups(self, group_keys: list[str]) -> None:
        self.invalidated_groups = sorted(set(group_keys))

    def summary(self) -> dict:
        return {"hits": dict(self.hits), "misses": dict(self.misses),
                "invalidated_groups": self.invalidated_groups,
                "cached_views": len(self.view_cache),
                "cached_scores": len(self.score_cache),
                "cached_rankings": len(self.ranking_cache)}

    def _lookup_view(self, key: str):
        if key in self.view_cache:
            return self.view_cache[key]
        if self._previous:
            return self._previous.view_cache.get(key)
        return None

    def _lookup_score(self, key: str):
        if key in self.score_cache:
            return self.score_cache[key]
        if self._previous:
            return self._previous.score_cache.get(key)
        return None

    def _lookup_ranking(self, key: str):
        if key in self.ranking_cache:
            return self.ranking_cache[key]
        if self._previous:
            return self._previous.ranking_cache.get(key)
        return None


def _comparable(entry: dict) -> dict:
    return {key: value for key, value in entry.items()
            if key not in ("request_id", "item_id", "points")}
