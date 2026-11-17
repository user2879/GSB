"""Persistent content-addressed cache with scoped invalidation."""

from __future__ import annotations

import json
from pathlib import Path

from .fingerprint import stable_json_dumps


class IncrementalCache:
    """Maps per-cell cache keys and merge dependency keys to payloads.

    The index is keyed by the logical (entity, feature, source) coordinate so a
    replaced source file, a whitelist change or a rule change produces a new
    key while unchanged coordinates keep hitting the old entry.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cell_entries: dict[str, dict] = {}
        self.merge_entries: dict[str, dict] = {}
        self.reused: list[dict] = []
        self.recomputed: list[dict] = []
        self.invalidated: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.cell_entries = payload.get("cell_entries", {})
            self.merge_entries = payload.get("merge_entries", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "cell_entries": self.cell_entries,
                    "merge_entries": self.merge_entries,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def get_cell(self, coordinate: str, cache_key: str):
        entry = self.cell_entries.get(coordinate)
        if entry is None:
            self.recomputed.append(
                {"coordinate": coordinate, "stage": "cell", "reason": "no_prior_entry"}
            )
            return None
        if entry["cache_key"] != cache_key:
            self.invalidated.append(
                {
                    "coordinate": coordinate,
                    "stage": "cell",
                    "reason": "cache_key_changed",
                    "old_key": entry["cache_key"],
                    "new_key": cache_key,
                }
            )
            self.recomputed.append(
                {"coordinate": coordinate, "stage": "cell",
                 "reason": "invalidated_by_input_or_config"}
            )
            return None
        self.reused.append({"coordinate": coordinate, "stage": "cell"})
        return entry["payload"]

    def put_cell(self, coordinate: str, cache_key: str, payload: dict) -> None:
        self.cell_entries[coordinate] = {
            "cache_key": cache_key,
            "payload": payload,
        }

    def get_merge(self, coordinate: str, dependency_key: str):
        entry = self.merge_entries.get(coordinate)
        if entry is None:
            self.recomputed.append(
                {"coordinate": coordinate, "stage": "merge",
                 "reason": "no_prior_entry"}
            )
            return None
        if entry["dependency_key"] != dependency_key:
            self.invalidated.append(
                {
                    "coordinate": coordinate,
                    "stage": "merge",
                    "reason": "dependency_key_changed",
                    "old_key": entry["dependency_key"],
                    "new_key": dependency_key,
                }
            )
            self.recomputed.append(
                {"coordinate": coordinate, "stage": "merge",
                 "reason": "invalidated_by_cell_or_rule"}
            )
            return None
        self.reused.append({"coordinate": coordinate, "stage": "merge"})
        return entry["payload"]

    def put_merge(self, coordinate: str, dependency_key: str, payload: dict) -> None:
        self.merge_entries[coordinate] = {
            "dependency_key": dependency_key,
            "payload": payload,
        }

    def prune_whitelist(self, whitelist: list[str]) -> None:
        """Drop coordinates whose feature left the whitelist."""
        allowed = set(whitelist)
        for store_name in ("cell_entries", "merge_entries"):
            store = getattr(self, store_name)
            for coordinate in list(store):
                feature = coordinate.split("|", 2)[1] if "|" in coordinate else ""
                if feature not in allowed:
                    store.pop(coordinate, None)
                    self.invalidated.append(
                        {
                            "coordinate": coordinate,
                            "stage": store_name.replace("_entries", ""),
                            "reason": "feature_left_whitelist",
                        }
                    )

    def audit(self) -> dict:
        return {
            "reused": sorted(self.reused, key=lambda x: stable_json_dumps(x)),
            "recomputed": sorted(
                self.recomputed, key=lambda x: stable_json_dumps(x)
            ),
            "invalidated": sorted(
                self.invalidated, key=lambda x: stable_json_dumps(x)
            ),
            "reused_count": len(self.reused),
            "recomputed_count": len(self.recomputed),
            "invalidated_count": len(self.invalidated),
        }

    def reset_audit(self) -> None:
        self.reused = []
        self.recomputed = []
        self.invalidated = []
