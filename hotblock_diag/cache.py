"""Persistent operator cache with dependency-aware invalidation.

Three artifact families carry content-addressed cache keys:

* block-pair score  -> upstream q/k block fingerprints + gear + prefilter key
* bitmap row        -> q-block key + prefilter + gear + iteration round
* hot table         -> bitmap rows it absorbed + capacity + gear + round

Dependency edges (block-pair -> bitmap rows -> hot table entries) are stored
in a JSON graph so an edit invalidates only the reachable downstream artifacts;
every cache hit and recompute is appended to an audit list.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from .util import dump_json, short_hash


def block_pair_cache_key(q_fp: str, k_fp: str, gear: str,
                         acc_bits: int, saturation: str) -> str:
    return short_hash(
        ("bp", q_fp, k_fp, gear, acc_bits, saturation).__repr__().encode()
    )


def bitmap_cache_key(q_fp: str, pre_fp: str, gear: str, round_idx: int) -> str:
    return short_hash(("bm", q_fp, pre_fp, gear, round_idx).__repr__().encode())


def hot_table_cache_key(bitmap_row_keys: list[str], capacity: int, keep: int,
                        gear: str, round_idx: int) -> str:
    joined = "|".join(bitmap_row_keys)
    return short_hash(
        ("ht", joined, capacity, keep, gear, round_idx).__repr__().encode()
    )


@dataclass
class CacheAudit:
    reused: list[str] = field(default_factory=list)
    recomputed: list[str] = field(default_factory=list)
    invalidated: list[dict[str, str]] = field(default_factory=list)

    def hit(self, key: str) -> None:
        if key not in self.reused:
            self.reused.append(key)

    def miss(self, key: str) -> None:
        if key not in self.recomputed:
            self.recomputed.append(key)

    def invalidate(self, family: str, key: str, reason: str) -> None:
        self.invalidated.append({"family": family, "key": key, "reason": reason})

    def to_dict(self) -> dict[str, Any]:
        return {
            "reused": sorted(self.reused),
            "recomputed": sorted(self.recomputed),
            "invalidated": sorted(self.invalidated,
                                   key=lambda item: (item["family"], item["key"])),
        }


class PersistentCache:
    FAMILIES = ("blockpair", "bitmap", "hottable")

    def __init__(self, root: str):
        self.root = root
        self.store_dir = os.path.join(root, "store")
        self.graph_path = os.path.join(root, "graph.json")
        os.makedirs(self.store_dir, exist_ok=True)
        self.graph = self._load_graph()
        self.audit = CacheAudit()
        self._lock = threading.RLock()

    def reset(self) -> None:
        for family in self.FAMILIES:
            directory = os.path.join(self.store_dir, family)
            if os.path.isdir(directory):
                for name in os.listdir(directory):
                    os.remove(os.path.join(directory, name))
        self.graph = {"nodes": {}, "edges": []}
        self._write_graph()
        self.audit = CacheAudit()

    def _path(self, family: str, key: str) -> str:
        directory = os.path.join(self.store_dir, family)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"{key}.json")

    def _load_graph(self) -> dict[str, Any]:
        if os.path.exists(self.graph_path):
            with open(self.graph_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {"nodes": {}, "edges": []}

    def _write_graph(self) -> None:
        dump_json(self.graph_path, self.graph)

    def get(self, family: str, key: str) -> Any | None:
        with self._lock:
            path = self._path(family, key)
            if not os.path.exists(path):
                self.audit.miss(key)
                return None
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.graph["nodes"].setdefault(key, {"family": family, "refs": []})
            self.audit.hit(key)
            return payload

    def put(self, family: str, key: str, payload: Any,
            refs: list[str] | None = None) -> None:
        with self._lock:
            path = self._path(family, key)
            tmp = path + ".tmp"
            dump_json(tmp, payload)
            os.replace(tmp, path)
            self.graph["nodes"][key] = {"family": family, "refs": sorted(refs or [])}
            for ref in refs or []:
                edge = {"from": ref, "to": key}
                if edge not in self.graph["edges"]:
                    self.graph["edges"].append(edge)
            self._write_graph()

    def invalidate_descendants(self, seed_keys: list[str], reason: str) -> list[str]:
        """Invalidate the seeds and every artifact reachable through graph edges."""
        seen: set[str] = set()
        stack = list(seed_keys)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for edge in self.graph["edges"]:
                if edge["from"] == current and edge["to"] not in seen:
                    stack.append(edge["to"])
        removed: list[str] = []
        for key in sorted(seen):
            node = self.graph["nodes"].get(key)
            if node is None:
                continue
            family = node["family"]
            path = self._path(family, key)
            if os.path.exists(path):
                os.remove(path)
                self.audit.invalidate(family, key, reason)
                removed.append(key)
            del self.graph["nodes"][key]
        self.graph["edges"] = [
            edge for edge in self.graph["edges"]
            if edge["from"] not in seen and edge["to"] not in seen
        ]
        self._write_graph()
        return removed

    def family_keys(self, family: str) -> list[str]:
        return sorted(key for key, node in self.graph["nodes"].items()
                      if node["family"] == family)
