"""Persistent node products with block-level residency and LRU eviction."""
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .fingerprint import tensor_fingerprint


def block_bytes(arr: np.ndarray) -> int:
    return int(np.prod(arr.shape)) * 8


@dataclass
class Product:
    node_id: str
    cache_key: str
    integer_output: np.ndarray
    dequant_values: np.ndarray
    meta: dict
    tensor_fp: str

    def nbytes(self) -> int:
        return block_bytes(self.integer_output) + block_bytes(
            self.dequant_values.astype(np.float64))


class ProductStore:
    """File-backed store; a separate residency ledger tracks live blocks."""

    def __init__(self, root: str, cache_budget_bytes: int):
        self.root = root
        self.cache_budget_bytes = cache_budget_bytes
        self.products_dir = os.path.join(root, "products")
        os.makedirs(self.products_dir, exist_ok=True)
        self._residents: Dict[str, Product] = {}
        self._lru: List[str] = []
        self.eviction_events: List[dict] = []
        self.recompute_events: List[dict] = []
        self._resident_bytes = 0

    # ---- persistence -----------------------------------------------------
    def _paths(self, node_id: str) -> Tuple[str, str, str]:
        base = os.path.join(self.products_dir, node_id)
        return base + ".int.npy", base + ".dq.npy", base + ".meta.json"

    def persist(self, product: Product) -> None:
        int_path, dq_path, meta_path = self._paths(product.node_id)
        tmp_int, tmp_dq, tmp_meta = int_path + ".tmp.npy", \
            dq_path + ".tmp.npy", meta_path + ".tmp"
        np.save(tmp_int, product.integer_output)
        np.save(tmp_dq, product.dequant_values)
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(product.meta, fh, indent=2, sort_keys=True)
        os.replace(tmp_int, int_path)
        os.replace(tmp_dq, dq_path)
        os.replace(tmp_meta, meta_path)

    def is_persisted(self, node_id: str) -> bool:
        int_path, _, meta_path = self._paths(node_id)
        return os.path.exists(int_path) and os.path.exists(meta_path)

    def load_meta(self, node_id: str) -> dict:
        with open(self._paths(node_id)[2], "r", encoding="utf-8") as fh:
            return json.load(fh)

    def load(self, node_id: str) -> Product:
        int_path, dq_path, meta_path = self._paths(node_id)
        integer_output = np.load(int_path)
        dequant_values = np.load(dq_path)
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return Product(node_id=node_id, cache_key=meta["cache_key"],
                       integer_output=integer_output,
                       dequant_values=dequant_values, meta=meta,
                       tensor_fp=meta.get("tensor_fp",
                                          tensor_fingerprint(integer_output)))

    def persisted_keys(self) -> Dict[str, str]:
        out = {}
        if not os.path.isdir(self.products_dir):
            return out
        for name in os.listdir(self.products_dir):
            if name.endswith(".meta.json"):
                nid = name[: -len(".meta.json")]
                out[nid] = self.load_meta(nid)["cache_key"]
        return out

    def clear(self) -> None:
        if os.path.isdir(self.products_dir):
            for name in os.listdir(self.products_dir):
                os.remove(os.path.join(self.products_dir, name))
        self._residents.clear()
        self._lru.clear()
        self._resident_bytes = 0

    def retain_only(self, node_ids) -> List[str]:
        """Delete persisted products of nodes no longer present; return ids."""
        keep = set(node_ids)
        removed = []
        for nid in list(self.persisted_keys()):
            if nid in keep:
                continue
            for path in self._paths(nid):
                if os.path.exists(path):
                    os.remove(path)
            removed.append(nid)
            self._residents.pop(nid, None)
            if nid in self._lru:
                self._lru.remove(nid)
        return removed

    # ---- residency / eviction -------------------------------------------
    def resident_node_ids(self) -> List[str]:
        return list(self._lru)

    def ensure_resident(self, node_id: str) -> Tuple[Optional[Product], bool]:
        """Return (product, was_reloaded). Evicts LRU blocks on overflow."""
        if node_id in self._residents:
            self._lru.remove(node_id)
            self._lru.append(node_id)
            return self._residents[node_id], False
        product = self.load(node_id)
        size = product.nbytes()
        evicted = []
        while self._resident_bytes + size > self.cache_budget_bytes \
                and self._lru:
            victim = self._lru[0]
            evicted.append(victim)
            self._evict(victim)
        self._residents[node_id] = product
        self._lru.append(node_id)
        self._resident_bytes += size
        if evicted:
            self.eviction_events.append(
                {"node_id": node_id, "evicted": list(evicted),
                 "block_bytes": size})
        return product, True

    def _evict(self, node_id: str) -> None:
        product = self._residents.pop(node_id)
        self._lru.remove(node_id)
        self._resident_bytes -= product.nbytes()

    def mark_recompute(self, node_id: str, reason: str, round_no: int) -> None:
        self.recompute_events.append(
            {"node_id": node_id, "reason": reason, "round": round_no})
