"""Content-addressed persistent products and append-only audit ledger."""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .util import atomic_write_text, canonical_json, digest_obj, read_json_file, sha256_hex


class CacheStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.products_dir = self.root / "products"
        self.manifest_path = self.root / "manifest.json"
        self.products_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def make_key(*, stage: str, upstream: list[dict[str, Any]], params_digest: str,
                 quantization: dict[str, Any], lease_generation: int) -> str:
        payload = {
            "schema": "mdet-cache-key-v1",
            "stage": stage,
            "upstream": sorted(upstream, key=lambda item: (item.get("role", ""), item.get("cache_key", ""))),
            "params_digest": params_digest,
            "quantization": quantization,
            "lease_generation": int(lease_generation),
        }
        return sha256_hex(canonical_json(payload))

    def product_path(self, key: str) -> Path:
        if not key or any(character not in "0123456789abcdef" for character in key):
            raise BusinessError("E_CACHE_KEY", "cache key is not a lowercase hexadecimal digest", {"key": key})
        return self.products_dir / f"{key}.json"

    def has(self, key: str) -> bool:
        return self.product_path(key).is_file()

    def load(self, key: str) -> dict[str, Any]:
        path = self.product_path(key)
        raw = read_json_file(path, code="E_CACHE_JSON")
        if not isinstance(raw, dict):
            raise BusinessError("E_CACHE_ARTIFACT", "cached product must be an object", {"path": str(path)})
        if raw.get("cache_key") != key:
            raise BusinessError("E_CACHE_ARTIFACT", "cached product key mismatch", {"path": str(path)})
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            raise BusinessError("E_CACHE_ARTIFACT", "cached product payload is invalid", {"path": str(path)})
        if raw.get("artifact_digest") != digest_obj(payload):
            raise BusinessError("E_CACHE_CORRUPT", "cached product artifact digest mismatch",
                                {"path": str(path), "key": key})
        return raw

    def save(self, product: dict[str, Any]) -> None:
        key = product.get("cache_key")
        if not isinstance(key, str):
            raise BusinessError("E_CACHE_ARTIFACT", "product has no cache key")
        product = copy.deepcopy(product)
        product["artifact_digest"] = digest_obj(product["payload"])
        text = canonical_json(product) + "\n"
        with self._lock:
            atomic_write_text(self.product_path(key), text)

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"format_version": 1, "tasks": {}}
        raw = read_json_file(self.manifest_path, code="E_CACHE_MANIFEST")
        if not isinstance(raw, dict) or raw.get("format_version") != 1:
            raise BusinessError("E_CACHE_MANIFEST", "cache manifest is invalid", {"path": str(self.manifest_path)})
        if not isinstance(raw.get("tasks"), dict):
            raise BusinessError("E_CACHE_MANIFEST", "cache manifest tasks must be an object")
        return raw

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        text = canonical_json(manifest) + "\n"
        with self._lock:
            atomic_write_text(self.manifest_path, text)


class LedgerWriter:
    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.sequence = 0
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self.sequence += 1
            record = {"run_id": self.run_id, "sequence": self.sequence, **entry}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(record) + "\n")

    def run_header(self, payload: dict[str, Any]) -> None:
        self._append({"event": "run_start", **payload})

    def task(self, payload: dict[str, Any]) -> None:
        self._append({"event": "task", **payload})

    def scheduler_batch(self, payload: dict[str, Any]) -> None:
        self._append({"event": "scheduler_batch", **payload})

    def event(self, payload: dict[str, Any]) -> None:
        self._append({"event": "event", **payload})

    def run_end(self, payload: dict[str, Any]) -> None:
        self._append({"event": "run_end", **payload})
