"""Content-addressed stage artifacts and deterministic LRU latent blocks."""

from __future__ import annotations

import io
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import numpy as np

from .util import WORKBENCH_FORMAT_VERSION, array_digest, bytes_digest, canonical_json, sha256_hex


T = TypeVar("T")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, (canonical_json(value) + "\n").encode("utf-8"))


@dataclass
class StageArtifact:
    key: str
    kind: str
    content_fingerprint: str
    array: np.ndarray | None = None
    png_bytes: bytes | None = None
    pixels: np.ndarray | None = None
    meta: dict[str, Any] | None = None


class ArtifactStore:
    """A persistent, content-keyed stage store.

    Cache keys are full SHA-256 strings.  A hit is returned only after the
    metadata key/kind and artifact file shape/dtype/digest have been checked;
    malformed or partially written entries are treated as misses.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.artifact_root = self.root / "artifacts"
        self.state_root = self.root / "state"
        self.snapshot_root = self.root / "snapshots"
        self.run_root = self.root / "runs"
        for directory in (self.artifact_root, self.state_root, self.snapshot_root, self.run_root):
            directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _directory(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("cache key must be a lowercase SHA-256 hex string")
        return self.artifact_root / key

    def has_array(self, key: str, kind: str) -> bool:
        with self._lock:
            return self._load_array_unlocked(key, kind) is not None

    def get_array(self, key: str, kind: str) -> np.ndarray | None:
        with self._lock:
            return self._load_array_unlocked(key, kind)

    def _load_array_unlocked(self, key: str, kind: str) -> np.ndarray | None:
        directory = self._directory(key)
        meta_path = directory / "meta.json"
        data_path = directory / "data.npy"
        if not meta_path.is_file() or not data_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("format_version") != WORKBENCH_FORMAT_VERSION:
                return None
            if meta.get("key") != key or meta.get("kind") != kind:
                return None
            array = np.load(data_path, allow_pickle=False)
            if list(array.shape) != meta.get("shape") or array.dtype.str != meta.get("dtype"):
                return None
            if array_digest(array) != meta.get("content_fingerprint"):
                return None
            return np.ascontiguousarray(array)
        except Exception:
            return None

    def put_array(
        self,
        key: str,
        kind: str,
        array: np.ndarray,
        *,
        parameters: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StageArtifact:
        contiguous = np.ascontiguousarray(array)
        fingerprint = array_digest(contiguous)
        meta = {
            "format_version": WORKBENCH_FORMAT_VERSION,
            "key": key,
            "kind": kind,
            "shape": list(contiguous.shape),
            "dtype": contiguous.dtype.str,
            "content_fingerprint": fingerprint,
            "parameters_fingerprint": sha256_hex("parameters-v1", parameters or {}),
            "version": WORKBENCH_FORMAT_VERSION,
            **(extra or {}),
        }
        with self._lock:
            directory = self._directory(key)
            _atomic_write(directory / "data.npy", _npy_bytes(contiguous))
            _write_json(directory / "meta.json", meta)
        return StageArtifact(
            key=key,
            kind=kind,
            content_fingerprint=fingerprint,
            array=contiguous,
            meta=meta,
        )

    def get_decoded(self, key: str) -> StageArtifact | None:
        with self._lock:
            directory = self._directory(key)
            meta_path = directory / "meta.json"
            png_path = directory / "image.png"
            pixels_path = directory / "pixels.npy"
            if not meta_path.is_file() or not png_path.is_file() or not pixels_path.is_file():
                return None
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("format_version") != WORKBENCH_FORMAT_VERSION:
                    return None
                if meta.get("key") != key or meta.get("kind") != "decode":
                    return None
                png_bytes = png_path.read_bytes()
                pixels = np.load(pixels_path, allow_pickle=False)
                if bytes_digest(png_bytes) != meta.get("png_sha256"):
                    return None
                if array_digest(pixels) != meta.get("pixels_fingerprint"):
                    return None
                if list(pixels.shape) != meta.get("pixels_shape"):
                    return None
                return StageArtifact(
                    key=key,
                    kind="decode",
                    content_fingerprint=meta.get("content_fingerprint", array_digest(pixels)),
                    png_bytes=png_bytes,
                    pixels=np.ascontiguousarray(pixels),
                    meta=meta,
                )
            except Exception:
                return None

    def put_decoded(
        self,
        key: str,
        png_bytes: bytes,
        pixels: np.ndarray,
        *,
        parameters: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StageArtifact:
        contiguous = np.ascontiguousarray(pixels)
        pixels_fingerprint = array_digest(contiguous)
        png_sha256 = bytes_digest(png_bytes)
        meta = {
            "format_version": WORKBENCH_FORMAT_VERSION,
            "key": key,
            "kind": "decode",
            "content_fingerprint": sha256_hex("decoded-v1", pixels_fingerprint, png_sha256),
            "pixels_fingerprint": pixels_fingerprint,
            "png_sha256": png_sha256,
            "pixels_shape": list(contiguous.shape),
            "parameters_fingerprint": sha256_hex("parameters-v1", parameters or {}),
            "version": WORKBENCH_FORMAT_VERSION,
            **(extra or {}),
        }
        with self._lock:
            directory = self._directory(key)
            _atomic_write(directory / "pixels.npy", _npy_bytes(contiguous))
            _atomic_write(directory / "image.png", png_bytes)
            _write_json(directory / "meta.json", meta)
        return StageArtifact(
            key=key,
            kind="decode",
            content_fingerprint=meta["content_fingerprint"],
            png_bytes=png_bytes,
            pixels=contiguous,
            meta=meta,
        )

    def load_manifest(self) -> dict[str, Any] | None:
        path = self.state_root / "last_manifest.json"
        with self._lock:
            if not path.is_file():
                return None
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        with self._lock:
            _write_json(self.state_root / "last_manifest.json", manifest)

    def save_snapshot(self, name: str, value: dict[str, Any]) -> Path:
        path = self.snapshot_root / name
        with self._lock:
            _write_json(path, value)
        return path

    def copy_png_to_run(self, run_id: str, relative_path: str, png_bytes: bytes) -> Path:
        path = self.run_root / run_id / relative_path
        with self._lock:
            _atomic_write(path, png_bytes)
        return path


@dataclass
class CacheDiagnostics:
    capacity: int
    resident_blocks: int
    accesses: int
    hits: int
    reuses: int
    misses: int
    evictions: int
    recomputes: int
    recompute_matches: int
    recompute_mismatches: int
    access_clock: int
    eviction_log: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "resident_blocks": self.resident_blocks,
            "accesses": self.accesses,
            "hits": self.hits,
            "reuses": self.reuses,
            "misses": self.misses,
            "evictions": self.evictions,
            "recomputes": self.recomputes,
            "recompute_matches": self.recompute_matches,
            "recompute_mismatches": self.recompute_mismatches,
            "access_clock": self.access_clock,
            "eviction_log": list(self.eviction_log),
        }


@dataclass
class _Entry(Generic[T]):
    value: T
    last_access: int
    digest: str | None


class LatentBlockCache(Generic[T]):
    """Bounded LRU cache whose eviction tie-break is lexical block key.

    On an access after a prior eviction, a caller-supplied recompute callback
    is invoked.  Its digest is checked against the formerly resident value.
    This makes cache correctness observable instead of inferring it from
    execution timing.
    """

    def __init__(self, capacity: int, *, log_limit: int = 16) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._entries: dict[str, _Entry[T]] = {}
        self._seen: set[str] = set()
        self._known_digests: dict[str, str] = {}
        self._clock = 0
        self.accesses = 0
        self.hits = 0
        self.reuses = 0
        self.misses = 0
        self.evictions = 0
        self.recomputes = 0
        self.recompute_matches = 0
        self.recompute_mismatches = 0
        self._log_limit = log_limit
        self._eviction_log: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def _next_access(self, requested: int | None) -> int:
        if requested is None:
            self._clock += 1
            return self._clock
        self._clock = max(self._clock, requested)
        return requested

    def _evict_one_unlocked(self) -> str:
        victim_key = min(self._entries, key=lambda key: (self._entries[key].last_access, key))
        entry = self._entries.pop(victim_key)
        self.evictions += 1
        if len(self._eviction_log) >= self._log_limit:
            self._eviction_log.pop(0)
        self._eviction_log.append(
            {
                "key": victim_key,
                "last_access": entry.last_access,
                "value_digest": entry.digest,
            }
        )
        return victim_key

    def put(
        self,
        key: str,
        value: T,
        *,
        access_seq: int | None = None,
        digest: str | None = None,
    ) -> None:
        with self._lock:
            sequence = self._next_access(access_seq)
            existing = self._entries.get(key)
            if existing is not None:
                existing.value = value
                existing.last_access = sequence
                existing.digest = digest
            else:
                while len(self._entries) >= self.capacity:
                    self._evict_one_unlocked()
                self._entries[key] = _Entry(value=value, last_access=sequence, digest=digest)
            self._seen.add(key)
            if digest is not None:
                self._known_digests[key] = digest(value) if callable(digest) else digest

    def get(
        self,
        key: str,
        *,
        access_seq: int | None = None,
        recompute: Callable[[], T] | None = None,
        digest: Callable[[T], str] | None = None,
    ) -> T | None:
        with self._lock:
            self.accesses += 1
            sequence = self._next_access(access_seq)
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_access = sequence
                self.hits += 1
                self.reuses += 1
                return entry.value

            was_seen = key in self._seen
            if was_seen:
                self.recomputes += 1
            else:
                self.misses += 1
            if recompute is None:
                return None
            value = recompute()
            value_digest = digest(value) if digest is not None else None
            prior_digest = self._known_digests.get(key)
            if prior_digest is not None and value_digest is not None:
                if prior_digest == value_digest:
                    self.recompute_matches += 1
                else:
                    self.recompute_mismatches += 1
            while len(self._entries) >= self.capacity:
                self._evict_one_unlocked()
            self._entries[key] = _Entry(value=value, last_access=sequence, digest=value_digest)
            self._seen.add(key)
            if value_digest is not None:
                self._known_digests[key] = value_digest
            return value

    def peek(self, key: str) -> T | None:
        with self._lock:
            entry = self._entries.get(key)
            return entry.value if entry is not None else None

    def diagnostics(self) -> CacheDiagnostics:
        with self._lock:
            return CacheDiagnostics(
                capacity=self.capacity,
                resident_blocks=len(self._entries),
                accesses=self.accesses,
                hits=self.hits,
                reuses=self.reuses,
                misses=self.misses,
                evictions=self.evictions,
                recomputes=self.recomputes,
                recompute_matches=self.recompute_matches,
                recompute_mismatches=self.recompute_mismatches,
                access_clock=self._clock,
                eviction_log=list(self._eviction_log),
            )


def cache_key(stage: str, upstream_content_fingerprint: str, parameter_fingerprint: str, strategy_version: str) -> str:
    """Derive a stage key from upstream content, parameters and table version."""

    return sha256_hex(
        "stage-cache-key-v1",
        {"stage": stage},
        {"upstream": upstream_content_fingerprint},
        {"parameters": parameter_fingerprint},
        {"strategy_version": strategy_version},
    )
