from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .hashing import canonical_json_bytes, sha256_bytes, sha256_json
from .model import ParsedPackage, Track


@dataclass
class CacheEntry:
    stage: str
    key: str
    sort_key: str
    refs: tuple[str, ...]
    meta: dict[str, Any]
    payload: bytes

    @property
    def payload_sha256(self) -> str:
        return sha256_bytes(self.payload)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "key": self.key,
            "sort_key": self.sort_key,
            "refs": list(self.refs),
            "meta": self.meta,
            "payload_sha256": self.payload_sha256,
            "payload_size": len(self.payload),
        }


def array_entry(
    stage: str,
    key: str,
    sort_key: str,
    array: np.ndarray,
    *,
    refs: Iterable[str] = (),
    meta: dict[str, Any] | None = None,
) -> CacheEntry:
    contiguous = np.ascontiguousarray(array)
    payload_meta = dict(meta or {})
    payload_meta.update(
        {
            "kind": "ndarray",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
    )
    return CacheEntry(
        stage=stage,
        key=key,
        sort_key=sort_key,
        refs=tuple(sorted(refs)),
        meta=payload_meta,
        payload=contiguous.tobytes(order="C"),
    )


def array_from_entry(entry: CacheEntry) -> np.ndarray:
    if entry.meta.get("kind") != "ndarray":
        raise ValueError(f"cache entry {entry.key} is not an ndarray")
    dtype = np.dtype(entry.meta["dtype"])
    shape = tuple(int(value) for value in entry.meta["shape"])
    array = np.frombuffer(entry.payload, dtype=dtype)
    expected = int(np.prod(shape)) if shape else 1
    if array.size != expected:
        raise ValueError(f"cache entry {entry.key} array size mismatch")
    return array.reshape(shape).copy()


def package_entry(
    package: ParsedPackage,
    sort_key: str,
    *,
    stage: str = "parse",
    key: str | None = None,
) -> CacheEntry:
    layout: list[dict[str, Any]] = []
    blocks: list[bytes] = []
    offset = 0
    for name in package.track_order:
        track = package.tracks[name]
        raw = np.ascontiguousarray(track.pcm, dtype="<i2").tobytes(order="C")
        layout.append(
            {
                "name": name,
                "offset": offset,
                "length": len(raw),
                "frames": track.frames,
                "sample_rate": track.sample_rate,
                "source_format": track.source_format,
                "metadata": track.metadata,
            }
        )
        blocks.append(raw)
        offset += len(raw)
    meta = {
        "kind": "package",
        "format_version": package.format_version,
        "sample_rate": package.sample_rate,
        "track_order": list(package.track_order),
        "track_layout": layout,
        "source_path": package.source_path,
        "sidecar": package.sidecar,
        "sidecar_sha256": sha256_bytes(package.sidecar_bytes),
        "sidecar_b64": base64.b64encode(package.sidecar_bytes).decode("ascii"),
        "package_fingerprint": package.package_fingerprint,
        "source_fingerprint": package.source_fingerprint,
        "parse_key": package.parse_key,
        "conversion_ledger": package.conversion_ledger,
    }
    return CacheEntry(
        stage=stage,
        key=key or (package.parse_key if stage == "parse" else package.package_fingerprint),
        sort_key=sort_key,
        refs=tuple(sorted(package.tracks)),
        meta=meta,
        payload=b"".join(blocks),
    )


def package_from_entry(entry: CacheEntry) -> ParsedPackage:
    if entry.meta.get("kind") != "package":
        raise ValueError(f"cache entry {entry.key} is not a package")
    tracks: dict[str, Track] = {}
    for item in entry.meta["track_layout"]:
        start = int(item["offset"])
        end = start + int(item["length"])
        pcm = np.frombuffer(entry.payload[start:end], dtype="<i2").copy()
        track = Track(
            name=item["name"],
            pcm=pcm,
            sample_rate=int(item["sample_rate"]),
            source_format=item["source_format"],
            metadata=dict(item["metadata"]),
        )
        tracks[track.name] = track
    sidecar = dict(entry.meta.get("sidecar", {}))
    sidecar_bytes = base64.b64decode(entry.meta.get("sidecar_b64", ""))
    if entry.meta.get("sidecar_sha256") != sha256_bytes(sidecar_bytes):
        raise ValueError(f"cache entry {entry.key} sidecar checksum mismatch")
    package = ParsedPackage(
        format_version=entry.meta["format_version"],
        sample_rate=int(entry.meta["sample_rate"]),
        tracks=tracks,
        track_order=tuple(entry.meta["track_order"]),
        source_path=entry.meta["source_path"],
        sidecar=sidecar,
        sidecar_bytes=sidecar_bytes,
        package_fingerprint=entry.meta["package_fingerprint"],
        source_fingerprint=entry.meta["source_fingerprint"],
        parse_key=entry.meta.get("parse_key", ""),
        conversion_ledger=list(entry.meta.get("conversion_ledger", [])),
    )
    return package


class CacheStore:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], CacheEntry] = {}
        self._lock = threading.RLock()

    def get(self, stage: str, key: str) -> CacheEntry | None:
        with self._lock:
            return self._entries.get((stage, key))

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            self._entries[(entry.stage, entry.key)] = entry

    def entries_for_stage(self, stage: str) -> list[CacheEntry]:
        with self._lock:
            entries = [entry for (entry_stage, _), entry in self._entries.items() if entry_stage == stage]
        return sorted(entries, key=lambda entry: (entry.sort_key, entry.key))

    def all_entries(self) -> list[CacheEntry]:
        with self._lock:
            entries = list(self._entries.values())
        return sorted(entries, key=lambda entry: (entry.stage, entry.sort_key, entry.key))

    def stage_version(self, stage: str) -> str:
        records = [
            {
                "key": entry.key,
                "sort_key": entry.sort_key,
                "meta_sha256": sha256_json(entry.meta),
                "payload_sha256": entry.payload_sha256,
            }
            for entry in self.entries_for_stage(stage)
        ]
        return sha256_json(records)

    def count(self, stage: str | None = None) -> int:
        with self._lock:
            if stage is None:
                return len(self._entries)
            return sum(1 for entry_stage, _ in self._entries if entry_stage == stage)

    def load_entry(self, entry: CacheEntry) -> None:
        self.put(entry)

    def merge(self, other: "CacheStore") -> None:
        for entry in other.all_entries():
            self.put(entry)

    def to_summary(self) -> dict[str, Any]:
        stages = sorted({entry.stage for entry in self.all_entries()})
        return {
            "entry_count": self.count(),
            "by_stage": {stage: self.count(stage) for stage in stages},
            "versions": {stage: self.stage_version(stage) for stage in stages},
        }


def entry_to_disk(entry: CacheEntry, stage_dir: Path) -> None:
    entries_dir = stage_dir / "entries"
    blobs_dir = stage_dir / "blobs"
    entries_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir.mkdir(parents=True, exist_ok=True)
    manifest = entry.to_manifest()
    (entries_dir / f"{entry.key}.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    (blobs_dir / f"{entry.key}.bin").write_bytes(entry.payload)


def entry_from_disk(stage_dir: Path, manifest: dict[str, Any]) -> CacheEntry:
    key = manifest["key"]
    payload = (stage_dir / "blobs" / f"{key}.bin").read_bytes()
    if sha256_bytes(payload) != manifest["payload_sha256"]:
        raise ValueError(f"snapshot blob checksum mismatch for {key}")
    return CacheEntry(
        stage=manifest["stage"],
        key=key,
        sort_key=manifest["sort_key"],
        refs=tuple(manifest.get("refs", [])),
        meta=dict(manifest.get("meta", {})),
        payload=payload,
    )


def load_stage_entries(stage_dir: Path) -> list[CacheEntry]:
    entries_dir = stage_dir / "entries"
    if not entries_dir.exists():
        return []
    result: list[CacheEntry] = []
    for path in sorted(entries_dir.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        result.append(entry_from_disk(stage_dir, manifest))
    return result
