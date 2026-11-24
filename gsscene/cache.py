"""Persistent artifact cache with precise key-based reuse."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import codes
from .models import CompositeRecord, Failure, KernelGroup, TileRecord
from .util import arrays_fingerprint, atomic_save_npz, atomic_write_json, jsonable, read_json, read_npz, remove_tree

CACHE_SCHEMA_VERSION = 3


def kernel_artifact_id(view_index: int, level: int) -> str:
    return f"{view_index:04d}_L{level:02d}"


def tile_artifact_id(view_index: int, level: int, row: int, col: int) -> str:
    return f"{view_index:04d}_L{level:02d}_R{row:02d}_C{col:02d}"


def composite_artifact_id(view_index: int, level: int) -> str:
    return f"{view_index:04d}_L{level:02d}"


class ArtifactCache:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = Path(work_dir).resolve()
        self.root = self.work_dir / "cache"
        self.kernel_dir = self.root / "kernels"
        self.tile_dir = self.root / "tiles"
        self.composite_dir = self.root / "composites"
        self.diagnostic_dir = self.root / "diagnostics"
        self.index_path = self.root / "index.json"
        self.errors: list[Failure] = []
        for directory in (self.kernel_dir, self.tile_dir, self.composite_dir, self.diagnostic_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        remove_tree(self.root)
        self.__init__(self.work_dir)

    def _meta_path(self, kind: str, artifact_id: str) -> Path:
        directory = {
            "kernel": self.kernel_dir,
            "tile": self.tile_dir,
            "composite": self.composite_dir,
        }[kind]
        return directory / f"{artifact_id}.meta.json"

    def _npz_path(self, kind: str, artifact_id: str) -> Path:
        directory = {
            "kernel": self.kernel_dir,
            "tile": self.tile_dir,
            "composite": self.composite_dir,
        }[kind]
        return directory / f"{artifact_id}.npz"

    def _load_artifact(self, kind: str, artifact_id: str) -> tuple[dict[str, Any] | None, dict[str, np.ndarray] | None]:
        meta_path = self._meta_path(kind, artifact_id)
        npz_path = self._npz_path(kind, artifact_id)
        if not meta_path.exists() or not npz_path.exists():
            return None, None
        try:
            meta = read_json(meta_path)
            if int(meta.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
                return None, None
            arrays = read_npz(npz_path)
            if arrays_fingerprint(arrays) != meta.get("arrays_fingerprint"):
                self.errors.append(
                    Failure(
                        code=codes.E_CACHE_CORRUPT,
                        message=f"array fingerprint mismatch for {kind} {artifact_id}",
                        source="cache",
                        view_index=meta.get("view"),
                        level=meta.get("level"),
                    )
                )
                return None, None
            return meta, arrays
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.errors.append(
                Failure(
                    code=codes.E_CACHE_CORRUPT,
                    message=f"cannot read {kind} {artifact_id}: {exc}",
                    source="cache",
                    view_index=None,
                    level=None,
                )
            )
            return None, None

    def load_kernel(self, view_index: int, level: int) -> tuple[dict[str, Any] | None, dict[str, np.ndarray] | None]:
        return self._load_artifact("kernel", kernel_artifact_id(view_index, level))

    def load_tile(self, view_index: int, level: int, row: int, col: int) -> tuple[dict[str, Any] | None, dict[str, np.ndarray] | None]:
        return self._load_artifact("tile", tile_artifact_id(view_index, level, row, col))

    def load_composite(self, view_index: int, level: int) -> tuple[dict[str, Any] | None, dict[str, np.ndarray] | None]:
        return self._load_artifact("composite", composite_artifact_id(view_index, level))

    def write_kernel_group(self, group: KernelGroup, requested_count: int) -> dict[str, Any]:
        artifact_id = kernel_artifact_id(group.view_index, group.level)
        meta = {
            "allowed_count": len(group.kernel_keys),
            "arrays_fingerprint": arrays_fingerprint(group.arrays),
            "cache_key": group.cache_key,
            "camera_fingerprint": group.camera_fingerprint,
            "grid_size": list(group.grid_size),
            "kernel_content_fingerprints": group.kernel_content_fingerprints,
            "kernel_keys": group.kernel_keys,
            "kind": "kernel",
            "level": group.level,
            "level_fingerprint": group.level_fingerprint,
            "projection_matrix_fingerprint": group.projection_matrix_fingerprint,
            "requested_count": int(requested_count),
            "schema_version": CACHE_SCHEMA_VERSION,
            "view": group.view_index,
            "view_content_fingerprint": group.view_content_fingerprint,
        }
        atomic_save_npz(self._npz_path("kernel", artifact_id), group.arrays)
        atomic_write_json(self._meta_path("kernel", artifact_id), meta)
        return meta

    def write_tile(self, record: TileRecord) -> dict[str, Any]:
        artifact_id = tile_artifact_id(record.view_index, record.level, record.row, record.col)
        meta = {
            "arrays_fingerprint": arrays_fingerprint(record.arrays),
            "cache_key": record.cache_key,
            "col": record.col,
            "kernel_refs": [int(item) for item in record.kernel_refs],
            "kind": "tile",
            "level": record.level,
            "row": record.row,
            "schema_version": CACHE_SCHEMA_VERSION,
            "view": record.view_index,
        }
        atomic_save_npz(self._npz_path("tile", artifact_id), record.arrays)
        atomic_write_json(self._meta_path("tile", artifact_id), meta)
        return meta

    def write_composite(self, record: CompositeRecord) -> dict[str, Any]:
        artifact_id = composite_artifact_id(record.view_index, record.level)
        meta = {
            "arrays_fingerprint": arrays_fingerprint(record.arrays),
            "cache_key": record.cache_key,
            "kind": "composite",
            "level": record.level,
            "lower_tile_refs": [list(item) for item in record.lower_tile_refs],
            "lower_composite_key": record.lower_composite_key,
            "schema_version": CACHE_SCHEMA_VERSION,
            "tile_refs": [list(item) for item in record.tile_refs],
            "view": record.view_index,
        }
        atomic_save_npz(self._npz_path("composite", artifact_id), record.arrays)
        atomic_write_json(self._meta_path("composite", artifact_id), meta)
        return meta

    def save_diagnostics(self, view_index: int, level: int, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> Path:
        artifact_id = kernel_artifact_id(view_index, level)
        npz_path = self.diagnostic_dir / f"{artifact_id}.npz"
        meta_path = self.diagnostic_dir / f"{artifact_id}.json"
        atomic_save_npz(npz_path, arrays)
        payload = dict(summary)
        payload["arrays_fingerprint"] = arrays_fingerprint(arrays)
        payload["npz"] = str(npz_path.relative_to(self.work_dir))
        atomic_write_json(meta_path, payload)
        return npz_path

    def cleanup(self, expected: dict[str, set[str]]) -> list[str]:
        removed: list[str] = []
        mapping = {
            "kernel": self.kernel_dir,
            "tile": self.tile_dir,
            "composite": self.composite_dir,
        }
        for kind, directory in mapping.items():
            allowed = expected.get(kind, set())
            for path in sorted(list(directory.iterdir()), key=lambda item: item.name):
                if not path.is_file() or not path.name.endswith((".npz", ".meta.json")):
                    continue
                artifact_id = path.name[: -len(".meta.json")] if path.name.endswith(".meta.json") else path.name[: -len(".npz")]
                if artifact_id not in allowed:
                    removed.append(f"{kind}:{artifact_id}:{path.suffix}")
                    path.unlink(missing_ok=True)
        for path in sorted(self.diagnostic_dir.iterdir(), key=lambda item: item.name):
            if path.is_file():
                artifact_id = path.name.split(".", 1)[0]
                expected_pair = any(
                    artifact_id == kernel_artifact_id(view_index, level)
                    for view_index, level in _parse_expected_kernel_ids(expected.get("kernel", set()))
                )
                if not expected_pair:
                    path.unlink(missing_ok=True)
        return removed

    def write_index(self, report: dict[str, Any]) -> None:
        payload = {
            "composites": {entry["cache_key"]: entry.get("artifact_id") for entry in report.get("composites", [])},
            "kernels": {entry["cache_key"]: entry.get("artifact_id") for entry in report.get("kernels", [])},
            "schema_version": CACHE_SCHEMA_VERSION,
            "tiles": {entry["cache_key"]: entry.get("artifact_id") for entry in report.get("tiles", [])},
        }
        atomic_write_json(self.index_path, payload)

    def key_manifest(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kernels": {}, "tiles": {}, "composites": {}}
        for path in sorted(self.kernel_dir.glob("*.meta.json"), key=lambda item: item.name):
            try:
                meta = read_json(path)
                artifact_id = path.name[: -len(".meta.json")]
                result["kernels"][artifact_id] = {
                    "cache_key": meta["cache_key"],
                    "kernel_keys": meta["kernel_keys"],
                }
            except Exception:
                continue
        for path in sorted(self.tile_dir.glob("*.meta.json"), key=lambda item: item.name):
            try:
                meta = read_json(path)
                artifact_id = path.name[: -len(".meta.json")]
                result["tiles"][artifact_id] = {
                    "cache_key": meta["cache_key"],
                    "kernel_refs": meta["kernel_refs"],
                }
            except Exception:
                continue
        for path in sorted(self.composite_dir.glob("*.meta.json"), key=lambda item: item.name):
            try:
                meta = read_json(path)
                artifact_id = path.name[: -len(".meta.json")]
                result["composites"][artifact_id] = {
                    "cache_key": meta["cache_key"],
                    "lower_composite_key": meta.get("lower_composite_key"),
                    "lower_tile_refs": meta.get("lower_tile_refs", []),
                    "tile_refs": meta.get("tile_refs", []),
                }
            except Exception:
                continue
        return result

    def artifact_meta_entries(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {"kernels": [], "tiles": [], "composites": []}
        for path in sorted(self.kernel_dir.glob("*.meta.json"), key=lambda item: item.name):
            try:
                meta = read_json(path)
                result["kernels"].append(
                    {
                        "artifact_id": path.name[: -len(".meta.json")],
                        "cache_key": meta["cache_key"],
                        "level": meta["level"],
                        "view": meta["view"],
                        "kernel_count": meta["allowed_count"],
                        "kernel_key_digest": _digest_list(meta["kernel_keys"]),
                        "arrays_fingerprint": meta["arrays_fingerprint"],
                    }
                )
            except Exception:
                continue
        for path in sorted(self.tile_dir.glob("*.meta.json"), key=lambda item: item.name):
            try:
                meta = read_json(path)
                result["tiles"].append(
                    {
                        "artifact_id": path.name[: -len(".meta.json")],
                        "cache_key": meta["cache_key"],
                        "col": meta["col"],
                        "kernel_refs": meta["kernel_refs"],
                        "level": meta["level"],
                        "row": meta["row"],
                        "view": meta["view"],
                        "arrays_fingerprint": meta["arrays_fingerprint"],
                    }
                )
            except Exception:
                continue
        for path in sorted(self.composite_dir.glob("*.meta.json"), key=lambda item: item.name):
            try:
                meta = read_json(path)
                result["composites"].append(
                    {
                        "artifact_id": path.name[: -len(".meta.json")],
                        "cache_key": meta["cache_key"],
                        "level": meta["level"],
                        "lower_tile_refs": meta.get("lower_tile_refs", []),
                        "lower_composite_key": meta.get("lower_composite_key"),
                        "tile_refs": meta.get("tile_refs", []),
                        "view": meta["view"],
                        "arrays_fingerprint": meta["arrays_fingerprint"],
                    }
                )
            except Exception:
                continue
        return result

    def read_artifact_arrays(self, kind: str, artifact_id: str) -> dict[str, np.ndarray] | None:
        _meta, arrays = self._load_artifact(kind, artifact_id)
        return arrays


def _digest_list(values: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_expected_kernel_ids(ids: Iterable[str]) -> Iterable[tuple[int, int]]:
    for artifact_id in ids:
        try:
            view_raw, level_raw = artifact_id.split("_L", 1)
            yield int(view_raw), int(level_raw)
        except (ValueError, AttributeError):
            continue
