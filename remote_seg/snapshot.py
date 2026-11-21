"""Versioned, atomically published global snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .classify import TileResult
from .errors import E_SNAPSHOT_STALE, Failure, SegmentationError
from .fusion import GlobalResult, RegionResult
from .util import (
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    read_json,
    sha256_file,
)


SCHEMA_VERSION = 1


@dataclass(slots=True)
class PendingTask:
    kind: str
    key: str
    stable_sort_key: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "stable_sort_key": self.stable_sort_key,
        }


@dataclass(slots=True)
class RuntimeState:
    tiles: dict[str, TileResult] = field(default_factory=dict)
    regions: dict[str, RegionResult] = field(default_factory=dict)
    global_result: GlobalResult | None = None
    pending: list[PendingTask] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    global_version: int = 0
    manifest: dict[str, Any] | None = None
    recovery_point: dict[str, Any] | None = None


def _safe_name(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    clean = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return f"{clean[:48]}-{digest}"


def core_state_fingerprint(
    *,
    config_hash: str,
    index_hash: str,
    fusion_version: str,
    postprocess_fingerprint: str,
    tiles: dict[str, TileResult],
    regions: dict[str, RegionResult],
    pending: list[PendingTask],
    failures: list[Failure],
) -> str:
    payload = {
        "config_hash": config_hash,
        "index_hash": index_hash,
        "fusion_version": fusion_version,
        "postprocess_fingerprint": postprocess_fingerprint,
        "tile_versions": {
            tile_id: {
                "cache_key": tile.cache_key,
                "result_version": tile.result_version,
            }
            for tile_id, tile in sorted(tiles.items())
        },
        "region_versions": {
            region_id: {
                "fingerprint": region.fingerprint,
                "region_version": region.region_version,
            }
            for region_id, region in sorted(regions.items())
        },
        "pending": [task.as_dict() for task in pending],
        "failures": [failure.as_dict() for failure in failures],
    }
    return canonical_hash(payload, prefix="snapshot-state-v1|")


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.generations = self.root / "generations"
        self.current_path = self.root / "CURRENT"
        self.generations.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        *,
        expected_config_hash: str | None = None,
        expected_index_hash: str | None = None,
        require_complete: bool = True,
    ) -> RuntimeState | None:
        if not self.current_path.exists():
            return None
        try:
            pointer = read_json(self.current_path)
            generation = pointer["generation"]
            state_fingerprint = pointer["state_fingerprint"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SegmentationError(
                E_SNAPSHOT_STALE,
                f"CURRENT pointer is invalid: {exc}",
                stage="recovery",
            ) from exc
        generation_dir = self.generations / generation
        manifest_path = generation_dir / "manifest.json"
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError) as exc:
            raise SegmentationError(
                E_SNAPSHOT_STALE,
                f"snapshot manifest is missing or invalid: {exc}",
                stage="recovery",
                details={"generation": generation},
            ) from exc
        if require_complete and manifest.get("complete") is not True:
            raise SegmentationError(
                E_SNAPSHOT_STALE,
                "snapshot generation is not marked complete",
                stage="recovery",
                details={"generation": generation},
            )
        if manifest.get("state_fingerprint") != state_fingerprint:
            raise SegmentationError(
                E_SNAPSHOT_STALE,
                "CURRENT state fingerprint does not match snapshot manifest",
                stage="recovery",
                details={"generation": generation},
            )
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise SegmentationError(
                E_SNAPSHOT_STALE,
                "snapshot manifest has no file checksums",
                stage="recovery",
                details={"generation": generation},
            )
        for relative, expected_hash in sorted(files.items()):
            path = generation_dir / relative
            if not path.is_file():
                raise SegmentationError(
                    E_SNAPSHOT_STALE,
                    f"snapshot file is missing: {relative}",
                    stage="recovery",
                    details={"generation": generation},
                )
            if sha256_file(path) != expected_hash:
                raise SegmentationError(
                    E_SNAPSHOT_STALE,
                    f"snapshot file checksum mismatch: {relative}",
                    stage="recovery",
                    details={"generation": generation},
                )
        if expected_config_hash is not None and manifest.get("config_hash") != expected_config_hash:
            raise SegmentationError(
                E_SNAPSHOT_STALE,
                "snapshot config hash does not match requested config",
                stage="recovery",
                details={
                    "generation": generation,
                    "snapshot_config_hash": manifest.get("config_hash"),
                    "expected_config_hash": expected_config_hash,
                },
            )
        if expected_index_hash is not None and manifest.get("index_hash") != expected_index_hash:
            raise SegmentationError(
                E_SNAPSHOT_STALE,
                "snapshot index hash does not match requested index",
                stage="recovery",
                details={
                    "generation": generation,
                    "snapshot_index_hash": manifest.get("index_hash"),
                    "expected_index_hash": expected_index_hash,
                },
            )
        return self._load_generation(generation_dir, manifest)

    def _load_generation(self, generation_dir: Path, manifest: dict[str, Any]) -> RuntimeState:
        tiles: dict[str, TileResult] = {}
        for tile_id, entry in sorted(manifest.get("tile_results", {}).items()):
            base = entry["base_name"]
            meta = read_json(generation_dir / "tiles" / f"{base}.json")
            with np.load(generation_dir / "tiles" / f"{base}.npz") as arrays:
                classes = arrays["classes"].copy()
                scores = arrays["scores"].copy()
            tiles[tile_id] = TileResult(
                tile_id=meta["tile_id"],
                row=meta["row"],
                col=meta["col"],
                cache_key=meta["cache_key"],
                content_fingerprint=meta["content_fingerprint"],
                band_order_version=meta["band_order_version"],
                result_version=meta["result_version"],
                classes=classes,
                scores=scores,
                feature_summary=meta["feature_summary"],
                score_summary=meta["score_summary"],
                class_histogram=meta["class_histogram"],
                windows=meta["windows"],
            )

        regions: dict[str, RegionResult] = {}
        for region_id, entry in sorted(manifest.get("fusion_regions", {}).items()):
            payload = read_json(generation_dir / "regions" / entry["file"])
            regions[region_id] = RegionResult(
                pair_id=payload["pair_id"],
                tile_ids=tuple(payload["tile_ids"]),
                overlap_bounds=tuple(payload["overlap_bounds"]),
                fingerprint=payload["fingerprint"],
                region_version=payload["region_version"],
                fusion_version=payload["fusion_version"],
                votes=payload["votes"],
                winner_histogram=payload["winner_histogram"],
            )

        global_result = None
        if manifest.get("global_file"):
            global_meta = read_json(generation_dir / manifest["global_file"])
            with np.load(generation_dir / manifest["global_array_file"]) as arrays:
                classes = arrays["classes"].copy()
            global_result = GlobalResult(
                bounds=tuple(global_meta["bounds"]),
                classes=classes,
                class_histogram=global_meta["class_histogram"],
                overlap_trace=global_meta["overlap_trace"],
                postprocess_records=global_meta["postprocess_records"],
                postprocess_components=global_meta["postprocess_components"],
                fusion_version=global_meta["fusion_version"],
            )

        failures = [
            Failure(
                code=item["code"],
                stage=item["stage"],
                message=item["message"],
                row=item.get("row"),
                col=item.get("col"),
                tile_id=item.get("tile_id"),
                injection_id=item.get("injection_id"),
                details=item.get("details", {}),
            )
            for item in manifest.get("failures", [])
        ]
        pending = [
            PendingTask(
                kind=item["kind"],
                key=item["key"],
                stable_sort_key=item["stable_sort_key"],
            )
            for item in manifest.get("pending", [])
        ]
        recovery_point = {
            "generation": manifest["generation"],
            "global_version": manifest["global_version"],
            "fusion_version": manifest["fusion_version"],
            "config_hash": manifest["config_hash"],
            "index_version": manifest["index_version"],
            "state_fingerprint": manifest["state_fingerprint"],
        }
        return RuntimeState(
            tiles=tiles,
            regions=regions,
            global_result=global_result,
            pending=pending,
            failures=failures,
            global_version=manifest["global_version"],
            manifest=manifest,
            recovery_point=recovery_point,
        )

    def commit(
        self,
        state: RuntimeState,
        *,
        config_hash: str,
        index_hash: str,
        index_version: str,
        fusion_version: str,
        postprocess_fingerprint: str,
        band_order_versions: list[str],
    ) -> dict[str, Any]:
        fingerprint = core_state_fingerprint(
            config_hash=config_hash,
            index_hash=index_hash,
            fusion_version=fusion_version,
            postprocess_fingerprint=postprocess_fingerprint,
            tiles=state.tiles,
            regions=state.regions,
            pending=state.pending,
            failures=state.failures,
        )
        current_state = None
        try:
            current_state = self.load()
        except SegmentationError:
            current_state = None
        if (
            current_state is not None
            and current_state.manifest is not None
            and current_state.manifest.get("state_fingerprint") == fingerprint
        ):
            state.global_version = current_state.global_version
            state.manifest = current_state.manifest
            state.recovery_point = current_state.recovery_point
            return current_state.recovery_point or {}

        next_version = (current_state.global_version if current_state else 0) + 1
        generation = f"g{next_version:08d}-{fingerprint[:12]}"
        temp_dir = Path(tempfile.mkdtemp(prefix=".building-", dir=self.generations))
        files: dict[str, str] = {}
        try:
            tiles_dir = temp_dir / "tiles"
            regions_dir = temp_dir / "regions"
            tiles_dir.mkdir()
            regions_dir.mkdir()
            tile_entries: dict[str, Any] = {}
            for tile_id, tile in sorted(state.tiles.items()):
                base = _safe_name(tile_id)
                np.savez_compressed(
                    tiles_dir / f"{base}.npz",
                    classes=tile.classes,
                    scores=tile.scores,
                )
                meta = {
                    "tile_id": tile.tile_id,
                    "row": tile.row,
                    "col": tile.col,
                    "cache_key": tile.cache_key,
                    "content_fingerprint": tile.content_fingerprint,
                    "band_order_version": tile.band_order_version,
                    "result_version": tile.result_version,
                    "feature_summary": tile.feature_summary,
                    "score_summary": tile.score_summary,
                    "class_histogram": tile.class_histogram,
                    "windows": tile.windows,
                }
                atomic_write_json(tiles_dir / f"{base}.json", meta)
                tile_entries[tile_id] = {
                    **tile.manifest_entry(),
                    "base_name": base,
                }
                files[f"tiles/{base}.npz"] = sha256_file(tiles_dir / f"{base}.npz")
                files[f"tiles/{base}.json"] = sha256_file(tiles_dir / f"{base}.json")

            region_entries: dict[str, Any] = {}
            for region_id, region in sorted(state.regions.items()):
                filename = f"{_safe_name(region_id)}.json"
                payload = {
                    "pair_id": region.pair_id,
                    "tile_ids": list(region.tile_ids),
                    "overlap_bounds": list(region.overlap_bounds),
                    "fingerprint": region.fingerprint,
                    "region_version": region.region_version,
                    "fusion_version": region.fusion_version,
                    "votes": region.votes,
                    "winner_histogram": region.winner_histogram,
                }
                atomic_write_json(regions_dir / filename, payload)
                region_entries[region_id] = {
                    **region.manifest_entry(),
                    "file": filename,
                }
                files[f"regions/{filename}"] = sha256_file(regions_dir / filename)

            global_file = None
            global_array_file = None
            if state.global_result is not None:
                global_file = "global.json"
                global_array_file = "global.npz"
                np.savez_compressed(
                    temp_dir / global_array_file,
                    classes=state.global_result.classes,
                )
                global_payload = {
                    "bounds": list(state.global_result.bounds),
                    "class_histogram": state.global_result.class_histogram,
                    "overlap_trace": state.global_result.overlap_trace,
                    "postprocess_records": state.global_result.postprocess_records,
                    "postprocess_components": state.global_result.postprocess_components,
                    "fusion_version": state.global_result.fusion_version,
                }
                atomic_write_json(temp_dir / global_file, global_payload)
                files[global_file] = sha256_file(temp_dir / global_file)
                files[global_array_file] = sha256_file(temp_dir / global_array_file)

            manifest = {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "generation": generation,
                "state_fingerprint": fingerprint,
                "global_version": next_version,
                "config_hash": config_hash,
                "index_hash": index_hash,
                "index_version": index_version,
                "fusion_version": fusion_version,
                "band_order_versions": sorted(set(band_order_versions)),
                "postprocess_fingerprint": postprocess_fingerprint,
                "tile_results": tile_entries,
                "fusion_regions": region_entries,
                "pending": [task.as_dict() for task in state.pending],
                "failures": [failure.as_dict() for failure in state.failures],
                "global_file": global_file,
                "global_array_file": global_array_file,
                "files": files,
            }
            atomic_write_json(temp_dir / "manifest.json", manifest)
            final_dir = self.generations / generation
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(temp_dir, final_dir)
            pointer = {
                "generation": generation,
                "global_version": next_version,
                "fusion_version": fusion_version,
                "state_fingerprint": fingerprint,
            }
            atomic_write_text(
                self.current_path,
                json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n",
            )
            state.global_version = next_version
            state.manifest = manifest
            state.recovery_point = {
                "generation": generation,
                "global_version": next_version,
                "fusion_version": fusion_version,
                "config_hash": config_hash,
                "index_version": index_version,
                "state_fingerprint": fingerprint,
            }
            return state.recovery_point
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
