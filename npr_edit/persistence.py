"""Atomic generation commits, crash recovery and orphan accounting."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import E_RECOVERY, E_SNAPSHOT_STALE, E_WORKSPACE, NPRError
from .geometry import BBox, FlatElement, FlatSegment, Point
from .model import Scene, scene_from_project_dict
from .snapshot import ElementRaster, RenderSnapshot, build_snapshot
from .util import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    fsync_dir,
    read_json,
    sha256_bytes,
    sha256_json,
)


CURRENT_FORMAT = "npr-current-v1"
CLOSED_FORMAT = "npr-closed-v1"
MANIFEST_FORMAT = "npr-state-manifest-v1"
ORPHAN_REPORT_FORMAT = "npr-orphan-report-v1"


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    scene: Scene
    snapshot: RenderSnapshot
    generation: str
    version: int
    clean_shutdown: bool
    closed_marker_present: bool
    closed_marker_valid: bool
    orphan_entries: tuple[dict[str, Any], ...]
    ignored_uncommitted: tuple[str, ...]
    recovered_scene_hash: str
    recovered_snapshot_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "version": self.version,
            "clean_shutdown": self.clean_shutdown,
            "closed_marker_present": self.closed_marker_present,
            "closed_marker_valid": self.closed_marker_valid,
            "orphan_count": len(self.orphan_entries),
            "orphan_entries": list(self.orphan_entries),
            "ignored_uncommitted": list(self.ignored_uncommitted),
            "recovered_scene_hash": self.recovered_scene_hash,
            "recovered_snapshot_hash": self.recovered_snapshot_hash,
        }


def _flat_from_dict(raw: Mapping[str, Any]) -> FlatElement:
    try:
        points = tuple(Point(float(point[0]), float(point[1])) for point in raw["points"])
        segments = tuple(
            FlatSegment(
                index=int(segment["index"]),
                kind=str(segment["kind"]),
                points=tuple(Point(float(point[0]), float(point[1])) for point in segment["points"]),
                error_estimate=float(segment["error_estimate"]),
                vertex_count=int(segment["vertex_count"]),
            )
            for segment in raw["segments"]
        )
        bbox_raw = raw["bbox"]
        bbox = BBox(float(bbox_raw["min_x"]), float(bbox_raw["min_y"]), float(bbox_raw["max_x"]), float(bbox_raw["max_y"]))
        flat = FlatElement(
            element_id=str(raw["element_id"]),
            element_hash=str(raw["element_hash"]),
            layer_id=str(raw["layer_id"]),
            layer_order=int(raw["layer_order"]),
            closed=bool(raw["closed"]),
            stroke_width=float(raw["stroke_width"]),
            tolerance=float(raw["tolerance"]),
            points=points,
            indices=tuple(int(index) for index in raw.get("indices", [])),
            segments=segments,
            bbox=bbox,
            geometry_hash=str(raw["geometry_hash"]),
            skeleton_hash=str(raw["skeleton_hash"]),
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise NPRError(E_RECOVERY, "persisted geometry record is invalid") from exc
    expected_geometry_payload = {
        "element_id": flat.element_id,
        "element_hash": flat.element_hash,
        "tolerance": flat.tolerance,
        "closed": flat.closed,
        "points": [[point.x, point.y] for point in flat.points],
        "indices": list(flat.indices),
        "segments": [segment.as_dict() for segment in flat.segments],
        "bbox": flat.bbox.as_dict(),
    }
    if sha256_json(expected_geometry_payload) != flat.geometry_hash:
        raise NPRError(E_RECOVERY, "persisted geometry hash is inconsistent", element_id=flat.element_id)
    return flat


def _layer_content_hash(layer_any: Any, buffer: bytes) -> str:
    return sha256_json(
        {
            "layer_id": layer_any.id,
            "order": layer_any.order,
            "pixel_hash": sha256_bytes(buffer),
        }
    )


def _tree_digest(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class PersistenceStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.state = self.root / "state"
        self.generations = self.state / "generations"
        self.tmp = self.state / "tmp"
        self.orphans = self.state / "orphans"
        self.current_path = self.state / "CURRENT"
        self.closed_path = self.state / "CLOSED"

    # ------------------------------------------------------------------
    # Layout and durability

    def ensure_layout(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.generations.mkdir(parents=True, exist_ok=True)
            self.tmp.mkdir(parents=True, exist_ok=True)
            self.orphans.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise NPRError(E_WORKSPACE, "workspace is not writable", root=str(self.root)) from exc

    def check_writable(self) -> None:
        self.ensure_layout()
        try:
            fd, name = tempfile.mkstemp(prefix=".npr-write-", dir=self.root)
            os.close(fd)
            os.unlink(name)
        except OSError as exc:
            raise NPRError(E_WORKSPACE, "workspace write probe failed", root=str(self.root)) from exc

    def reset_state(self) -> None:
        """Remove only this application's state directory."""

        expected = self.root / "state"
        if self.state != expected or self.state.parent != self.root:
            raise NPRError(E_WORKSPACE, "refusing to reset an unexpected state path")
        if self.state.exists():
            shutil.rmtree(self.state)
        self.ensure_layout()

    def _remove_closed_marker(self) -> None:
        try:
            self.closed_path.unlink()
            fsync_dir(self.state)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise NPRError(E_WORKSPACE, "cannot remove the previous close marker") from exc

    # ------------------------------------------------------------------
    # Commit path

    def _manifest_and_files(
        self,
        scene: Scene,
        snapshot: RenderSnapshot,
        *,
        closed: bool,
    ) -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes]]:
        snapshot.verify()
        if snapshot.scene_hash != scene.content_hash:
            raise NPRError(E_SNAPSHOT_STALE, "snapshot and scene hashes do not match during persistence")
        layer_manifest = {layer.id: f"layers/layer-{index:04d}.bgra" for index, layer in enumerate(scene.layers)}
        sprite_entries: list[dict[str, Any]] = []
        sprite_bytes = bytearray()
        for element_id in sorted(snapshot.element_rasters):
            raster = snapshot.element_rasters[element_id]
            metadata = raster.metadata()
            metadata["offset"] = len(sprite_bytes)
            metadata["byte_length"] = len(raster.pixels)
            sprite_entries.append(metadata)
            sprite_bytes.extend(raster.pixels)
        core = {
            "format": MANIFEST_FORMAT,
            "schema_version": scene.schema_version,
            "version": int(snapshot.version),
            "closed": bool(closed),
            "complete": True,
            "scene_hash": scene.content_hash,
            "snapshot_hash": snapshot.snapshot_hash,
            "image_hash": snapshot.image_hash,
            "mode": snapshot.mode,
            "viewport": [int(snapshot.viewport[0]), int(snapshot.viewport[1])],
            "element_count": len(scene.elements),
            "layer_count": len(scene.layers),
            "skeleton_hashes": dict(sorted(snapshot.skeleton_hashes.items())),
            "render_hashes": dict(sorted(snapshot.element_hashes.items())),
            "layer_hashes": dict(sorted(snapshot.layer_hashes.items())),
            "geometry_hashes": dict(sorted(snapshot.geometry_hashes.items())),
            "strategy": dict(snapshot.strategy),
            "cache": {
                "layers": layer_manifest,
                "sprites_index": "sprites.json",
                "sprites_data": "sprites.bin",
                "geometry": "geometry.json",
            },
        }
        manifest_digest = sha256_json(core)
        generation = f"g{snapshot.version:08d}-{manifest_digest[:16]}"
        manifest = dict(core)
        manifest["generation"] = generation
        files: dict[str, bytes] = {
            "project.json": scene.project_bytes,
            "manifest.json": canonical_json_bytes(manifest) + b"\n",
            "snapshot.bgra": snapshot.image,
            "geometry.json": canonical_json_bytes([snapshot.flat_geometry[element_id].as_dict() for element_id in sorted(snapshot.flat_geometry)]) + b"\n",
            "sprites.json": canonical_json_bytes(sprite_entries) + b"\n",
            "sprites.bin": bytes(sprite_bytes),
        }
        for layer in scene.layers:
            buffer = snapshot.layer_buffers.get(layer.id)
            if buffer is None:
                raise NPRError(E_SNAPSHOT_STALE, "snapshot is missing a layer buffer", layer_id=layer.id)
            files[layer_manifest[layer.id]] = buffer
        return manifest, files, {layer.id: snapshot.layer_hashes[layer.id] for layer in scene.layers}

    def _write_tree(self, target: Path, files: Mapping[str, bytes]) -> None:
        target.mkdir(parents=True, exist_ok=False)
        for relative, data in sorted(files.items()):
            path = target / relative
            atomic_write_bytes(path, data)
        fsync_dir(target)

    def _install_generation(self, generation: str, files: Mapping[str, bytes]) -> Path:
        final_dir = self.generations / generation
        if final_dir.exists():
            return final_dir
        temp_dir = self.tmp / f"gen-{generation}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        self._write_tree(temp_dir, files)
        try:
            os.replace(temp_dir, final_dir)
        except OSError:
            if final_dir.exists():
                shutil.rmtree(temp_dir)
            else:
                raise
        fsync_dir(self.generations)
        return final_dir

    def commit(self, scene: Scene, snapshot: RenderSnapshot, *, closed: bool = False) -> dict[str, Any]:
        self.check_writable()
        manifest, files, _layer_hashes = self._manifest_and_files(scene, snapshot, closed=closed)
        generation = manifest["generation"]
        generation_dir = self._install_generation(generation, files)
        pointer = {
            "format": CURRENT_FORMAT,
            "generation": generation,
            "version": snapshot.version,
            "scene_hash": scene.content_hash,
            "snapshot_hash": snapshot.snapshot_hash,
        }
        self._remove_closed_marker()
        atomic_write_json(self.current_path, pointer)
        committed_marker = {
            "format": "npr-generation-committed-v1",
            "generation": generation,
            "version": snapshot.version,
            "scene_hash": scene.content_hash,
            "snapshot_hash": snapshot.snapshot_hash,
        }
        atomic_write_json(generation_dir / "COMMITTED", committed_marker)
        return manifest

    def close(self, scene: Scene, snapshot: RenderSnapshot) -> dict[str, Any]:
        manifest = self.commit(scene, snapshot, closed=True)
        marker = {
            "format": CLOSED_FORMAT,
            "clean": True,
            "generation": manifest["generation"],
            "version": snapshot.version,
            "scene_hash": scene.content_hash,
            "snapshot_hash": snapshot.snapshot_hash,
        }
        atomic_write_json(self.closed_path, marker)
        return marker

    def write_uncommitted_generation(self, scene: Scene, snapshot: RenderSnapshot) -> Path:
        """Stage files as if a writer was killed just before publishing CURRENT."""

        self.ensure_layout()
        manifest, files, _ = self._manifest_and_files(scene, snapshot, closed=False)
        generation = manifest["generation"]
        temp_dir = self.tmp / f"crash-{generation}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        # Deliberately leave out layer/sprites and the final manifest.  This
        # represents an interrupted write that must never be treated as stable.
        atomic_write_bytes(temp_dir / "project.json", files["project.json"])
        partial_manifest = dict(manifest)
        partial_manifest["complete"] = False
        atomic_write_json(temp_dir / "manifest.partial.json", partial_manifest)
        atomic_write_bytes(temp_dir / "snapshot.bgra.partial", snapshot.image[: max(1, len(snapshot.image) // 3)])
        fsync_dir(temp_dir)
        return temp_dir

    # ------------------------------------------------------------------
    # Recovery path

    def _read_pointer(self) -> dict[str, Any] | None:
        if not self.current_path.exists():
            return None
        try:
            pointer = read_json(self.current_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise NPRError(E_RECOVERY, "CURRENT pointer is not valid JSON") from exc
        if pointer.get("format") != CURRENT_FORMAT:
            raise NPRError(E_RECOVERY, "CURRENT pointer has an unsupported format")
        for key in ("generation", "version", "scene_hash", "snapshot_hash"):
            if key not in pointer:
                raise NPRError(E_RECOVERY, "CURRENT pointer is incomplete", missing=key)
        return pointer

    def _load_sprites(self, generation_dir: Path, manifest: Mapping[str, Any]) -> dict[str, ElementRaster]:
        cache = manifest.get("cache", {})
        index_path = generation_dir / str(cache.get("sprites_index", "sprites.json"))
        data_path = generation_dir / str(cache.get("sprites_data", "sprites.bin"))
        try:
            entries = read_json(index_path)
            data = data_path.read_bytes()
        except (OSError, json.JSONDecodeError) as exc:
            raise NPRError(E_RECOVERY, "persisted sprite cache is missing or invalid") from exc
        if not isinstance(entries, list):
            raise NPRError(E_RECOVERY, "sprite cache index must be a list")
        result: dict[str, ElementRaster] = {}
        for entry in entries:
            try:
                offset = int(entry["offset"])
                length = int(entry["byte_length"])
                pixels = data[offset : offset + length]
                if len(pixels) != length or sha256_bytes(pixels) != entry["pixel_hash"]:
                    raise NPRError(E_RECOVERY, "sprite pixels do not match their hash", element_id=entry.get("element_id"))
                raster = ElementRaster(
                    element_id=str(entry["element_id"]),
                    element_hash=str(entry["element_hash"]),
                    geometry_hash=str(entry["geometry_hash"]),
                    render_key=str(entry["render_key"]),
                    render_hash=str(entry["render_hash"]),
                    mode=str(entry["mode"]),
                    x=int(entry["x"]),
                    y=int(entry["y"]),
                    width=int(entry["width"]),
                    height=int(entry["height"]),
                    pixels=pixels,
                )
            except (KeyError, TypeError, ValueError) as exc:
                if isinstance(exc, NPRError):
                    raise
                raise NPRError(E_RECOVERY, "sprite cache entry is invalid") from exc
            result[raster.element_id] = raster
        return result

    def _load_generation(self, pointer: Mapping[str, Any]) -> tuple[Scene, RenderSnapshot, dict[str, Any]]:
        generation = str(pointer["generation"])
        generation_dir = self.generations / generation
        if not generation_dir.is_dir():
            raise NPRError(E_RECOVERY, "CURRENT references a missing generation", generation=generation)
        try:
            manifest = read_json(generation_dir / "manifest.json")
            project_bytes = (generation_dir / "project.json").read_bytes()
            image = (generation_dir / "snapshot.bgra").read_bytes()
        except (OSError, json.JSONDecodeError) as exc:
            raise NPRError(E_RECOVERY, "committed generation is incomplete") from exc
        if manifest.get("format") != MANIFEST_FORMAT or not manifest.get("complete"):
            raise NPRError(E_RECOVERY, "committed generation does not have a complete manifest")
        if str(manifest.get("generation")) != generation:
            raise NPRError(E_RECOVERY, "generation id mismatch", expected=generation, got=manifest.get("generation"))
        if str(pointer.get("snapshot_hash")) != str(manifest.get("snapshot_hash")):
            raise NPRError(E_RECOVERY, "CURRENT and manifest snapshot hashes disagree")
        try:
            raw = json.loads(project_bytes.decode("utf-8"))
            from .formats import migrate_project

            scene = scene_from_project_dict(migrate_project(raw))
        except NPRError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise NPRError(E_RECOVERY, "committed project JSON is invalid") from exc
        if scene.content_hash != manifest.get("scene_hash") or scene.content_hash != pointer.get("scene_hash"):
            raise NPRError(E_RECOVERY, "committed project hash does not match its manifest")
        if sha256_bytes(image) != manifest.get("image_hash"):
            raise NPRError(E_RECOVERY, "committed BGRA image hash does not match its manifest")
        try:
            geometry_entries = read_json(generation_dir / "geometry.json")
        except (OSError, json.JSONDecodeError) as exc:
            raise NPRError(E_RECOVERY, "committed geometry cache is invalid") from exc
        if not isinstance(geometry_entries, list):
            raise NPRError(E_RECOVERY, "committed geometry cache must be a list")
        flats = {entry["element_id"]: _flat_from_dict(entry) for entry in geometry_entries}
        rasters = self._load_sprites(generation_dir, manifest)
        if set(rasters) != {element.id for element in scene.elements}:
            raise NPRError(E_RECOVERY, "committed sprite cache does not cover the scene exactly")
        layer_buffers: dict[str, bytes] = {}
        for layer in scene.layers:
            relative = manifest.get("cache", {}).get("layers", {}).get(layer.id)
            if not relative:
                raise NPRError(E_RECOVERY, "manifest is missing a layer cache entry", layer_id=layer.id)
            try:
                buffer = (generation_dir / str(relative)).read_bytes()
            except OSError as exc:
                raise NPRError(E_RECOVERY, "committed layer cache is missing", layer_id=layer.id) from exc
            if len(buffer) != scene.canvas_width * scene.canvas_height * 4 and len(buffer) != int(manifest["viewport"][0]) * int(manifest["viewport"][1]) * 4:
                raise NPRError(E_RECOVERY, "committed layer has the wrong size", layer_id=layer.id)
            expected_hash = manifest.get("layer_hashes", {}).get(layer.id)
            if _layer_content_hash(layer, buffer) != expected_hash:
                raise NPRError(E_RECOVERY, "committed layer hash is inconsistent", layer_id=layer.id)
            layer_buffers[layer.id] = buffer
        if set(flats) != {element.id for element in scene.elements}:
            raise NPRError(E_RECOVERY, "committed geometry cache does not cover the scene exactly")
        for element in scene.elements:
            flat = flats[element.id]
            raster = rasters[element.id]
            if flat.element_hash != element.content_hash:
                raise NPRError(E_RECOVERY, "geometry cache belongs to a different element state", element_id=element.id)
            if flat.geometry_hash != manifest.get("geometry_hashes", {}).get(element.id):
                raise NPRError(E_RECOVERY, "geometry cache hash does not match the manifest", element_id=element.id)
            if flat.skeleton_hash != manifest.get("skeleton_hashes", {}).get(element.id):
                raise NPRError(E_RECOVERY, "skeleton hash does not match the manifest", element_id=element.id)
            if raster.element_hash != element.content_hash or raster.geometry_hash != flat.geometry_hash:
                raise NPRError(E_RECOVERY, "sprite cache belongs to a different element state", element_id=element.id)
            if raster.mode != manifest.get("mode"):
                raise NPRError(E_RECOVERY, "sprite cache mode does not match the snapshot", element_id=element.id)
        snapshot = build_snapshot(
            version=int(manifest["version"]),
            mode=str(manifest["mode"]),
            viewport=(int(manifest["viewport"][0]), int(manifest["viewport"][1])),
            scene_hash=str(manifest["scene_hash"]),
            image=image,
            layer_buffers=layer_buffers,
            layer_hashes={str(key): str(value) for key, value in manifest["layer_hashes"].items()},
            element_rasters=rasters,
            skeleton_hashes={str(key): str(value) for key, value in manifest["skeleton_hashes"].items()},
            geometry_hashes={str(key): str(value) for key, value in manifest["geometry_hashes"].items()},
            flat_geometry=flats,
            strategy=dict(manifest.get("strategy", {})),
        )
        if snapshot.snapshot_hash != manifest.get("snapshot_hash"):
            raise NPRError(E_RECOVERY, "restored snapshot hash does not match the committed manifest")
        if dict(snapshot.element_hashes) != {str(key): str(value) for key, value in manifest.get("render_hashes", {}).items()}:
            raise NPRError(E_RECOVERY, "restored element render hashes do not match the manifest")
        return scene, snapshot, manifest

    def _move_to_orphans(self, source: Path, kind: str, entries: list[dict[str, Any]]) -> None:
        digest = _tree_digest(source) if source.is_dir() else sha256_bytes(source.read_bytes())
        base_name = f"{kind}-{source.name}"
        destination = self.orphans / base_name
        if destination.exists():
            destination = self.orphans / f"{kind}-{source.name}-{digest[:12]}"
            if destination.exists():
                shutil.rmtree(source) if source.is_dir() else source.unlink()
                return
        os.replace(source, destination)
        entries.append(
            {
                "kind": kind,
                "name": source.name,
                "orphan_name": destination.name,
                "content_hash": digest,
            }
        )

    def _scan_orphans(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for item in sorted(self.orphans.iterdir()) if self.orphans.exists() else []:
            if item.name == "RECOVERY.json":
                continue
            digest = _tree_digest(item) if item.is_dir() else sha256_bytes(item.read_bytes())
            entries.append({"kind": "existing", "name": item.name, "orphan_name": item.name, "content_hash": digest})
        return entries

    def _valid_closed_marker(self, pointer: Mapping[str, Any]) -> tuple[bool, bool]:
        if not self.closed_path.exists():
            return False, False
        try:
            marker = read_json(self.closed_path)
        except (OSError, json.JSONDecodeError):
            return True, False
        valid = (
            marker.get("format") == CLOSED_FORMAT
            and marker.get("clean") is True
            and marker.get("generation") == pointer.get("generation")
            and marker.get("version") == pointer.get("version")
            and marker.get("scene_hash") == pointer.get("scene_hash")
            and marker.get("snapshot_hash") == pointer.get("snapshot_hash")
        )
        return True, valid

    def recover(self) -> RecoveryReport | None:
        self.ensure_layout()
        pointer = self._read_pointer()
        if pointer is None:
            existing = [item for item in self.generations.iterdir() if item.is_dir()]
            if existing:
                raise NPRError(E_RECOVERY, "generations exist but CURRENT is missing")
            return None
        scene, snapshot, manifest = self._load_generation(pointer)
        new_orphans: list[dict[str, Any]] = []
        current_generation = str(pointer["generation"])
        for item in sorted(self.generations.iterdir()):
            if not item.is_dir() or item.name == current_generation:
                continue
            if (item / "COMMITTED").exists():
                # A previously committed generation is retained as history.
                continue
            kind = "staged_generation"
            try:
                candidate_manifest = read_json(item / "manifest.json")
                if not candidate_manifest.get("complete"):
                    kind = "incomplete_generation"
            except (OSError, json.JSONDecodeError):
                kind = "incomplete_generation"
            self._move_to_orphans(item, kind, new_orphans)
        for item in sorted(self.tmp.iterdir()):
            self._move_to_orphans(item, "temporary", new_orphans)
        ignored_uncommitted = tuple(sorted(entry["name"] for entry in new_orphans))
        all_orphans = self._scan_orphans()
        report_payload = {
            "format": ORPHAN_REPORT_FORMAT,
            "current_generation": current_generation,
            "new_count": len(new_orphans),
            "new_entries": new_orphans,
            "orphan_entries": all_orphans,
        }
        atomic_write_json(self.orphans / "RECOVERY.json", report_payload)
        marker_present, marker_valid = self._valid_closed_marker(pointer)
        return RecoveryReport(
            scene=scene,
            snapshot=snapshot,
            generation=current_generation,
            version=int(pointer["version"]),
            clean_shutdown=marker_valid,
            closed_marker_present=marker_present,
            closed_marker_valid=marker_valid,
            orphan_entries=tuple(all_orphans),
            ignored_uncommitted=ignored_uncommitted,
            recovered_scene_hash=scene.content_hash,
            recovered_snapshot_hash=snapshot.snapshot_hash,
        )
