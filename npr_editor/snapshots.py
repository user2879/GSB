"""Atomic, crash-safe snapshot persistence for projects and render caches."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import NprError, require
from .model import Project, canonical_hash, canonical_json, project_from_dict, version_stamp_key
from .png import encode_png
from .render import RenderEngine, RenderResult, restore_render_cache, verify_render_hashes

SNAPSHOT_FORMAT = "npr-snapshot-v1"
COMMITTED_NAME = "COMMITTED"
MANIFEST_NAME = "manifest.json"

FaultHook = Callable[[str], None]


@dataclass(frozen=True)
class SnapshotInfo:
    path: Path
    version_stamp: str
    snapshot_hash: str
    project_hash: str
    mode: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class RecoveryReport:
    latest: SnapshotInfo | None
    committed_snapshots: tuple[SnapshotInfo, ...]
    reclaimed_orphans: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "latest": None
            if self.latest is None
            else {
                "version_stamp": self.latest.version_stamp,
                "snapshot_hash": self.latest.snapshot_hash,
                "project_hash": self.latest.project_hash,
                "mode": self.latest.mode,
            },
            "committed_count": len(self.committed_snapshots),
            "reclaimed_orphans": list(self.reclaimed_orphans),
            "status": "recovered" if self.latest is not None else "empty",
        }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise NprError(
            "E_SNAPSHOT_STALE", "snapshot JSON cannot be read", path=str(path)
        ) from exc
    require(
        isinstance(value, dict),
        "E_SNAPSHOT_STALE",
        "snapshot JSON root must be an object",
        path=str(path),
    )
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    base = root.resolve()
    require(
        candidate == base or base in candidate.parents,
        "E_SNAPSHOT_STALE",
        "snapshot manifest references a path outside its directory",
        path=relative,
    )
    return candidate


class SnapshotStore:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace)
        self.root = self.workspace / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True)

    def _manifest_for(
        self, *, project: Project, mode: str, result: RenderResult, exported: dict[str, dict[str, object]]
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        require(
            mode == result.mode,
            "E_RENDER_MISMATCH",
            "snapshot mode does not match render result",
            mode=mode,
            result_mode=result.mode,
        )
        verify_render_hashes(result.shape_hashes, {shape: data["render_hash"] for shape, data in exported.items()})
        shape_files: dict[str, str] = {}
        raw_files: dict[str, bytes] = {}
        for shape_id, payload in sorted(exported.items()):
            filename = hashlib.sha256(shape_id.encode("utf-8")).hexdigest()[:24] + ".rgba"
            relative = f"shapes/{filename}"
            shape_files[shape_id] = relative
            raw_files[relative] = payload["rgba"]  # type: ignore[assignment]
        png_bytes = encode_png(result.image)
        require(
            hashlib.sha256(png_bytes).hexdigest() == result.png_sha256,
            "E_RENDER_MISMATCH",
            "render result PNG hash is inconsistent",
        )
        manifest: dict[str, Any] = {
            "format": SNAPSHOT_FORMAT,
            "version_stamp": project.version_stamp,
            "project_hash": project.content_hash(),
            "geometry_hashes": dict(sorted(result.geometry_hashes.items())),
            "mode": mode,
            "render_hashes": dict(sorted(result.shape_hashes.items())),
            "render_cache_keys": {
                shape_id: payload["cache_key"] for shape_id, payload in sorted(exported.items())
            },
            "render_config": result_config_dict(result),
            "full_pixel_sha256": result.pixel_sha256,
            "full_png_sha256": result.png_sha256,
            "shape_files": shape_files,
        }
        files = {
            "project.json": canonical_json(project.to_dict()).encode("utf-8") + b"\n",
            "render.png": png_bytes,
            **raw_files,
        }
        manifest["files"] = {
            relative: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
            for relative, payload in sorted(files.items())
        }
        manifest["snapshot_hash"] = canonical_hash(manifest)
        return manifest, files

    def _scan(self, *, reclaim: bool) -> tuple[list[SnapshotInfo], list[str]]:
        committed: list[SnapshotInfo] = []
        orphans: list[str] = []
        for entry in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not entry.is_dir():
                continue
            if entry.name.startswith(".staging-"):
                if reclaim:
                    shutil.rmtree(entry)
                    orphans.append(entry.name)
                continue
            commit_file = entry / COMMITTED_NAME
            if not commit_file.is_file():
                if reclaim:
                    shutil.rmtree(entry)
                    orphans.append(entry.name)
                continue
            manifest = _read_json(entry / MANIFEST_NAME)
            require(
                manifest.get("format") == SNAPSHOT_FORMAT,
                "E_SNAPSHOT_STALE",
                "unknown snapshot format",
                path=str(entry),
            )
            snapshot_hash = manifest.get("snapshot_hash")
            require(
                isinstance(snapshot_hash, str),
                "E_SNAPSHOT_STALE",
                "snapshot hash missing",
                path=str(entry),
            )
            try:
                marker_hash = commit_file.read_text(encoding="ascii").strip()
            except OSError as exc:
                raise NprError(
                    "E_SNAPSHOT_STALE", "commit marker cannot be read", path=str(entry)
                ) from exc
            require(
                marker_hash == snapshot_hash,
                "E_SNAPSHOT_STALE",
                "snapshot commit marker does not match manifest",
                path=str(entry),
            )
            unhashed = dict(manifest)
            unhashed.pop("snapshot_hash", None)
            require(
                canonical_hash(unhashed) == snapshot_hash,
                "E_SNAPSHOT_STALE",
                "snapshot manifest content hash is invalid",
                path=str(entry),
            )
            files = manifest.get("files")
            require(
                isinstance(files, dict) and files,
                "E_SNAPSHOT_STALE",
                "snapshot manifest has no files",
                path=str(entry),
            )
            for relative, metadata in files.items():
                require(
                    isinstance(relative, str) and isinstance(metadata, dict),
                    "E_SNAPSHOT_STALE",
                    "snapshot file inventory is malformed",
                )
                file_path = _safe_relative(entry, relative)
                require(
                    file_path.is_file(),
                    "E_SNAPSHOT_STALE",
                    "snapshot file is missing",
                    file=relative,
                )
                require(
                    _hash_file(file_path) == metadata.get("sha256"),
                    "E_SNAPSHOT_STALE",
                    "snapshot file hash mismatch",
                    file=relative,
                )
            version_stamp = manifest.get("version_stamp")
            project_hash = manifest.get("project_hash")
            mode = manifest.get("mode")
            require(
                isinstance(version_stamp, str)
                and version_stamp.isdigit()
                and isinstance(project_hash, str)
                and isinstance(mode, str),
                "E_SNAPSHOT_STALE",
                "snapshot metadata is incomplete",
                path=str(entry),
            )
            committed.append(
                SnapshotInfo(
                    path=entry,
                    version_stamp=version_stamp,
                    snapshot_hash=snapshot_hash,
                    project_hash=project_hash,
                    mode=mode,
                    manifest=manifest,
                )
            )
        by_stamp: dict[int, SnapshotInfo] = {}
        for snapshot in committed:
            key = version_stamp_key(snapshot.version_stamp)
            previous = by_stamp.get(key)
            if previous is not None and previous.snapshot_hash != snapshot.snapshot_hash:
                raise NprError(
                    "E_SNAPSHOT_STALE",
                    "two snapshots share a version stamp with different content",
                    version_stamp=snapshot.version_stamp,
                )
            if previous is None or snapshot.snapshot_hash > previous.snapshot_hash:
                by_stamp[key] = snapshot
        return list(by_stamp.values()), orphans

    def committed_snapshots(self) -> list[SnapshotInfo]:
        snapshots, _ = self._scan(reclaim=False)
        return sorted(snapshots, key=lambda snapshot: version_stamp_key(snapshot.version_stamp))

    def latest(self) -> SnapshotInfo | None:
        snapshots = self.committed_snapshots()
        return snapshots[-1] if snapshots else None

    def recover(self) -> RecoveryReport:
        snapshots, orphans = self._scan(reclaim=True)
        snapshots.sort(key=lambda snapshot: version_stamp_key(snapshot.version_stamp))
        return RecoveryReport(
            latest=snapshots[-1] if snapshots else None,
            committed_snapshots=tuple(snapshots),
            reclaimed_orphans=tuple(sorted(orphans)),
        )

    def commit(
        self,
        project: Project,
        *,
        mode: str,
        result: RenderResult,
        engine: RenderEngine,
        fault_hook: FaultHook | None = None,
    ) -> SnapshotInfo:
        require(
            result.mode == mode,
            "E_RENDER_MISMATCH",
            "commit mode differs from render result",
        )
        require(
            engine.project.content_hash() == project.content_hash(),
            "E_RENDER_MISMATCH",
            "render engine is not bound to the project being committed",
        )
        exported = engine.export_cache(mode)
        manifest, files = self._manifest_for(
            project=project, mode=mode, result=result, exported=exported
        )
        latest = self.latest()
        new_key = version_stamp_key(project.version_stamp)
        if latest is not None:
            latest_key = version_stamp_key(latest.version_stamp)
            if new_key < latest_key:
                raise NprError(
                    "E_SNAPSHOT_STALE",
                    "snapshot version stamp is older than the latest committed snapshot",
                    version_stamp=project.version_stamp,
                    latest=latest.version_stamp,
                )
            if new_key == latest_key:
                if latest.snapshot_hash == manifest["snapshot_hash"]:
                    return latest
                raise NprError(
                    "E_SNAPSHOT_STALE",
                    "snapshot version stamp already exists with different content",
                    version_stamp=project.version_stamp,
                )

        staging = self.root / f".staging-{os.getpid()}-{uuid.uuid4().hex}"
        staging.mkdir(parents=False, exist_ok=False)
        try:
            for relative, payload in sorted(files.items()):
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(destination, payload)
            atomic_write_bytes(
                staging / MANIFEST_NAME,
                (canonical_json(manifest) + "\n").encode("utf-8"),
            )
            if fault_hook is not None:
                fault_hook("after_payload")
            atomic_write_bytes(
                staging / COMMITTED_NAME,
                (manifest["snapshot_hash"] + "\n").encode("ascii"),
            )
            if fault_hook is not None:
                fault_hook("before_publish")
            _fsync_directory(staging)
            final = self.root / f"{project.version_stamp}-{manifest['snapshot_hash'][:16]}"
            if final.exists():
                if _read_json(final / MANIFEST_NAME).get("snapshot_hash") == manifest["snapshot_hash"]:
                    shutil.rmtree(staging, ignore_errors=True)
                else:
                    raise NprError(
                        "E_SNAPSHOT_STALE",
                        "snapshot destination collision",
                        path=str(final),
                    )
            else:
                os.rename(staging, final)
                _fsync_directory(self.root)
            if fault_hook is not None:
                fault_hook("after_publish")
        except BaseException:
            # A fault before rename deliberately leaves staging behind. The
            # next startup treats it as an orphan; it is never visible as latest.
            raise
        return SnapshotInfo(
            path=final,
            version_stamp=project.version_stamp,
            snapshot_hash=manifest["snapshot_hash"],
            project_hash=project.content_hash(),
            mode=mode,
            manifest=manifest,
        )

    def load_project(self, snapshot: SnapshotInfo) -> Project:
        data = _read_json(snapshot.path / "project.json")
        project = project_from_dict(data)
        require(
            project.content_hash() == snapshot.project_hash,
            "E_SNAPSHOT_STALE",
            "snapshot project content hash mismatch",
            version_stamp=snapshot.version_stamp,
        )
        return project

    def restore_render_cache(
        self, snapshot: SnapshotInfo, engine: RenderEngine, mode: str
    ) -> list[str]:
        require(
            mode == snapshot.mode,
            "E_SNAPSHOT_STALE",
            "snapshot render mode differs from requested mode",
            snapshot_mode=snapshot.mode,
            mode=mode,
        )
        require(
            engine.project.content_hash() == snapshot.project_hash,
            "E_SNAPSHOT_STALE",
            "snapshot belongs to a different project content hash",
        )
        manifest = snapshot.manifest
        require(
            manifest.get("geometry_hashes") == engine.geometry.geometry_hashes(),
            "E_SNAPSHOT_STALE",
            "snapshot geometry hashes do not match the current project",
        )
        keys = manifest.get("render_cache_keys")
        hashes = manifest.get("render_hashes")
        shape_files = manifest.get("shape_files")
        require(
            isinstance(keys, dict) and isinstance(hashes, dict) and isinstance(shape_files, dict),
            "E_SNAPSHOT_STALE",
            "snapshot render cache inventory is malformed",
        )
        images: dict[str, bytes] = {}
        for shape_id, relative in shape_files.items():
            if shape_id not in keys:
                continue
            images[shape_id] = _safe_relative(snapshot.path, relative).read_bytes()
        return restore_render_cache(engine, mode, keys, images, hashes)


def result_config_dict(result: RenderResult) -> dict[str, int]:
    height, width, _ = result.image.shape
    return {"width": int(width), "height": int(height)}
