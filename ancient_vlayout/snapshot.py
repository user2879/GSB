"""Generation-directory snapshots with an atomically replaced current pointer."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import E_SNAPSHOT_INVALID, E_SNAPSHOT_RECOVERED
from .util import (
    atomic_write_json,
    pretty_json,
    read_json,
    sha256_bytes,
    sha256_json,
)


@dataclass
class SnapshotHandle:
    path: Path
    manifest: dict[str, Any]
    pointer_recovered: bool = False
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def generation(self) -> int:
        return int(self.manifest.get("generation", 0))

    def read_stage_bytes(self, relative_path: str) -> bytes:
        target = (self.path / relative_path).resolve()
        if self.path.resolve() not in target.parents and target != self.path.resolve():
            raise ValueError("stage artifact path escapes snapshot")
        return target.read_bytes()

    def read_stage_json(self, relative_path: str) -> Any:
        return read_json((self.path / relative_path).resolve())


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshots_dir = self.root / "snapshots"
        self.pointer_path = self.root / "current.json"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_staging()

    def _cleanup_staging(self) -> None:
        for path in self.root.glob(".staging-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    def load_current(self) -> SnapshotHandle | None:
        diagnostics: list[dict[str, Any]] = []
        if self.pointer_path.is_file():
            try:
                pointer = read_json(self.pointer_path)
                relative = pointer["manifest"]
                if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
                    raise ValueError("invalid manifest pointer")
                manifest_path = (self.root / relative).resolve()
                if self.root.resolve() not in manifest_path.parents:
                    raise ValueError("pointer escapes state root")
                manifest = read_json(manifest_path)
                snapshot_dir = manifest_path.parent
                if not (snapshot_dir / "results.json").is_file():
                    raise ValueError("snapshot results missing")
                return SnapshotHandle(path=snapshot_dir, manifest=manifest, diagnostics=diagnostics)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                diagnostics.append(
                    {
                        "code": E_SNAPSHOT_INVALID,
                        "message": f"current pointer invalid: {exc}",
                    }
                )

        candidates: list[SnapshotHandle] = []
        for manifest_path in self.snapshots_dir.glob("*/manifest.json"):
            try:
                manifest = read_json(manifest_path)
                if not (manifest_path.parent / "results.json").is_file():
                    raise ValueError("results missing")
                candidates.append(SnapshotHandle(path=manifest_path.parent, manifest=manifest))
            except (OSError, ValueError):
                continue
        if not candidates:
            return None
        candidates.sort(key=lambda handle: (int(handle.manifest.get("generation", 0)), handle.path.name))
        recovered = candidates[-1]
        recovered.pointer_recovered = True
        recovered.diagnostics = diagnostics + [
            {
                "code": E_SNAPSHOT_RECOVERED,
                "message": "recovered latest complete snapshot generation",
                "generation": recovered.manifest.get("generation"),
            }
        ]
        self._write_pointer(recovered)
        return recovered

    def next_generation(self, current: SnapshotHandle | None) -> int:
        return (current.generation + 1) if current is not None else 1

    def _write_pointer(self, handle: SnapshotHandle) -> None:
        relative = (handle.path / "manifest.json").relative_to(self.root).as_posix()
        atomic_write_json(
            self.pointer_path,
            {
                "schema_version": 1,
                "generation": handle.generation,
                "global_version": handle.manifest.get("global_version"),
                "manifest": relative,
            },
        )

    def commit(
        self,
        manifest_fields: dict[str, Any],
        files: dict[str, bytes | str | Any],
        current: SnapshotHandle | None,
    ) -> SnapshotHandle:
        generation = self.next_generation(current)
        normalized_files: dict[str, bytes] = {}
        for relative, value in files.items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("snapshot file path must be relative and contained")
            if isinstance(value, bytes):
                normalized_files[relative] = value
            elif isinstance(value, str):
                normalized_files[relative] = value.encode("utf-8")
            else:
                normalized_files[relative] = pretty_json(value).encode("utf-8")

        file_hashes = {relative: sha256_bytes(data) for relative, data in sorted(normalized_files.items())}
        version_material = {
            "generation": generation,
            "environment_hash": manifest_fields["environment_hash"],
            "files": file_hashes,
        }
        content_hash = sha256_json(version_material)
        global_version = f"v{generation:06d}-{content_hash[:12]}"
        manifest = dict(manifest_fields)
        manifest.update(
            {
                "schema_version": 1,
                "generation": generation,
                "global_version": global_version,
                "files": file_hashes,
            }
        )
        normalized_files["manifest.json"] = pretty_json(manifest).encode("utf-8")

        staging = self.root / f".staging-{os.getpid()}-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            for relative, data in normalized_files.items():
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as handle:
                    handle.write(data)
            final_dir = self.snapshots_dir / f"gen-{generation:06d}-{content_hash[:10]}"
            if final_dir.exists():
                raise FileExistsError(f"snapshot generation already exists: {final_dir.name}")
            os.replace(staging, final_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        committed = SnapshotHandle(path=final_dir, manifest=manifest)
        self._write_pointer(committed)
        return committed


def stage_version_map(stage_files: dict[str, dict[str, str]]) -> dict[str, str]:
    """Hash key/path pairs per stage for manifest artifact versions."""

    return {
        stage: sha256_json(sorted(mapping.items()))
        for stage, mapping in sorted(stage_files.items())
    }
