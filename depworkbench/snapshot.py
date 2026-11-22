from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .io import read_json, write_json_atomic


SNAPSHOT_SCHEMA_VERSION = 1
STAGES = ("load", "parse", "arcs", "trees")


class SnapshotStore:
    """Version-directory snapshots with an atomically replaced CURRENT pointer."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.versions_dir = self.root / "versions"

    def _current_name(self) -> str | None:
        pointer = self.root / "CURRENT"
        if not pointer.exists():
            return None
        try:
            name = pointer.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValidationError("E_SNAPSHOT_STALE", f"cannot read snapshot pointer: {exc}") from exc
        if not name:
            return None
        return name

    def latest_manifest(self) -> dict[str, Any] | None:
        name = self._current_name()
        if name is None:
            return None
        manifest_path = self.versions_dir / name / "manifest.json"
        if not manifest_path.exists():
            raise ValidationError(
                "E_SNAPSHOT_STALE",
                f"snapshot pointer {name!r} does not resolve to a manifest",
            )
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValidationError("E_SNAPSHOT_STALE", "snapshot manifest is not an object")
        return manifest

    def load_latest(self) -> dict[str, Any] | None:
        name = self._current_name()
        if name is None:
            return None
        version_dir = self.versions_dir / name
        manifest = self.latest_manifest()
        if manifest is None:
            return None
        loaded: dict[str, Any] = {"manifest": manifest, "snapshot_name": name}
        for stage in STAGES:
            path = version_dir / f"{stage}.json"
            if path.exists():
                loaded[stage] = read_json(path)
            else:
                loaded[stage] = None
        return loaded

    def commit(
        self,
        *,
        stages: dict[str, Any],
        model_hash: str,
        config_hash: str,
        corpus_hash: str,
        mode: str,
        completed_sentences: list[str],
        sentence_cursors: dict[str, Any],
        extra_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous = self.latest_manifest()
        previous_version = int(previous.get("global_version", 0)) if previous else 0
        version = previous_version + 1
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        version_name = f"v{version:06d}"
        while (self.versions_dir / version_name).exists():
            version += 1
            version_name = f"v{version:06d}"

        tmp_parent = self.root / "tmp"
        tmp_parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="snapshot-", dir=tmp_parent))
        try:
            for stage in STAGES:
                write_json_atomic(tmp_dir / f"{stage}.json", stages.get(stage, {}))
            stage_versions = {stage: version for stage in STAGES}
            manifest: dict[str, Any] = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "global_version": version,
                "snapshot_name": version_name,
                "model_hash": model_hash,
                "config_hash": config_hash,
                "corpus_hash": corpus_hash,
                "mode": mode,
                "completed_sentences": sorted(completed_sentences),
                "sentence_cursors": dict(sorted(sentence_cursors.items())),
                "stage_versions": stage_versions,
            }
            if extra_manifest:
                manifest.update(extra_manifest)
            write_json_atomic(tmp_dir / "manifest.json", manifest)
            final_dir = self.versions_dir / version_name
            os.replace(tmp_dir, final_dir)
            pointer_tmp = self.root / f".CURRENT.{os.getpid()}.tmp"
            with pointer_tmp.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(version_name + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pointer_tmp, self.root / "CURRENT")
            return manifest
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def reinitialize(self) -> None:
        pointer = self.root / "CURRENT"
        if pointer.exists():
            try:
                pointer.unlink()
            except OSError:
                pass
