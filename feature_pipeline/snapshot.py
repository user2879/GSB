"""Content-addressed, atomically committed stage snapshots."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .util import (
    fingerprint,
    parallel_sorted_map,
    pretty_json,
    read_json,
    sha256_bytes,
)

STAGE_ORDER = (
    "cross_encoding",
    "standardization",
    "dimensionality_reduction",
    "metrics",
)


@dataclass
class StageCommit:
    stage: str
    version: str
    manifest: dict[str, Any]
    replaced: bool
    events: list[dict[str, Any]] = field(default_factory=list)
    max_observed_workers: int = 0

    @property
    def entries(self) -> dict[str, dict[str, Any]]:
        return {entry["sort_key"]: entry for entry in self.manifest["items"]}


class SnapshotStore:
    def __init__(self, cache_root: str | Path, temp_root: str | Path):
        self.cache_root = Path(cache_root)
        self.temp_root = Path(temp_root)
        self.stages_root = self.cache_root / "stages"
        self.manifest_path = self.cache_root / "manifest.json"
        self.stages_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        (self.temp_root / "stages").mkdir(parents=True, exist_ok=True)

    def load_global_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            payload = read_json(self.manifest_path)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def write_global_manifest(self, manifest: dict[str, Any]) -> None:
        from .util import atomic_write_json

        atomic_write_json(self.manifest_path, manifest)

    def _stage_root(self, stage: str) -> Path:
        return self.stages_root / stage

    def _current_link(self, stage: str) -> Path:
        return self._stage_root(stage) / "current"

    def _version_dir(self, stage: str, version: str) -> Path:
        return self._stage_root(stage) / "versions" / f"v-{version}"

    def _valid_version_dir(self, stage: str, directory: Path) -> bool:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = read_json(manifest_path)
            return manifest.get("stage") == stage and manifest.get("version") == directory.name[2:]
        except Exception:
            return False

    def current_version_dir(self, stage: str) -> Path | None:
        link = self._current_link(stage)
        if not link.exists():
            return None
        try:
            resolved = link.resolve(strict=True)
        except OSError:
            return None
        if resolved.is_dir() and self._valid_version_dir(stage, resolved):
            return resolved
        return None

    def stage_manifest(self, stage: str) -> dict[str, Any] | None:
        directory = self.current_version_dir(stage)
        if directory is None:
            return None
        try:
            payload = read_json(directory / "manifest.json")
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def artifact_path(self, stage: str, entry: dict[str, Any]) -> Path | None:
        directory = self.current_version_dir(stage)
        if directory is None:
            return None
        path = directory / "artifacts" / str(entry.get("file", ""))
        return path if path.is_file() else None

    def load_entry(self, stage: str, sort_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = self.stage_manifest(stage)
        if manifest is None:
            raise KeyError(f"no committed snapshot for stage {stage}")
        entries = {entry["sort_key"]: entry for entry in manifest.get("items", [])}
        if sort_key not in entries:
            raise KeyError(f"missing artifact {stage}/{sort_key}")
        path = self.artifact_path(stage, entries[sort_key])
        if path is None:
            raise KeyError(f"missing artifact file {stage}/{sort_key}")
        return entries[sort_key], read_json(path)

    def repair(self) -> list[dict[str, Any]]:
        """Remove abandoned staging directories and repair broken current links."""

        actions: list[dict[str, Any]] = []
        for child in sorted(self.temp_root.glob(".stage-build-*")):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                actions.append(
                    {
                        "action": "remove_abandoned_staging",
                        "path": child.name,
                        "sort_key": f"recovery/{child.name}",
                    }
                )
        for child in sorted(self.stages_root.glob(".*.current-*")):
            if child.is_symlink() or child.exists():
                try:
                    child.unlink()
                except OSError:
                    pass
        global_manifest = self.load_global_manifest() or {}
        global_stages = global_manifest.get("stages", {}) if isinstance(global_manifest, dict) else {}
        for stage in STAGE_ORDER:
            stage_root = self._stage_root(stage)
            if not stage_root.exists():
                continue
            if self.current_version_dir(stage) is not None:
                continue
            candidate: Path | None = None
            expected = global_stages.get(stage, {}).get("version") if isinstance(global_stages, dict) else None
            if expected:
                directory = self._version_dir(stage, str(expected))
                if directory.is_dir() and self._valid_version_dir(stage, directory):
                    candidate = directory
            if candidate is None:
                candidates = [
                    path
                    for path in (stage_root / "versions").glob("v-*")
                    if path.is_dir() and self._valid_version_dir(stage, path)
                ]
                if candidates:
                    candidate = max(
                        candidates,
                        key=lambda path: (path.stat().st_mtime_ns, path.name),
                    )
            if candidate is not None:
                self._set_current(stage, candidate.name[2:])
                actions.append(
                    {
                        "action": "restore_current_pointer",
                        "stage": stage,
                        "version": candidate.name[2:],
                        "sort_key": f"recovery/{stage}",
                    }
                )
        return sorted(actions, key=lambda item: (item["sort_key"], item["action"]))

    def _set_current(self, stage: str, version: str) -> None:
        stage_root = self._stage_root(stage)
        (stage_root / "versions").mkdir(parents=True, exist_ok=True)
        temporary = stage_root / f".{stage}.current-{uuid.uuid4().hex}"
        target = Path("versions") / f"v-{version}"
        os.symlink(target, temporary, target_is_directory=True)
        os.replace(temporary, stage_root / "current")

    def _copy_or_link(self, source: Path, destination: Path) -> None:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    def commit_stage(
        self,
        stage: str,
        *,
        descriptors: list[dict[str, Any]],
        metadata: dict[str, Any],
        max_workers: int,
    ) -> StageCommit:
        """Commit a complete new stage snapshot, reusing valid artifacts by content key."""

        if stage not in STAGE_ORDER:
            raise ValueError(f"unknown stage {stage}")
        old_manifest = self.stage_manifest(stage)
        old_directory = self.current_version_dir(stage)
        old_entries = {
            entry["sort_key"]: entry for entry in (old_manifest or {}).get("items", [])
        }
        ordered = sorted(descriptors, key=lambda item: item["sort_key"])
        tasks: list[tuple[str, Callable[[], dict[str, Any]]]] = []
        events: list[dict[str, Any]] = []
        reuse_sources: dict[str, Path] = {}
        for descriptor in ordered:
            sort_key = descriptor["sort_key"]
            old_entry = old_entries.get(sort_key)
            reusable = False
            reason = "cold_miss"
            source_path: Path | None = None
            if old_entry is not None:
                if old_entry.get("cache_key") == descriptor["cache_key"] and old_directory is not None:
                    candidate = old_directory / "artifacts" / str(old_entry.get("file", ""))
                    if candidate.is_file():
                        if sha256_bytes(candidate.read_bytes()) == old_entry.get("file_sha256"):
                            reusable = True
                            source_path = candidate
                        else:
                            reason = "artifact_corrupt"
                    else:
                        reason = "artifact_missing"
                else:
                    reason = "cache_key_changed"
            if reusable and source_path is not None:
                reuse_sources[sort_key] = source_path
            else:
                builder = descriptor.get("builder")
                if builder is None:
                    raise ValueError(f"descriptor {sort_key} has no builder")
                tasks.append((sort_key, builder))
                events.append(
                    {
                        "stage": stage,
                        "sort_key": sort_key,
                        "item": descriptor.get("item", sort_key),
                        "cache_key": descriptor["cache_key"],
                        "action": "recomputed_invalidated" if old_entry is not None else "recomputed_miss",
                        "reason": reason,
                    }
                )

        built, observed = parallel_sorted_map(tasks, max_workers=max_workers, sort_key=lambda item: item)
        artifacts: dict[str, dict[str, Any]] = {}
        item_records: list[dict[str, Any]] = []
        for descriptor in ordered:
            sort_key = descriptor["sort_key"]
            if sort_key in reuse_sources:
                old_entry = old_entries[sort_key]
                item_records.append(
                    {
                        "sort_key": sort_key,
                        "item": descriptor.get("item", sort_key),
                        "cache_key": descriptor["cache_key"],
                        "content_fingerprint": old_entry["content_fingerprint"],
                        "file": old_entry["file"],
                        "file_sha256": old_entry["file_sha256"],
                    }
                )
                events.append(
                    {
                        "stage": stage,
                        "sort_key": sort_key,
                        "item": descriptor.get("item", sort_key),
                        "cache_key": descriptor["cache_key"],
                        "action": "reused",
                        "reason": "cache_key_and_content_valid",
                        "content_fingerprint": old_entry["content_fingerprint"],
                    }
                )
                continue
            payload = built[sort_key]
            content = payload.get("content")
            if content is None:
                raise ValueError(f"builder {sort_key} did not return content")
            content_fp = fingerprint(content)
            artifact = {
                "schema_version": 1,
                "stage": stage,
                "sort_key": sort_key,
                "item": descriptor.get("item", sort_key),
                "cache_key": descriptor["cache_key"],
                "content_fingerprint": content_fp,
                "content": content,
                "metadata": payload.get("metadata", {}),
            }
            data = pretty_json(artifact).encode("utf-8")
            file_name = f"{fingerprint({'cache_key': descriptor['cache_key'], 'content': content_fp})[:32]}.json"
            artifacts[sort_key] = {
                "file": file_name,
                "bytes": data,
                "content_fingerprint": content_fp,
                "file_sha256": sha256_bytes(data),
            }
            item_records.append(
                {
                    "sort_key": sort_key,
                    "item": descriptor.get("item", sort_key),
                    "cache_key": descriptor["cache_key"],
                    "content_fingerprint": content_fp,
                    "file": file_name,
                    "file_sha256": sha256_bytes(data),
                }
            )

        current_keys = {descriptor["sort_key"] for descriptor in ordered}
        for sort_key in sorted(set(old_entries) - current_keys):
            entry = old_entries[sort_key]
            events.append(
                {
                    "stage": stage,
                    "sort_key": sort_key,
                    "item": entry.get("item", sort_key),
                    "cache_key": entry.get("cache_key", ""),
                    "action": "removed",
                    "reason": "no_longer_selected",
                    "content_fingerprint": entry.get("content_fingerprint", ""),
                }
            )

        version = fingerprint(
            {
                "stage": stage,
                "metadata": metadata,
                "items": [
                    {
                        "sort_key": item["sort_key"],
                        "cache_key": item["cache_key"],
                        "content_fingerprint": item["content_fingerprint"],
                    }
                    for item in item_records
                ],
            }
        )
        manifest = {
            "schema_version": 1,
            "stage": stage,
            "version": version,
            "metadata": metadata,
            "items": item_records,
        }
        replaced = old_manifest is None or old_manifest.get("version") != version
        if replaced:
            self._publish_snapshot(
                stage=stage,
                version=version,
                manifest=manifest,
                artifacts=artifacts,
                reuse_sources=reuse_sources,
            )

        events = sorted(
            events,
            key=lambda event: (STAGE_ORDER.index(event["stage"]), event["sort_key"], event["action"]),
        )
        return StageCommit(
            stage=stage,
            version=version,
            manifest=manifest,
            replaced=replaced,
            events=events,
            max_observed_workers=observed,
        )

    def _publish_snapshot(
        self,
        *,
        stage: str,
        version: str,
        manifest: dict[str, Any],
        artifacts: dict[str, dict[str, Any]],
        reuse_sources: dict[str, Path],
    ) -> None:
        staging = self.temp_root / f".stage-build-{stage}-{uuid.uuid4().hex}"
        artifacts_dir = staging / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=False)
        try:
            for sort_key, source in sorted(reuse_sources.items()):
                file_name = next(
                    item["file"] for item in manifest["items"] if item["sort_key"] == sort_key
                )
                self._copy_or_link(source, artifacts_dir / file_name)
            for sort_key, artifact in sorted(artifacts.items()):
                (artifacts_dir / artifact["file"]).write_bytes(artifact["bytes"])
            (staging / "manifest.json").write_text(pretty_json(manifest), encoding="utf-8")
            version_dir = self._version_dir(stage, version)
            version_dir.parent.mkdir(parents=True, exist_ok=True)
            if version_dir.exists():
                shutil.rmtree(staging, ignore_errors=True)
            else:
                try:
                    os.replace(staging, version_dir)
                except OSError:
                    shutil.copytree(staging, version_dir)
                    shutil.rmtree(staging, ignore_errors=True)
            self._set_current(stage, version)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
