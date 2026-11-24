"""Versioned, atomically promoted snapshots for stage artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .util import atomic_write_json, sha256_json


@dataclass
class LoadedSnapshot:
    version: int
    config_hash: str
    state: Dict[str, Any]
    artifacts: Dict[str, Dict[str, Dict[str, Any]]]
    path: Path


class SnapshotStore:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.root = self.output_dir / "state"
        self.versions_dir = self.root / "versions"
        self.current_pointer = self.root / "CURRENT"
        self.current_version = 0

    def load(self) -> Optional[LoadedSnapshot]:
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        candidates = []
        if self.current_pointer.is_file():
            try:
                pointer = self.current_pointer.read_text(encoding="utf-8").strip()
                pointer_path = (self.root / pointer).resolve()
                if pointer_path.is_dir():
                    candidates.append(pointer_path)
            except OSError:
                pass
        if not candidates:
            versions = sorted(
                (path for path in self.versions_dir.iterdir() if path.is_dir() and path.name.startswith("v")),
                reverse=True,
            )
            candidates.extend(versions[:1])
        for candidate in candidates:
            loaded = self._load_version_directory(candidate)
            if loaded is not None:
                self.current_version = loaded.version
                return loaded
        return None

    def _load_version_directory(self, directory: Path) -> Optional[LoadedSnapshot]:
        state_path = directory / "state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if state.get("schema_version") != 1 or not isinstance(state.get("pages"), dict):
            return None
        artifacts: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for page, metadata in state["pages"].items():
            if not isinstance(metadata, dict):
                continue
            page_artifacts: Dict[str, Dict[str, Any]] = {}
            for stage, stage_meta in metadata.get("artifacts", {}).items():
                artifact_path = directory / "artifacts" / page / f"{stage}.json"
                try:
                    data = json.loads(artifact_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                key = stage_meta.get("key")
                version = stage_meta.get("version")
                if not isinstance(key, str) or version != sha256_json(data):
                    continue
                page_artifacts[stage] = {"key": key, "version": version, "data": data}
            if page_artifacts:
                artifacts[page] = page_artifacts
        version = state.get("version")
        if not isinstance(version, int) or version < 1:
            try:
                version = int(directory.name.lstrip("v"))
            except ValueError:
                return None
        config_hash = state.get("config_hash")
        if not isinstance(config_hash, str):
            config_hash = ""
        return LoadedSnapshot(
            version=version,
            config_hash=config_hash,
            state=state,
            artifacts=artifacts,
            path=directory,
        )

    def commit(self, state: Dict[str, Any], artifacts: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        version = self.current_version + 1
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
        try:
            payload = json.loads(json.dumps(state, ensure_ascii=False, sort_keys=True))
            payload["schema_version"] = 1
            payload["version"] = version
            payload_pages = payload.setdefault("pages", {})
            for page, page_artifacts in artifacts.items():
                page_meta = payload_pages.setdefault(page, {})
                artifact_meta: Dict[str, Any] = {}
                for stage, entry in page_artifacts.items():
                    data = entry["data"]
                    version_hash = sha256_json(data)
                    artifact_meta[stage] = {"key": entry["key"], "version": version_hash}
                    artifact_path = staging / "artifacts" / page / f"{stage}.json"
                    artifact_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(artifact_path, data)
                page_meta["artifacts"] = artifact_meta
            atomic_write_json(staging / "state.json", payload)

            final_directory = self.versions_dir / f"v{version:08d}"
            while final_directory.exists():
                version += 1
                payload["version"] = version
                final_directory = self.versions_dir / f"v{version:08d}"
            atomic_write_json(staging / "state.json", payload)
            os.replace(staging, final_directory)
            pointer_value = final_directory.relative_to(self.root).as_posix()
            pointer_temp = self.current_pointer.with_name(f".CURRENT.tmp-{os.getpid()}")
            with pointer_temp.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(pointer_value + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pointer_temp, self.current_pointer)
            self.current_version = version
            self._cleanup_old_versions(keep=3)
            return payload
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def _cleanup_old_versions(self, keep: int) -> None:
        versions = sorted(
            (path for path in self.versions_dir.iterdir() if path.is_dir() and path.name.startswith("v")),
            key=lambda path: path.name,
            reverse=True,
        )
        for old in versions[keep:]:
            shutil.rmtree(old, ignore_errors=True)
