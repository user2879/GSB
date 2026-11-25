"""Content-addressed, whole-snapshot persistence."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import load_structured, write_json


STAGE_FILES = [
    "features",
    "statistics",
    "scores",
    "decisions",
    "summaries",
    "diagnostics",
    "fusion",
    "pixel_map",
    "audit",
    "run_manifest",
]


@dataclass
class PreviousSnapshot:
    sequence: int
    directory: Path
    stages: dict[str, Any]

    def stage(self, name: str, default: Any = None) -> Any:
        return self.stages.get(name, default)


class SnapshotStore:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir.resolve()
        self.versions_dir = self.state_dir / "versions"
        self.current_link = self.state_dir / "current"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def load_current(self) -> PreviousSnapshot | None:
        if not self.current_link.is_symlink() and not self.current_link.exists():
            return None
        directory = self.current_link.resolve()
        if not directory.is_dir():
            return None
        stages: dict[str, Any] = {}
        for name in STAGE_FILES:
            path = directory / f"{name}.json"
            if path.exists():
                stages[name] = load_structured(path)
        sequence = 0
        manifest_path = directory / "run_manifest.json"
        if manifest_path.exists():
            manifest = stages.get("run_manifest") or load_structured(manifest_path)
            sequence = int(manifest.get("snapshot_sequence", 0))
        return PreviousSnapshot(sequence=sequence, directory=directory, stages=stages)

    def next_sequence(self) -> int:
        existing_sequences = []
        for child in self.versions_dir.iterdir():
            if child.is_dir() and child.name.startswith("v"):
                try:
                    existing_sequences.append(int(child.name[1:]))
                except ValueError:
                    continue
        return max(existing_sequences, default=0) + 1

    def commit(
        self,
        stages: dict[str, Any],
        binary_files: dict[str, bytes] | None = None,
    ) -> Path:
        sequence = self.next_sequence()
        version_name = f"v{sequence:06d}"
        staging = Path(
            tempfile.mkdtemp(prefix=".staging-", dir=str(self.versions_dir))
        )
        try:
            for name, value in stages.items():
                write_json(staging / f"{name}.json", value)
            for relative_name, payload in (binary_files or {}).items():
                destination = staging / relative_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            final_directory = self.versions_dir / version_name
            os.replace(staging, final_directory)
            temporary_link = self.state_dir / ".current.next"
            try:
                temporary_link.unlink()
            except FileNotFoundError:
                pass
            os.symlink(os.path.relpath(final_directory, self.state_dir), temporary_link)
            os.replace(temporary_link, self.current_link)
            return final_directory
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def publish_file(self, snapshot_directory: Path, relative_name: str, destination: Path) -> None:
        source = snapshot_directory / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.next"
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)


def previous_by_key(previous: PreviousSnapshot | None, stage: str) -> dict[str, Any]:
    if previous is None:
        return {}
    value = previous.stage(stage, {})
    if not isinstance(value, dict):
        return {}
    by_key = value.get("by_key", {})
    return by_key if isinstance(by_key, dict) else {}


def previous_by_window(previous: PreviousSnapshot | None, stage: str) -> dict[str, Any]:
    if previous is None:
        return {}
    value = previous.stage(stage, {})
    if not isinstance(value, dict):
        return {}
    by_window = value.get("by_window", {})
    return by_window if isinstance(by_window, dict) else {}
