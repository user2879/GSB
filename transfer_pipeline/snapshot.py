"""Atomic stage snapshots and conservative recovery checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .canonical import SNAPSHOT_SCHEMA_VERSION, atomic_write_json
from .errors import SnapshotError


class SnapshotStore:
    """One atomically replaced manifest containing all stage artifacts."""

    def __init__(self, state_directory: str | Path) -> None:
        self.state_directory = Path(state_directory)
        self.path = self.state_directory / "snapshot.json"

    def ensure_writable(self) -> None:
        try:
            self.state_directory.mkdir(parents=True, exist_ok=True)
            probe = self.state_directory / ".write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise SnapshotError(
                f"state directory is not writable: {self.state_directory}"
            ) from exc

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"cannot read snapshot {self.path}: {exc}") from exc
        if not isinstance(value, dict):
            raise SnapshotError("snapshot root must be an object")
        if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotError(
                "unsupported snapshot schema version "
                f"{value.get('schema_version')!r}"
            )
        for key in (
            "global_version",
            "config_hash",
            "run_id",
            "representation_cache",
            "adapter_cache",
            "evaluation_cache",
        ):
            if key not in value:
                raise SnapshotError(f"snapshot is missing required field {key!r}")
        if not isinstance(value["global_version"], int):
            raise SnapshotError("snapshot global_version must be an integer")
        for key in ("representation_cache", "adapter_cache", "evaluation_cache"):
            if not isinstance(value[key], Mapping):
                raise SnapshotError(f"snapshot {key} must be an object")
        return value

    def commit(self, snapshot: Mapping[str, Any]) -> None:
        self.state_directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, snapshot)
