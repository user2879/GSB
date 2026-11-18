"""Pool snapshot persistence, version detection and whole-state replacement.

A snapshot is an immutable JSON document describing the full workbench state:
parameter version, connection collection and states, quota usage, queued tasks,
scheduling traces and outcome statistics.  On restore, a parameter-version
mismatch triggers a *full replacement* of parameters, connections and quota
state rather than an incremental merge.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BusinessError, ErrorCode

SNAPSHOT_FILE_RE = re.compile(r"snapshot-v(\d+)-(\d+)\.json$")
_SNAPSHOT_TOP_FIELDS = (
    "snapshot_id", "param_version", "sample_data_version", "pool_config",
    "pool_state", "queued_tasks", "tasks", "traces", "outcome_counts",
)
_CONFIG_FIELDS = (
    "max_connections", "workload_quotas", "hold_timeout_ms",
    "acquire_timeout_ms", "param_version",
)


@dataclass
class VersionDiff:
    old_version: int
    new_version: int
    changed_params: list[dict[str, Any]]
    affected_queued_tasks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "changed_params": self.changed_params,
            "affected_queued_tasks": self.affected_queued_tasks,
        }


class SnapshotStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def list_snapshots(self) -> list[Path]:
        return sorted(
            (path for path in self.directory.glob("snapshot-*.json")
             if SNAPSHOT_FILE_RE.search(path.name)),
            key=lambda path: SNAPSHOT_FILE_RE.search(path.name).groups(),
        )

    def _next_index(self) -> int:
        indices = [
            int(SNAPSHOT_FILE_RE.search(path.name).group(2))
            for path in self.list_snapshots()
        ]
        return (max(indices) + 1) if indices else 1

    def save(self, document: dict[str, Any]) -> Path:
        self._validate_document(document)
        with self._lock:
            index = self._next_index()
            name = f"snapshot-v{document['param_version']}-{index:03d}.json"
            path = self.directory / name
            path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return path

    @staticmethod
    def _validate_document(document: dict[str, Any]) -> None:
        missing = [field for field in _SNAPSHOT_TOP_FIELDS if field not in document]
        if missing:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT,
                "snapshot document is missing required field(s): " + ", ".join(missing),
                [f"snapshot.{field}" for field in missing],
            )
        if "param_version" not in document.get("pool_config", {}):
            raise BusinessError(
                ErrorCode.E_SNAPSHOT,
                "snapshot pool_config is missing param_version",
                ["snapshot.pool_config.param_version"],
            )

    def load(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise BusinessError(
                ErrorCode.E_SNAPSHOT, f"snapshot file not found: {path}", [str(path)]
            )
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT,
                f"invalid snapshot JSON: {exc.msg} at line {exc.lineno}",
                [f"{path}:{exc.lineno}"],
            ) from exc
        if not isinstance(document, dict):
            raise BusinessError(
                ErrorCode.E_SNAPSHOT, "snapshot root must be an object", [str(path)]
            )
        self._validate_document(document)
        return document


def diff_pool_config(
    snapshot_config: dict[str, Any], current_config: dict[str, Any]
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for field in _CONFIG_FIELDS:
        old_value = snapshot_config.get(field)
        new_value = current_config.get(field)
        if old_value != new_value:
            changes.append({"param": field, "snapshot_value": old_value,
                            "current_value": new_value})
    return changes


def compute_version_diff(
    snapshot_document: dict[str, Any],
    current_pool_config: dict[str, Any],
    queued_tasks: list[str],
) -> VersionDiff:
    snapshot_config = snapshot_document["pool_config"]
    snapshot_version = int(snapshot_config["param_version"])
    current_version = int(current_pool_config["param_version"])
    return VersionDiff(
        old_version=snapshot_version,
        new_version=current_version,
        changed_params=diff_pool_config(snapshot_config, current_pool_config),
        affected_queued_tasks=list(queued_tasks),
    )
