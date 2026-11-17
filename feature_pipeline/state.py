"""Immutable stage snapshots and crash recovery."""

from __future__ import annotations

import copy
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    sha256_bytes,
    sha256_value,
)

SNAPSHOT_FORMAT_VERSION = 1
GENERATION_PATTERN = re.compile(r"^g(\d{6})$")
STAGE_ORDER = ("events", "features", "samples")


@dataclass
class RuntimeState:
    generation: int = 0
    manifest: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    definitions: dict[str, Any] = field(default_factory=dict)
    features: list[dict[str, Any]] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    cache_ledger: list[dict[str, Any]] = field(default_factory=list)
    backfills: list[dict[str, Any]] = field(default_factory=list)

    @property
    def completed_stages(self) -> list[str]:
        return list(self.manifest.get("completed_stages", []))

    @property
    def event_cursor(self) -> int:
        value = self.manifest.get("event_cursor", -1)
        return int(value)

    @property
    def global_version(self) -> int:
        return int(self.manifest.get("global_version", self.generation))

    def clone(self) -> "RuntimeState":
        return RuntimeState(
            generation=self.generation,
            manifest=copy.deepcopy(self.manifest),
            events=copy.deepcopy(self.events),
            definitions=copy.deepcopy(self.definitions),
            features=copy.deepcopy(self.features),
            samples=copy.deepcopy(self.samples),
            cache_ledger=copy.deepcopy(self.cache_ledger),
            backfills=copy.deepcopy(self.backfills),
        )


class SnapshotError(RuntimeError):
    pass


class SnapshotStore:
    def __init__(self, runtime_dir: Path):
        self.runtime_dir = runtime_dir
        self.snapshot_root = runtime_dir / "snapshots"
        self.temp_root = runtime_dir / ".tmp"
        self.pointer_path = runtime_dir / "CURRENT"

    def reset(self) -> None:
        if self.runtime_dir.exists():
            shutil.rmtree(self.runtime_dir)
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def recover(self) -> tuple[RuntimeState, dict[str, Any]]:
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        candidates = self._candidate_generations()
        pointer = self._read_pointer()
        selected: Path | None = None
        if pointer is not None and pointer in candidates:
            selected = self.snapshot_root / pointer
            try:
                state = self._load_generation(selected)
            except SnapshotError:
                selected = None
            else:
                return state, self._recovery_info(state, "CURRENT")
        for candidate in reversed(candidates):
            path = self.snapshot_root / candidate
            try:
                state = self._load_generation(path)
            except SnapshotError:
                continue
            selected = path
            return state, self._recovery_info(state, "highest-valid-generation")
        empty = RuntimeState()
        return empty, {
            "resumed": False,
            "source": "empty",
            "generation": 0,
            "completed_stages": [],
            "event_cursor": -1,
            "definition_hash": None,
        }

    def commit(
        self,
        state: RuntimeState,
        stage: str,
        *,
        definition_hash: str,
        event_cursor: int,
        completed_stages: list[str],
    ) -> RuntimeState:
        if stage not in STAGE_ORDER:
            raise ValueError(f"unknown snapshot stage: {stage}")
        generation = self._next_generation()
        target = self.snapshot_root / f"g{generation:06d}"
        temporary = self.temp_root / f"commit-{uuid.uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            artifacts = self._write_artifacts(temporary, state)
            stage_versions = self._stage_versions(state, completed_stages)
            manifest = {
                "format_version": SNAPSHOT_FORMAT_VERSION,
                "generation": generation,
                "global_version": generation,
                "definition_hash": definition_hash,
                "event_cursor": event_cursor,
                "completed_stages": list(completed_stages),
                "stage_versions": stage_versions,
                "artifact_versions": {
                    name: sha256_bytes((temporary / name).read_bytes())
                    for name in sorted(artifacts)
                },
            }
            atomic_write_json(temporary / "manifest.json", manifest)
            os.replace(temporary, target)
            self._fsync_directory(self.snapshot_root)
            atomic_write_bytes(self.pointer_path, (target.name + "\n").encode("ascii"))
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
        committed = state.clone()
        committed.generation = generation
        committed.manifest = manifest
        return committed

    def _write_artifacts(self, directory: Path, state: RuntimeState) -> list[str]:
        artifacts = {
            "events.json": state.events,
            "definitions.json": state.definitions,
            "features.json": state.features,
            "samples.json": state.samples,
            "cache_ledger.json": state.cache_ledger,
            "backfills.json": state.backfills,
        }
        for name, value in artifacts.items():
            atomic_write_json(directory / name, value)
        return sorted(artifacts)

    def _stage_versions(self, state: RuntimeState, completed_stages: list[str]) -> dict[str, str]:
        versions: dict[str, str] = {}
        if "events" in completed_stages:
            versions["events"] = sha256_value(
                {
                    "events": state.events,
                    "definitions": state.definitions,
                    "backfills": state.backfills,
                }
            )
        if "features" in completed_stages:
            versions["features"] = sha256_value(
                {
                    "features": state.features,
                    "cache_ledger": state.cache_ledger,
                }
            )
        if "samples" in completed_stages:
            versions["samples"] = sha256_value(state.samples)
        return versions

    def _load_generation(self, directory: Path) -> RuntimeState:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise SnapshotError("manifest missing")
        try:
            manifest = read_json(manifest_path)
        except Exception as exc:  # noqa: BLE001 - corrupt snapshots must be skipped
            raise SnapshotError(f"invalid manifest: {exc}") from exc
        if manifest.get("format_version") != SNAPSHOT_FORMAT_VERSION:
            raise SnapshotError("unsupported snapshot format")
        completed = manifest.get("completed_stages", [])
        if "events" not in completed:
            raise SnapshotError("event stage not complete")
        expected = manifest.get("artifact_versions", {})
        for stage in completed:
            if stage not in STAGE_ORDER:
                raise SnapshotError(f"unknown completed stage: {stage}")
        required = ["events.json", "definitions.json", "backfills.json"]
        if "features" in completed:
            required.extend(["features.json", "cache_ledger.json"])
        if "samples" in completed:
            required.append("samples.json")
        for name in required:
            path = directory / name
            if not path.is_file():
                raise SnapshotError(f"required artifact missing: {name}")
            if name in expected and sha256_bytes(path.read_bytes()) != expected[name]:
                raise SnapshotError(f"artifact hash mismatch: {name}")
        artifacts: dict[str, Any] = {}
        for name in (
            "events.json",
            "definitions.json",
            "features.json",
            "samples.json",
            "cache_ledger.json",
            "backfills.json",
        ):
            path = directory / name
            if path.is_file():
                try:
                    artifacts[name] = read_json(path)
                except Exception as exc:  # noqa: BLE001
                    raise SnapshotError(f"invalid artifact {name}: {exc}") from exc
        return RuntimeState(
            generation=int(manifest["generation"]),
            manifest=manifest,
            events=artifacts.get("events.json", []),
            definitions=artifacts.get("definitions.json", {}),
            features=artifacts.get("features.json", []),
            samples=artifacts.get("samples.json", []),
            cache_ledger=artifacts.get("cache_ledger.json", []),
            backfills=artifacts.get("backfills.json", []),
        )

    def _candidate_generations(self) -> list[str]:
        if not self.snapshot_root.exists():
            return []
        return sorted(
            path.name
            for path in self.snapshot_root.iterdir()
            if path.is_dir() and GENERATION_PATTERN.match(path.name)
        )

    def _read_pointer(self) -> str | None:
        try:
            value = self.pointer_path.read_text(encoding="ascii").strip()
        except OSError:
            return None
        return value if GENERATION_PATTERN.match(value) else None

    def _next_generation(self) -> int:
        generations = [int(name[1:]) for name in self._candidate_generations()]
        return (max(generations) if generations else 0) + 1

    @staticmethod
    def _recovery_info(state: RuntimeState, source: str) -> dict[str, Any]:
        return {
            "resumed": state.generation > 0,
            "source": source,
            "generation": state.generation,
            "completed_stages": state.completed_stages,
            "event_cursor": state.event_cursor,
            "definition_hash": state.manifest.get("definition_hash"),
        }

    @staticmethod
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


def build_manifest_preview(
    generation: int, definition_hash: str, event_cursor: int, completed_stages: list[str]
) -> dict[str, Any]:
    return {
        "global_version": generation,
        "definition_hash": definition_hash,
        "event_cursor": event_cursor,
        "completed_stages": completed_stages,
    }


def canonical_cache_ledger(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "feature": record["feature"],
                "entity_type": record["entity_type"],
                "entity_id": record["entity_id"],
                "event_id": record["event_id"],
                "event_timestamp": record["event_timestamp"],
                "value": record["value"],
                "empty": record["empty"],
                "definition_hash": record["definition_hash"],
                "cache_key": record["cache_key"],
            }
            for record in features
        ],
        key=lambda item: (
            item["feature"],
            item["event_timestamp"],
            item["event_id"],
            item["entity_type"],
            item["entity_id"],
        ),
    )
