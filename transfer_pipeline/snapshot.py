"""Atomic, generation-based snapshots with a committed stable task prefix."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, pretty_json, sha256_json, sha256_text


@dataclass
class SnapshotData:
    generation: str
    path: Path
    manifest: dict[str, Any]
    shared_forward: dict[str, Any]
    adapter_stage: dict[str, Any]
    evaluation_stage: dict[str, Any]
    task_cache: list[dict[str, Any]]
    failures: list[dict[str, Any]]

    @property
    def global_version(self) -> int:
        return int(self.manifest.get("global_version", 0))

    @property
    def completed_tasks(self) -> list[dict[str, Any]]:
        value = self.manifest.get("completed_tasks", [])
        return value if isinstance(value, list) else []

    def completed_by_name(self) -> dict[str, dict[str, Any]]:
        return {
            str(entry.get("task_name")): entry
            for entry in self.completed_tasks
            if isinstance(entry, dict) and entry.get("task_name") is not None
        }

    def cache_by_name(self) -> dict[str, dict[str, Any]]:
        return {
            str(task.get("task_name")): task
            for task in self.task_cache
            if isinstance(task, dict) and task.get("task_name") is not None
        }


class SnapshotManager:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.snapshots_dir = self.workspace / "snapshots"
        self.current_file = self.workspace / "CURRENT.json"

    def ensure(self) -> None:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def clear(self) -> None:
        shutil.rmtree(self.snapshots_dir, ignore_errors=True)
        try:
            self.current_file.unlink()
        except FileNotFoundError:
            pass

    def commit(
        self,
        *,
        version: int,
        config_hash: str,
        base_fingerprint: str,
        base_layers: list[dict[str, Any]],
        input_state_hash: str,
        shared_forward: dict[str, Any],
        completed_tasks: list[dict[str, Any]],
        input_fingerprints: dict[str, Any] | None = None,
    ) -> SnapshotData:
        self.ensure()
        clean_tasks = []
        for task in completed_tasks:
            clean = dict(task)
            clean.pop("source", None)
            clean_tasks.append(clean)
        adapter_tasks = []
        evaluation_tasks = []
        failures = []
        for task in clean_tasks:
            adapter_entry = {
                key: value
                for key, value in task.items()
                if key not in {"validation"}
            }
            adapter_tasks.append(adapter_entry)
            evaluation_entry = {
                "task_name": task.get("task_name"),
                "sort_key": task.get("sort_key"),
                "target_layer": task.get("target_layer"),
                "cache_key": task.get("cache_key"),
                "status": task.get("status"),
            }
            if "validation" in task:
                evaluation_entry["validation"] = task["validation"]
            if "failure" in task:
                evaluation_entry["failure"] = task["failure"]
                failures.append(task["failure"])
            evaluation_tasks.append(evaluation_entry)

        adapter_stage = {"schema_version": 1, "tasks": adapter_tasks}
        evaluation_stage = {"schema_version": 1, "tasks": evaluation_tasks}
        shared_stage_version = "sf-" + sha256_json(shared_forward)[:24]
        adapter_stage_version = "ad-" + sha256_json(adapter_stage)[:24]
        evaluation_stage_version = "ev-" + sha256_json(evaluation_stage)[:24]
        completed_manifest = [
            {
                "task_name": task.get("task_name"),
                "sort_key": task.get("sort_key"),
                "cache_key": task.get("cache_key"),
                "status": task.get("status"),
            }
            for task in clean_tasks
        ]
        manifest = {
            "schema_version": 1,
            "complete": True,
            "global_version": int(version),
            "config_hash": config_hash,
            "base": {
                "fingerprint": base_fingerprint,
                "layers": base_layers,
            },
            "input_state_hash": input_state_hash,
            "input_fingerprints": input_fingerprints or {},
            "completed_tasks": completed_manifest,
            "stages": {
                "shared_forward": shared_stage_version,
                "adapters": adapter_stage_version,
                "evaluations": evaluation_stage_version,
            },
        }
        manifest_hash = sha256_json(manifest)
        generation = f"g{version:06d}-{manifest_hash[:10]}"
        temporary = Path(tempfile.mkdtemp(prefix=".tmp-gen-", dir=self.snapshots_dir))
        try:
            atomic_write_json(temporary / "manifest.json", manifest)
            atomic_write_json(temporary / "shared_forward.json", shared_forward)
            atomic_write_json(temporary / "adapters.json", adapter_stage)
            atomic_write_json(temporary / "evaluations.json", evaluation_stage)
            atomic_write_json(temporary / "task_cache.json", clean_tasks)
            atomic_write_json(temporary / "failures.json", failures)
            final_path = self.snapshots_dir / generation
            if final_path.exists():
                shutil.rmtree(temporary)
            else:
                os.rename(temporary, final_path)
            atomic_write_json(
                self.current_file,
                {
                    "generation": generation,
                    "manifest_sha256": manifest_hash,
                },
            )
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        snapshot = self.load_generation(generation)
        if snapshot is None:
            raise RuntimeError(f"failed to read just-committed snapshot {generation}")
        self._prune(max_generations=64)
        return snapshot

    def load_current(self) -> SnapshotData | None:
        pointer: dict[str, Any] | None = None
        if self.current_file.is_file():
            try:
                with self.current_file.open("r", encoding="utf-8") as handle:
                    pointer_value = json.load(handle)
                if isinstance(pointer_value, dict):
                    pointer = pointer_value
            except (OSError, json.JSONDecodeError):
                pointer = None
        if pointer and pointer.get("generation"):
            snapshot = self.load_generation(str(pointer["generation"]))
            if snapshot is not None:
                expected_hash = pointer.get("manifest_sha256")
                if expected_hash is None or expected_hash == sha256_json(snapshot.manifest):
                    return snapshot
        if not self.snapshots_dir.is_dir():
            return None
        for path in sorted(
            (entry for entry in self.snapshots_dir.iterdir() if entry.is_dir() and not entry.name.startswith(".")),
            key=lambda item: item.name,
            reverse=True,
        ):
            snapshot = self.load_generation(path.name)
            if snapshot is not None:
                return snapshot
        return None

    def load_generation(self, generation: str) -> SnapshotData | None:
        path = self.snapshots_dir / generation
        manifest_path = path / "manifest.json"
        required = [
            path / "shared_forward.json",
            path / "adapters.json",
            path / "evaluations.json",
            path / "task_cache.json",
            path / "failures.json",
        ]
        if not manifest_path.is_file() or any(not item.is_file() for item in required):
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or manifest.get("complete") is not True:
                return None
            shared_forward = json.loads(required[0].read_text(encoding="utf-8"))
            adapter_stage = json.loads(required[1].read_text(encoding="utf-8"))
            evaluation_stage = json.loads(required[2].read_text(encoding="utf-8"))
            task_cache = json.loads(required[3].read_text(encoding="utf-8"))
            failures = json.loads(required[4].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(task_cache, list) or not isinstance(failures, list):
            return None
        stages = manifest.get("stages")
        if not isinstance(stages, dict):
            return None
        expected_stages = {
            "shared_forward": "sf-" + sha256_json(shared_forward)[:24],
            "adapters": "ad-" + sha256_json(adapter_stage)[:24],
            "evaluations": "ev-" + sha256_json(evaluation_stage)[:24],
        }
        if any(stages.get(key) != value for key, value in expected_stages.items()):
            return None
        if len(manifest.get("completed_tasks", [])) != len(task_cache):
            return None
        return SnapshotData(
            generation=generation,
            path=path,
            manifest=manifest,
            shared_forward=shared_forward,
            adapter_stage=adapter_stage,
            evaluation_stage=evaluation_stage,
            task_cache=task_cache,
            failures=failures,
        )

    def _prune(self, max_generations: int) -> None:
        if not self.snapshots_dir.is_dir():
            return
        generations = sorted(
            (entry for entry in self.snapshots_dir.iterdir() if entry.is_dir() and not entry.name.startswith(".")),
            key=lambda item: item.name,
            reverse=True,
        )
        for path in generations[max_generations:]:
            shutil.rmtree(path, ignore_errors=True)


def generation_manifest_summary(snapshot: SnapshotData | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "generation": snapshot.generation,
        "global_version": snapshot.global_version,
        "config_hash": snapshot.manifest.get("config_hash"),
        "input_state_hash": snapshot.manifest.get("input_state_hash"),
        "base": snapshot.manifest.get("base"),
        "completed_tasks": [entry.get("task_name") for entry in snapshot.completed_tasks],
        "stages": snapshot.manifest.get("stages"),
    }


def stable_prefix(tasks: list[dict[str, Any]], ready: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    for task in tasks:
        value = ready.get(str(task.get("task_name")))
        if value is None:
            break
        prefix.append(value)
    return prefix


def fingerprint_of_stage(value: dict[str, Any]) -> str:
    return sha256_text(pretty_json(value))
