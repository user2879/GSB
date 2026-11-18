"""Immutable versioned snapshots with an atomically replaced CURRENT pointer."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigMismatchError, SnapshotError
from .loader import STAGE_ORDER
from .util import atomic_write_json, atomic_write_text, dumps_report, load_json, sha256_bytes


SNAPSHOT_SCHEMA_VERSION = 1
CURRENT_FILE = "CURRENT"
VERSIONS_DIR = "versions"


@dataclass
class SnapshotState:
    root: Path
    version: int
    manifest: dict[str, Any]
    artifacts: dict[str, dict[str, Any]]

    @property
    def stages_done(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("stages_done", ()))

    def artifact(self, stage: str) -> dict[str, Any] | None:
        return self.artifacts.get(stage)

    def records(self, stage: str) -> list[dict[str, Any]]:
        artifact = self.artifact(stage)
        return [] if artifact is None else list(artifact.get("records", []))


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_dir = self.root / VERSIONS_DIR
        self.current_path = self.root / CURRENT_FILE

    def reset(self) -> None:
        if self.versions_dir.exists():
            shutil.rmtree(self.versions_dir)
        self.current_path.unlink(missing_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def read_current(self) -> SnapshotState | None:
        if not self.current_path.exists():
            return None
        try:
            pointer = load_json(self.current_path)
            version = pointer["version"]
            if not isinstance(version, int) or version < 1:
                raise ValueError("invalid version")
        except Exception as exc:  # noqa: BLE001
            raise SnapshotError(f"invalid CURRENT pointer: {exc}") from exc

        version_dir = self.versions_dir / f"v{version:06d}"
        manifest_path = version_dir / "manifest.json"
        if not manifest_path.is_file():
            raise SnapshotError(f"snapshot version directory is incomplete: {version_dir}")
        try:
            manifest = load_json(manifest_path)
        except Exception as exc:  # noqa: BLE001
            raise SnapshotError(f"cannot read snapshot manifest: {exc}") from exc
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotError("unsupported snapshot schema_version")
        if manifest.get("global_version") != version:
            raise SnapshotError("CURRENT pointer and manifest version differ")

        artifacts: dict[str, dict[str, Any]] = {}
        artifacts_meta = manifest.get("artifacts", {})
        if not isinstance(artifacts_meta, dict):
            raise SnapshotError("manifest artifacts must be an object")
        for stage, meta in artifacts_meta.items():
            if not isinstance(meta, dict):
                raise SnapshotError(f"invalid artifact metadata for stage {stage}")
            file_name = meta.get("file")
            expected_hash = meta.get("sha256")
            if not isinstance(file_name, str) or not isinstance(expected_hash, str):
                raise SnapshotError(f"incomplete artifact metadata for stage {stage}")
            artifact_path = version_dir / file_name
            if not artifact_path.is_file():
                raise SnapshotError(f"snapshot artifact is missing: {artifact_path}")
            raw_bytes = artifact_path.read_bytes()
            actual_hash = sha256_bytes(raw_bytes)
            if actual_hash != expected_hash:
                raise SnapshotError(f"snapshot artifact hash mismatch for stage {stage}")
            try:
                artifact = json.loads(raw_bytes.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise SnapshotError(f"invalid artifact JSON for stage {stage}: {exc}") from exc
            artifacts[stage] = artifact

        return SnapshotState(self.root, version, manifest, artifacts)

    def assert_compatible(
        self,
        state: SnapshotState,
        *,
        config_hash: str,
        model_hash: str,
        input_hash: str,
    ) -> None:
        manifest = state.manifest
        mismatches: list[str] = []
        if manifest.get("config_hash") != config_hash:
            mismatches.append("config_hash")
        if manifest.get("model_hash") != model_hash:
            mismatches.append("model_hash")
        if manifest.get("input_hash") != input_hash:
            mismatches.append("input_hash")
        if mismatches:
            raise ConfigMismatchError(
                "snapshot is incompatible with the current run: " + ", ".join(mismatches),
                details={
                    "mismatched": mismatches,
                    "snapshot_version": state.version,
                },
            )

    def commit(
        self,
        *,
        stage: str,
        records: list[dict[str, Any]],
        stage_done: bool,
        config_hash: str,
        model_hash: str,
        input_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotState:
        if stage not in STAGE_ORDER:
            raise SnapshotError(f"unknown stage {stage}")
        previous = self.read_current()
        if previous is not None:
            if previous.manifest.get("config_hash") != config_hash:
                raise ConfigMismatchError("cannot commit a snapshot with a different config")
            if previous.manifest.get("model_hash") != model_hash:
                raise ConfigMismatchError("cannot commit a snapshot with a different model")
            if previous.manifest.get("input_hash") != input_hash:
                raise ConfigMismatchError("cannot commit a snapshot with different inputs")
            next_version = previous.version + 1
        else:
            next_version = 1

        version_dir = self.versions_dir / f"v{next_version:06d}"
        if version_dir.exists():
            raise SnapshotError(f"snapshot version already exists: {version_dir}")
        temp_dir = self.versions_dir / f".tmp-v{next_version:06d}-{os.getpid()}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True)

        artifacts: dict[str, dict[str, Any]] = {}
        stage_versions: dict[str, int] = {}
        stages_done: list[str] = []
        stage_completed: dict[str, list[str]] = {}
        if previous is not None:
            for previous_stage, artifact in previous.artifacts.items():
                previous_meta = previous.manifest["artifacts"][previous_stage]
                shutil.copy2(
                    self.versions_dir
                    / f"v{previous.version:06d}"
                    / previous_meta["file"],
                    temp_dir / previous_meta["file"],
                )
                artifacts[previous_stage] = artifact
                stage_versions[previous_stage] = previous_meta["version"]
                stage_completed[previous_stage] = list(
                    previous.manifest.get("stage_completed_sentences", {}).get(
                        previous_stage, []
                    )
                )
            stages_done = list(previous.manifest.get("stages_done", []))
            for previous_stage in previous.artifacts:
                old_artifact = previous.artifacts[previous_stage]
                if old_artifact.get("stage_done") and previous_stage not in stages_done:
                    stages_done.append(previous_stage)

        sorted_records = sorted(records, key=lambda item: item["sentence_id"])
        completed_ids = sorted({record["sentence_id"] for record in sorted_records})
        artifact = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "stage": stage,
            "stage_done": bool(stage_done),
            "records": sorted_records,
            "completed_sentence_ids": completed_ids,
        }
        if metadata:
            artifact.update(metadata)
        artifact_text = dumps_report(artifact)
        artifact_file = f"{stage}.json"
        atomic_write_text(temp_dir / artifact_file, artifact_text)
        artifacts[stage] = artifact
        stage_versions[stage] = next_version
        stage_completed[stage] = completed_ids
        if stage_done and stage not in stages_done:
            stages_done.append(stage)
        if not stage_done and stage in stages_done:
            stages_done.remove(stage)
        stages_done.sort(key=STAGE_ORDER.index)

        artifact_meta: dict[str, dict[str, Any]] = {}
        for artifact_stage in sorted(artifacts, key=STAGE_ORDER.index):
            file_name = f"{artifact_stage}.json"
            artifact_meta[artifact_stage] = {
                "file": file_name,
                "sha256": sha256_bytes((temp_dir / file_name).read_bytes()),
                "version": stage_versions[artifact_stage],
                "stage_done": bool(artifacts[artifact_stage].get("stage_done")),
            }
        completed_sentences = sorted(
            {
                sentence_id
                for sentence_ids in stage_completed.values()
                for sentence_id in sentence_ids
            }
        )
        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "global_version": next_version,
            "previous_version": None if previous is None else previous.version,
            "config_hash": config_hash,
            "model_hash": model_hash,
            "input_hash": input_hash,
            "stages_done": stages_done,
            "completed_sentences": completed_sentences,
            "stage_completed_sentences": {
                key: sorted(value) for key, value in sorted(stage_completed.items())
            },
            "artifacts": artifact_meta,
        }
        atomic_write_text(temp_dir / "manifest.json", dumps_report(manifest))

        # The version directory becomes visible only after every artifact and the
        # manifest are complete. The CURRENT pointer is separately atomic.
        os.replace(temp_dir, version_dir)
        atomic_write_json(self.current_path, {"version": next_version})
        return SnapshotState(self.root, next_version, manifest, artifacts)


def snapshot_summary(state: SnapshotState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "version": state.version,
        "stages_done": list(state.stages_done),
        "completed_sentences": list(state.manifest.get("completed_sentences", [])),
        "stage_versions": {
            stage: meta["version"]
            for stage, meta in sorted(
                state.manifest.get("artifacts", {}).items(), key=lambda item: STAGE_ORDER.index(item[0])
            )
        },
    }
