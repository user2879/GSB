from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import (
    IMPLEMENTATION_VERSION,
    SCHEMA_VERSION,
    atomic_write_text,
    canonical_json_text,
    fingerprint,
    fingerprint_file,
)
from .config import dump_config_yaml, load_config_yaml, validate_config
from .errors import SnapshotStaleError


STAGE_NAMES = ("features", "training", "inference", "summary")


@dataclass
class SnapshotData:
    version_dir: Path
    manifest: dict[str, Any]
    sample_lines: list[str]
    raw_config: dict[str, Any]
    applied_batches: list[str]
    feature_row_cache: dict[str, dict[str, Any]]
    inference_row_cache: dict[str, dict[str, Any]]
    stages: dict[str, dict[str, Any]]
    ledger: list[dict[str, Any]]
    pipeline_complete: bool
    pending: list[str]
    plan_fingerprint: str | None


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.pointer = self.root / "CURRENT"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def has_current(self) -> bool:
        return self.pointer.is_file()

    def current_pointer(self) -> str | None:
        if not self.has_current():
            return None
        value = self.pointer.read_text(encoding="utf-8").strip()
        return value or None

    def current_version_dir(self) -> Path | None:
        pointer = self.current_pointer()
        if pointer is None:
            return None
        path = self.versions_dir / pointer
        if not path.is_dir():
            raise SnapshotStaleError(
                f"CURRENT points to missing snapshot {pointer}", actual=pointer
            )
        return path

    def load_manifest(self) -> dict[str, Any] | None:
        version_dir = self.current_version_dir()
        if version_dir is None:
            return None
        manifest_path = version_dir / "manifest.json"
        if not manifest_path.is_file():
            raise SnapshotStaleError(
                f"snapshot {version_dir.name} has no manifest",
                actual=version_dir.name,
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotStaleError(
                f"snapshot manifest cannot be read: {exc}", actual=version_dir.name
            ) from exc
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise SnapshotStaleError(
                "snapshot schema version is incompatible",
                expected=str(SCHEMA_VERSION),
                actual=str(manifest.get("schema_version")),
            )
        if manifest.get("engine_version") != IMPLEMENTATION_VERSION:
            raise SnapshotStaleError(
                "snapshot engine version is incompatible",
                expected=IMPLEMENTATION_VERSION,
                actual=str(manifest.get("engine_version")),
            )
        return manifest

    def validate_resume(
        self,
        *,
        expected_version: int | None = None,
        expected_config_hash: str | None = None,
    ) -> dict[str, Any] | None:
        manifest = self.load_manifest()
        if manifest is None:
            return None
        if expected_version is not None and manifest.get("global_version") != expected_version:
            raise SnapshotStaleError(
                "snapshot global version differs from the requested version",
                expected=str(expected_version),
                actual=str(manifest.get("global_version")),
            )
        if (
            expected_config_hash is not None
            and manifest.get("config_hash") != expected_config_hash
        ):
            raise SnapshotStaleError(
                "snapshot configuration hash differs from the supplied configuration",
                expected=expected_config_hash,
                actual=str(manifest.get("config_hash")),
            )
        return manifest

    def load_current(self) -> SnapshotData | None:
        version_dir = self.current_version_dir()
        if version_dir is None:
            return None
        manifest = self.load_manifest()
        if manifest is None:
            return None
        samples_path = version_dir / "samples.jsonl"
        config_path = version_dir / "config.yaml"
        state_path = version_dir / "state.json"
        if not samples_path.is_file() or not config_path.is_file() or not state_path.is_file():
            raise SnapshotStaleError(
                "snapshot state files are incomplete", actual=version_dir.name
            )
        try:
            sample_text = samples_path.read_text(encoding="utf-8")
            sample_lines = sample_text.splitlines()
            raw_config = load_config_yaml(str(config_path))
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SnapshotStaleError(
                f"snapshot state cannot be loaded: {exc}", actual=version_dir.name
            ) from exc
        if validate_config(raw_config).config_hash != manifest.get("config_hash"):
            raise SnapshotStaleError(
                "snapshot configuration hash does not match config.yaml",
                expected=str(manifest.get("config_hash")),
                actual=validate_config(raw_config).config_hash,
            )
        if fingerprint("raw-sample-lines:v1", sample_lines) != manifest.get(
            "raw_data_fingerprint"
        ):
            raise SnapshotStaleError(
                "snapshot sample content does not match raw_data_fingerprint",
                expected=str(manifest.get("raw_data_fingerprint")),
                actual=fingerprint("raw-sample-lines:v1", sample_lines),
            )
        stages: dict[str, dict[str, Any]] = {}
        for stage in STAGE_NAMES:
            path = version_dir / "stages" / f"{stage}.json"
            if path.is_file():
                stages[stage] = json.loads(path.read_text(encoding="utf-8"))
        ledger_path = version_dir / "ledger.json"
        ledger = (
            json.loads(ledger_path.read_text(encoding="utf-8"))
            if ledger_path.is_file()
            else []
        )
        return SnapshotData(
            version_dir=version_dir,
            manifest=manifest,
            sample_lines=sample_lines,
            raw_config=raw_config,
            applied_batches=list(state.get("applied_batches", [])),
            feature_row_cache=dict(state.get("feature_row_cache", {})),
            inference_row_cache=dict(state.get("inference_row_cache", {})),
            stages=stages,
            ledger=ledger,
            pipeline_complete=bool(state.get("pipeline_complete", True)),
            pending=list(state.get("pending", [])),
            plan_fingerprint=state.get("plan_fingerprint"),
        )

    def _version_numbers(self) -> list[int]:
        numbers: list[int] = []
        for path in self.versions_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("v"):
                continue
            try:
                numbers.append(int(path.name[1:].split("-", 1)[0]))
            except ValueError:
                continue
        return numbers

    def commit(
        self,
        *,
        sample_lines: list[str],
        raw_config: dict[str, Any],
        applied_batches: list[str],
        feature_row_cache: dict[str, dict[str, Any]],
        inference_row_cache: dict[str, dict[str, Any]],
        stages: dict[str, dict[str, Any]],
        ledger: list[dict[str, Any]],
        pipeline_complete: bool,
        pending: list[str],
        config_hash: str,
        data_fingerprint: str,
        raw_data_fingerprint: str,
        sample_count: int,
        plan_fingerprint: str | None,
    ) -> dict[str, Any]:
        sample_text = "\n".join(sample_lines) + ("\n" if sample_lines else "")
        config_text = dump_config_yaml(raw_config)
        state: dict[str, Any] = {
            "applied_batches": list(applied_batches),
            "feature_row_cache": feature_row_cache,
            "inference_row_cache": inference_row_cache,
            "pipeline_complete": pipeline_complete,
            "pending": list(pending),
            "plan_fingerprint": plan_fingerprint,
        }
        stage_versions = {
            stage: stages[stage].get("cache_key") if stage in stages else None
            for stage in STAGE_NAMES
        }
        stable_ledger = [
            {
                "stage": entry.get("stage"),
                "cache_key": entry.get("cache_key"),
                "sample_ids": entry.get("sample_ids", []),
                "upstreams": entry.get("upstreams", []),
            }
            for entry in ledger
        ]
        content_identity = fingerprint(
            "snapshot-content:v1",
            {
                "samples": sample_text,
                "config": raw_config,
                "state": state,
                "stage_versions": stage_versions,
                "stage_content": {
                    stage: stages[stage].get("content_fingerprint")
                    for stage in STAGE_NAMES
                    if stage in stages
                },
                "ledger": stable_ledger,
            },
        )

        current_manifest = None
        if self.has_current():
            current_manifest = self.load_manifest()
        if (
            current_manifest is not None
            and current_manifest.get("content_identity") == content_identity
        ):
            return current_manifest

        next_version = max(self._version_numbers(), default=0) + 1
        version_id = f"v{next_version:06d}-{content_identity[:12]}"
        final_dir = self.versions_dir / version_id
        staging_dir = Path(
            tempfile.mkdtemp(prefix=".staging-", dir=str(self.versions_dir))
        )
        try:
            (staging_dir / "stages").mkdir(parents=True, exist_ok=True)
            atomic_write_text(staging_dir / "samples.jsonl", sample_text)
            atomic_write_text(staging_dir / "config.yaml", config_text)
            atomic_write_text(staging_dir / "state.json", canonical_json_text(state))
            atomic_write_text(
                staging_dir / "ledger.json", canonical_json_text(ledger)
            )
            for stage, artifact in stages.items():
                if stage not in STAGE_NAMES:
                    continue
                atomic_write_text(
                    staging_dir / "stages" / f"{stage}.json",
                    canonical_json_text(artifact),
                )
            manifest: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "engine_version": IMPLEMENTATION_VERSION,
                "global_version": next_version,
                "version_id": version_id,
                "content_identity": content_identity,
                "config_hash": config_hash,
                "data_fingerprint": data_fingerprint,
                "raw_data_fingerprint": raw_data_fingerprint,
                "sample_count": sample_count,
                "stage_versions": stage_versions,
                "stage_files": {
                    stage: fingerprint_file(staging_dir / "stages" / f"{stage}.json")
                    for stage in stages
                    if stage in STAGE_NAMES
                },
                "pipeline_complete": pipeline_complete,
                "pending": list(pending),
                "applied_batches": list(applied_batches),
            }
            atomic_write_text(
                staging_dir / "manifest.json", canonical_json_text(manifest)
            )
            os.replace(staging_dir, final_dir)
            atomic_write_text(self.pointer, version_id + "\n")
            return manifest
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def list_versions(self) -> list[dict[str, Any]]:
        versions: list[dict[str, Any]] = []
        for path in sorted(self.versions_dir.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or not path.name.startswith("v"):
                continue
            manifest_path = path / "manifest.json"
            if manifest_path.is_file():
                versions.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        return versions

    def resolve_version(self, version: int | str) -> Path:
        if isinstance(version, int):
            for path in self.versions_dir.iterdir():
                if not path.is_dir() or not path.name.startswith("v"):
                    continue
                manifest_path = path / "manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("global_version") == version:
                    return path
            raise FileNotFoundError(f"snapshot version {version} was not found")
        path = self.versions_dir / version
        if not path.is_dir():
            raise FileNotFoundError(f"snapshot {version} was not found")
        return path

    def restore(self, version: int | str) -> dict[str, Any]:
        path = self.resolve_version(version)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise SnapshotStaleError(
                "selected snapshot has no manifest", actual=path.name
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise SnapshotStaleError(
                "selected snapshot schema is incompatible",
                expected=str(SCHEMA_VERSION),
                actual=str(manifest.get("schema_version")),
            )
        if manifest.get("engine_version") != IMPLEMENTATION_VERSION:
            raise SnapshotStaleError(
                "selected snapshot engine is incompatible",
                expected=IMPLEMENTATION_VERSION,
                actual=str(manifest.get("engine_version")),
            )
        atomic_write_text(self.pointer, path.name + "\n")
        return manifest
