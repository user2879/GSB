"""Whole-snapshot staging, atomic commit, and resume classification."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hashing import sha256_json, stable_json
from .models import Issue

SNAPSHOT_SCHEMA_VERSION = 1
STAGE_NAMES = ("loading", "position_encoding", "scoring", "error_tracking")


@dataclass
class LoadedSnapshot:
    state: str
    snapshot_id: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    config_hash_matches: bool = False

    @property
    def compatible(self) -> bool:
        return self.state in {"exact_resume", "incremental"}


class SnapshotStore:
    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.snapshots_dir = self.state_dir / "snapshots"
        self.current_path = self.state_dir / "CURRENT"
        self.staging_dir = self.state_dir / ".staging"

    def _issue(self, code: str, field_name: str, message: str) -> Issue:
        return Issue(code=code, sample_id=None, field=field_name, message=message)

    def load_current(self, current_config_hash: str | None = None) -> LoadedSnapshot:
        issues: list[Issue] = []
        if not self.current_path.exists():
            return LoadedSnapshot(state="cold")
        try:
            snapshot_id = self.current_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            return LoadedSnapshot(
                state="rebuild",
                issues=[self._issue("E_SNAPSHOT_POINTER", str(self.current_path), str(exc))],
            )
        if not snapshot_id:
            return LoadedSnapshot(
                state="rebuild",
                issues=[self._issue("E_SNAPSHOT_POINTER", str(self.current_path), "pointer is empty")],
            )
        snapshot_dir = self.snapshots_dir / snapshot_id
        manifest_path = snapshot_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return LoadedSnapshot(
                state="rebuild",
                snapshot_id=snapshot_id,
                issues=[
                    self._issue("E_SNAPSHOT_MANIFEST", str(manifest_path), "manifest is missing")
                ],
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return LoadedSnapshot(
                state="rebuild",
                snapshot_id=snapshot_id,
                issues=[
                    self._issue("E_SNAPSHOT_MANIFEST", str(manifest_path), f"invalid manifest: {exc}")
                ],
            )
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            issues.append(
                self._issue(
                    "E_SNAPSHOT_VERSION",
                    "schema_version",
                    f"snapshot schema {manifest.get('schema_version')} is not compatible with {SNAPSHOT_SCHEMA_VERSION}",
                )
            )
            return LoadedSnapshot(
                state="rebuild",
                snapshot_id=snapshot_id,
                manifest=manifest,
                issues=issues,
            )
        config_hash_matches = bool(
            current_config_hash is not None and manifest.get("config_hash") == current_config_hash
        )
        stages: dict[str, dict[str, dict[str, Any]]] = {}
        for stage in STAGE_NAMES:
            stage_path = snapshot_dir / "stages" / f"{stage}.json"
            try:
                payload = json.loads(stage_path.read_text(encoding="utf-8"))
                entries = payload.get("entries", {})
                if not isinstance(entries, dict):
                    raise ValueError("entries must be an object")
                stages[stage] = entries
            except FileNotFoundError:
                stages[stage] = {}
                issues.append(
                    self._issue("E_SNAPSHOT_STAGE", str(stage_path), "stage file is missing")
                )
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                stages[stage] = {}
                issues.append(
                    self._issue("E_SNAPSHOT_STAGE", str(stage_path), f"invalid stage file: {exc}")
                )
        state = "exact_resume" if config_hash_matches else "incremental"
        return LoadedSnapshot(
            state=state,
            snapshot_id=snapshot_id,
            manifest=manifest,
            stages=stages,
            issues=issues,
            config_hash_matches=config_hash_matches,
        )

    def commit(
        self,
        *,
        config: dict[str, Any],
        config_hash: str,
        stages: dict[str, dict[str, dict[str, Any]]],
        completed_schemes: list[str],
        completed_samples: list[str],
        consumed_budget_total: int,
        last_run_consumed_points: int,
        pending_samples: list[str],
        resume_cursor: str | None,
        command_cursor: int | None,
        sample_content_hash: str,
        previous_snapshot: LoadedSnapshot | None = None,
        extra_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        previous_version = 0
        if previous_snapshot is not None:
            try:
                previous_version = int(previous_snapshot.manifest.get("snapshot_version", 0))
            except (TypeError, ValueError):
                previous_version = 0
        snapshot_version = previous_version + 1
        temp_dir = Path(tempfile.mkdtemp(prefix="snapshot-", dir=self.staging_dir))
        stage_counts = {stage: len(stages.get(stage, {})) for stage in STAGE_NAMES}
        manifest: dict[str, Any] = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_version": snapshot_version,
            "config_hash": config_hash,
            "config": config,
            "completed_schemes": sorted(set(completed_schemes)),
            "completed_samples": sorted(set(completed_samples)),
            "pending_samples": sorted(set(pending_samples)),
            "resume_cursor": resume_cursor,
            "command_cursor": command_cursor,
            "sample_content_hash": sample_content_hash,
            "consumed_budget": int(consumed_budget_total),
            "last_run_consumed_points": int(last_run_consumed_points),
            "stage_counts": stage_counts,
            "stages": list(STAGE_NAMES),
            "atomic_commit": "staging-dir-then-pointer-replace",
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        try:
            (temp_dir / "stages").mkdir(parents=True, exist_ok=True)
            for stage in STAGE_NAMES:
                payload = {
                    "stage": stage,
                    "entries": stages.get(stage, {}),
                }
                self._write_json(temp_dir / "stages" / f"{stage}.json", payload)
            self._write_json(temp_dir / "manifest.json", manifest)
            snapshot_id = f"{snapshot_version:08d}-{sha256_json(manifest)[:12]}"
            while (self.snapshots_dir / snapshot_id).exists():
                snapshot_version += 1
                manifest["snapshot_version"] = snapshot_version
                self._write_json(temp_dir / "manifest.json", manifest)
                snapshot_id = f"{snapshot_version:08d}-{sha256_json(manifest)[:12]}"
            final_dir = self.snapshots_dir / snapshot_id
            os.replace(temp_dir, final_dir)
            pointer_temp = self.current_path.with_name("CURRENT.tmp")
            pointer_temp.write_text(snapshot_id + "\n", encoding="utf-8")
            self._fsync_file(pointer_temp)
            os.replace(pointer_temp, self.current_path)
            self._prune_old_snapshots(keep=final_dir.name)
            return {
                "snapshot_id": snapshot_id,
                "snapshot_version": snapshot_version,
                "manifest": manifest,
            }
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        temp_path.write_text(stable_json(payload, pretty=True) + "\n", encoding="utf-8")
        self._fsync_file(temp_path)
        os.replace(temp_path, path)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        try:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            pass

    def _prune_old_snapshots(self, keep: str) -> None:
        try:
            for child in self.snapshots_dir.iterdir():
                if child.is_dir() and child.name != keep:
                    shutil.rmtree(child, ignore_errors=True)
            if self.staging_dir.exists():
                for child in self.staging_dir.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
        except OSError:
            pass

    def write_trace(self, payload: Any) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_dir / "run_trace.json"
        temp_path = path.with_name(path.name + ".tmp")
        temp_path.write_text(stable_json(payload, pretty=True) + "\n", encoding="utf-8")
        os.replace(temp_path, path)
        return path
