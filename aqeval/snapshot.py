"""Four-stage snapshot commits with stale detection and resumable manifests."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .util import atomic_write_json, pretty_json_bytes, read_json, sha256_json


SNAPSHOT_SCHEMA = "aqeval.snapshot.v1"
STAGES = ("parse", "features", "scores", "conversion")


class SnapshotStore:
    def __init__(
        self,
        root: str | Path,
        *,
        job_version: int,
        config_hash: str,
        level_map_hash: str,
        active_version: int,
        global_run: int,
    ) -> None:
        self.root = Path(root) / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.current_path = self.root / "current.json"
        self.job_version = job_version
        self.config_hash = config_hash
        self.level_map_hash = level_map_hash
        self.active_version = active_version
        self.global_run = global_run
        self.global_version = global_run + 1
        self.version_fingerprint = sha256_json(
            {
                "active_version": active_version,
                "level_map_hash": level_map_hash,
                "job_version": job_version,
            }
        )
        self.run_id = f"run-{global_run + 1:06d}"
        self.stage_root = self.root / "runs" / self.run_id
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.stale = False
        self.stale_reason = ""
        self.manifest = self._load_manifest()
        if self.manifest is not None and not self._matches_version():
            self.stale = True
            self.stale_reason = "global version, configuration hash or version fingerprint changed"
            self._clear_manifest()
            self.manifest = None
        if self.manifest is not None:
            previous_run_id = self.manifest.get("run_id")
            if isinstance(previous_run_id, str) and previous_run_id != self.run_id:
                self.run_id = previous_run_id
                self.stage_root = self.root / "runs" / self.run_id
                self.stage_root.mkdir(parents=True, exist_ok=True)
        if self.manifest is None:
            self.manifest = {
                "schema": SNAPSHOT_SCHEMA,
                "run_id": self.run_id,
                "global_version": global_run + 1,
                "job_version": job_version,
                "config_hash": config_hash,
                "level_map_hash": level_map_hash,
                "active_version": active_version,
                "version_fingerprint": self.version_fingerprint,
                "content_fingerprint": "",
                "stages": {},
                "complete": False,
            }
            self._write_manifest()

    def _load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            value = read_json(self.manifest_path)
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None

    def _matches_version(self) -> bool:
        if self.manifest is None:
            return False
        return (
            self.manifest.get("job_version") == self.job_version
            and self.manifest.get("config_hash") == self.config_hash
            and self.manifest.get("level_map_hash") == self.level_map_hash
            and self.manifest.get("active_version") == self.active_version
            and self.manifest.get("global_version") in (self.global_run, self.global_run + 1)
        )

    def _clear_manifest(self) -> None:
        if self.manifest_path.exists():
            self.manifest_path.unlink()
        if self.current_path.exists():
            self.current_path.unlink()
        runs = self.root / "runs"
        if runs.exists():
            shutil.rmtree(runs)
        self.stage_root.mkdir(parents=True, exist_ok=True)

    def _write_manifest(self) -> None:
        assert self.manifest is not None
        atomic_write_json(self.manifest_path, self.manifest)

    def set_content_fingerprint(self, value: str) -> None:
        assert self.manifest is not None
        self.manifest["content_fingerprint"] = value
        self._write_manifest()

    def stage_entry(self, stage: str) -> dict[str, Any] | None:
        assert self.manifest is not None
        value = self.manifest.get("stages", {}).get(stage)
        return value if isinstance(value, dict) else None

    def can_resume(self, stage: str, input_fingerprint: str) -> bool:
        entry = self.stage_entry(stage)
        if not entry or entry.get("input_fingerprint") != input_fingerprint:
            return False
        path = self.stage_root / str(entry.get("file", ""))
        return bool(entry.get("complete")) and path.exists()

    def load_stage(self, stage: str) -> dict[str, Any]:
        entry = self.stage_entry(stage)
        if entry is None:
            raise KeyError(f"stage {stage} is not present")
        path = self.stage_root / str(entry["file"])
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"stage {stage} artifact is not an object")
        return value

    def commit_stage(self, stage: str, payload: dict[str, Any], input_fingerprint: str) -> dict[str, Any]:
        assert self.manifest is not None
        digest = sha256_json(payload)
        stage_root = self.root / "stages"
        stage_root.mkdir(parents=True, exist_ok=True)
        filename = f"{self.run_id}-{stage}-{digest[:16]}.json"
        final_path = stage_root / filename
        tmp_path = self.stage_root / f".{stage}.tmp"
        tmp_path.write_bytes(pretty_json_bytes(payload))
        tmp_path.replace(final_path)
        entry = {
            "input_fingerprint": input_fingerprint,
            "artifact_version": digest,
            "file": f"../../stages/{filename}",
            "complete": True,
        }
        self.manifest.setdefault("stages", {})[stage] = entry
        self.manifest["complete"] = False
        self._write_manifest()
        return entry

    def complete(self, report_path: str, output_digest: str) -> None:
        assert self.manifest is not None
        self.manifest["complete"] = True
        self.manifest["global_version"] = self.global_version
        self.manifest["report_path"] = report_path
        self.manifest["output_digest"] = output_digest
        self._write_manifest()
        atomic_write_json(
            self.current_path,
            {
                "schema": "aqeval.snapshot-current.v1",
                "run_id": self.run_id,
                "manifest": "manifest.json",
                "config_hash": self.config_hash,
                "version_fingerprint": self.version_fingerprint,
                "complete": True,
            },
        )
        self._prune_history()

    def _prune_history(self) -> None:
        assert self.manifest is not None
        referenced: set[Path] = set()
        for entry in self.manifest.get("stages", {}).values():
            if isinstance(entry, dict) and entry.get("file"):
                referenced.add((self.stage_root / str(entry["file"])).resolve())
        stages_root = self.root / "stages"
        if stages_root.exists():
            for path in stages_root.glob("*.json"):
                if path.resolve() not in referenced:
                    path.unlink()
        runs_root = self.root / "runs"
        if runs_root.exists():
            for path in runs_root.iterdir():
                if path.is_dir() and path.name != self.run_id:
                    shutil.rmtree(path, ignore_errors=True)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.manifest or {})
