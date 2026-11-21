"""Atomic stage snapshots with versioned manifest and resume detection."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .cache import STAGES, fingerprint
from .errors import SnapshotCorruptError, SnapshotStaleError

MANIFEST_NAME = "manifest.json"


@dataclass
class Manifest:
    global_version: int
    config_hash: str
    data_fingerprint: str
    stage_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "data_fingerprint": self.data_fingerprint,
            "stage_versions": self.stage_versions,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Manifest":
        return cls(
            global_version=int(raw["global_version"]),
            config_hash=str(raw["config_hash"]),
            data_fingerprint=str(raw["data_fingerprint"]),
            stage_versions={k: str(v) for k, v in raw.get("stage_versions", {}).items()},
        )


class SnapshotStore:
    """Versioned snapshot store using whole-directory atomic replacement."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current = self.root / "current"
        self.tmp_parent = self.root / "tmp"
        self.tmp_parent.mkdir(exist_ok=True)

    def manifest_path(self) -> Path:
        return self.current / MANIFEST_NAME

    def load_manifest(self) -> Manifest | None:
        path = self.manifest_path()
        if not path.exists():
            return None
        try:
            return Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as exc:
            raise SnapshotCorruptError(f"manifest unreadable: {exc}",
                                       field="snapshot.manifest") from exc

    def verify_resume(self, config_hash: str, data_fingerprint: str) -> Manifest:
        manifest = self.load_manifest()
        if manifest is None:
            raise SnapshotStaleError("no snapshot available for resume",
                                     field="snapshot")
        if manifest.config_hash != config_hash:
            raise SnapshotStaleError(
                f"config hash mismatch: stored={manifest.config_hash[:10]} "
                f"expected={config_hash[:10]}", field="config")
        if manifest.data_fingerprint != data_fingerprint:
            raise SnapshotStaleError(
                "data fingerprint mismatch; rebuild required", field="data")
        return manifest

    def stage_dir(self, stage: str) -> Path | None:
        path = self.current / "stages" / stage
        return path if path.exists() else None

    def commit(self, stage_payloads: dict[str, dict], config_hash: str,
               data_fingerprint: str, previous: Manifest | None) -> Manifest:
        """Write all stage artifacts into a temp dir, then atomically replace."""
        temp_dir = Path(tempfile.mkdtemp(prefix="snap_", dir=self.tmp_parent))
        try:
            stages_root = temp_dir / "stages"
            stage_versions: dict[str, str] = {}
            for stage in STAGES:
                payload = stage_payloads.get(stage)
                if payload is None:
                    continue
                stage_dir = stages_root / stage
                stage_dir.mkdir(parents=True, exist_ok=True)
                body = json.dumps(payload, sort_keys=True, ensure_ascii=False)
                (stage_dir / "artifact.json").write_text(body, encoding="utf-8")
                stage_versions[stage] = fingerprint(
                    {"stage": stage, "body": json.loads(body)}, prefix="v")
            next_version = 1 if previous is None else previous.global_version + 1
            manifest = Manifest(
                global_version=next_version,
                config_hash=config_hash,
                data_fingerprint=data_fingerprint,
                stage_versions=stage_versions,
            )
            (temp_dir / MANIFEST_NAME).write_text(
                json.dumps(manifest.to_dict(), sort_keys=True, indent=2),
                encoding="utf-8",
            )
            # Version comparison gate: never overwrite without a version change.
            if previous is not None \
                    and manifest.stage_versions == previous.stage_versions \
                    and manifest.config_hash == previous.config_hash \
                    and manifest.data_fingerprint == previous.data_fingerprint:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return previous
            self._atomic_replace(temp_dir)
            return manifest
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _atomic_replace(self, temp_dir: Path) -> None:
        backup = self.root / "backup"
        if self.current.exists():
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            os.replace(self.current, backup)
        try:
            os.replace(temp_dir, self.current)
        except OSError:
            if backup.exists():
                os.replace(backup, self.current)
            raise
        finally:
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)

    def reset(self) -> None:
        shutil.rmtree(self.current, ignore_errors=True)
