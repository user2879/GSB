"""Atomic stage snapshots with manifest based resume/version detection."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .errors import E_SNAPSHOT_CORRUPT, E_SNAPSHOT_VERSION_MISMATCH, SnapshotCorrupt, SnapshotVersionMismatch
from .fingerprint import canonical_json, sha256_hex
from .models import Artifact


MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ResumeDecision:
    resume: bool
    reason: str
    manifest: Optional[dict]
    code: Optional[str]


class SnapshotStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self.tmp_root = self.root / "tmp"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def _version_dir(self, version: int) -> Path:
        return self.root / f"v{version:06d}"

    def commit_stage(
        self,
        version: int,
        stage: str,
        artifacts: Tuple[Artifact, ...],
        manifest_base: Dict[str, Any],
        budget_used: Dict[str, int],
    ) -> Path:
        target = self._version_dir(version) / "stages" / stage
        temp = self.tmp_root / f"v{version:06d}-{stage}-{os.getpid()}"
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir(parents=True)
        payload = [artifact.to_dict() for artifact in artifacts]
        (temp / "artifacts.json").write_text(
            canonical_json(payload) + "\n", encoding="utf-8"
        )
        completed = set(manifest_base.get("completed_stages", ()))
        completed.add(stage)
        manifest = {
            "global_version": version,
            "config_hash": manifest_base["config_hash"],
            "sample_fingerprint": manifest_base["sample_fingerprint"],
            "receipt_fingerprint": manifest_base["receipt_fingerprint"],
            "completed_stages": sorted(completed),
            "budget_used": dict(budget_used),
            "stage_file_hash": sha256_hex(payload),
        }
        (temp / MANIFEST_NAME).write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = self.tmp_root / f"backup-v{version:06d}-{stage}-{os.getpid()}"
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(temp, target)
        except OSError:
            if backup.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        # Root manifest points at the newest complete stage commit.
        root_manifest = self.root / MANIFEST_NAME
        root_tmp = self.tmp_root / f"{MANIFEST_NAME}.{os.getpid()}"
        root_tmp.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        os.replace(root_tmp, root_manifest)
        return target

    def read_manifest(self) -> Optional[dict]:
        path = self.root / MANIFEST_NAME
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SnapshotCorrupt(f"manifest is not valid JSON: {exc}")

    def resume_check(
        self,
        expected_version: int,
        config_hash: str,
        sample_fingerprint: str,
        receipt_fingerprint: str,
    ) -> ResumeDecision:
        manifest = self.read_manifest()
        if manifest is None:
            return ResumeDecision(False, "no existing manifest", None, None)
        required = ("global_version", "config_hash", "sample_fingerprint",
                    "receipt_fingerprint", "completed_stages", "budget_used")
        if any(key not in manifest for key in required):
            return ResumeDecision(False, "manifest missing required fields", manifest, E_SNAPSHOT_CORRUPT)
        if manifest["global_version"] != expected_version:
            return ResumeDecision(
                False,
                f"version changed {manifest['global_version']} -> {expected_version}; full replacement required",
                manifest,
                E_SNAPSHOT_VERSION_MISMATCH,
            )
        if manifest["config_hash"] != config_hash:
            return ResumeDecision(
                False,
                "config hash changed; resume rejected and a new versioned directory will be used",
                manifest,
                E_SNAPSHOT_VERSION_MISMATCH,
            )
        if (manifest["sample_fingerprint"], manifest["receipt_fingerprint"]) != (
            sample_fingerprint,
            receipt_fingerprint,
        ):
            return ResumeDecision(
                False,
                "sample or receipt fingerprint changed",
                manifest,
                E_SNAPSHOT_VERSION_MISMATCH,
            )
        return ResumeDecision(True, "manifest version/config/fingerprints match", manifest, None)

    def load_stage(self, version: int, stage: str) -> Tuple[Artifact, ...]:
        path = self._version_dir(version) / "stages" / stage / "artifacts.json"
        if not path.exists():
            return ()
        data = json.loads(path.read_text(encoding="utf-8"))
        return tuple(
            Artifact(
                artifact_id=item["artifact_id"],
                stage=item["stage"],
                scope=item["scope"],
                payload=item["payload"],
                digest=item["digest"],
                upstream=tuple(item["upstream"]),
                params=item["params"],
                param_versions=item["param_versions"],
                quant_step=item["quant_step"],
                residual=item["residual"],
            )
            for item in data
        )
