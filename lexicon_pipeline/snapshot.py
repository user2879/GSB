"""Versioned stage snapshots with temp-directory commit and recovery."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .errors import E_SNAPSHOT_STALE
from .hashing import fingerprint


class SnapshotManager:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.releases = self.root / "releases"
        self.current_link = self.root / "current"

    def _write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        tmp.replace(path)

    def recover_interrupted(self) -> list[str]:
        if not self.root.exists():
            return []
        removed: list[str] = []
        for staging in self.root.glob(".staging-*"):
            if staging.is_dir():
                shutil.rmtree(staging)
                removed.append(str(staging))
        return removed

    def load_current_manifest(self) -> dict[str, Any] | None:
        manifest_path = self.current_link / "manifest.json"
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def startup_decision(
        self,
        config_hash: str,
        lexicon_revision: int | str,
        expected_snapshot_version: int | None = None,
    ) -> dict[str, Any]:
        removed = self.recover_interrupted()
        manifest = self.load_current_manifest()
        if manifest is None:
            return {
                "action": "rebuild",
                "code": E_SNAPSHOT_STALE,
                "reason": "committed snapshot manifest is missing",
                "removed_staging": removed,
            }
        if expected_snapshot_version is not None and manifest.get("snapshot_version") != expected_snapshot_version:
            return {
                "action": "rebuild",
                "reason": "snapshot-version-changed",
                "manifest_version": manifest.get("snapshot_version"),
                "expected_version": expected_snapshot_version,
                "removed_staging": removed,
            }
        if manifest.get("config_hash") != config_hash:
            return {"action": "rebuild", "reason": "config-hash-changed", "manifest_version": manifest.get("snapshot_version"), "removed_staging": removed}
        if manifest.get("lexicon_revision") != lexicon_revision:
            return {"action": "rebuild", "reason": "lexicon-revision-changed", "manifest_version": manifest.get("snapshot_version"), "removed_staging": removed}
        return {"action": "resume", "reason": "identity-match", "manifest": manifest, "removed_staging": removed}

    def commit(
        self,
        stages: dict[str, Any],
        identity: dict[str, Any],
        pending: list[dict[str, Any]],
        cursor: dict[str, Any],
        consumed_budget: int,
        lexicon_revision: int | str,
    ) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        old_manifest = self.load_current_manifest()
        if old_manifest is None:
            snapshot_version = 1
            generation = 1
        else:
            identity_changed = (
                old_manifest.get("config_hash") != identity.get("config_hash")
                or old_manifest.get("lexicon_revision") != lexicon_revision
            )
            snapshot_version = int(old_manifest.get("snapshot_version", 0)) + 1
            generation = int(old_manifest.get("generation", 1)) + (1 if identity_changed else 0)

        commit_id = fingerprint({"v": snapshot_version, "g": generation, "identity": identity}, 12)
        staging = self.root / f".staging-{snapshot_version}-{commit_id}"
        release = self.releases / f"snapshot-{snapshot_version}-{commit_id}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        for stage_name, product in stages.items():
            self._write_json(staging / "stages" / f"{stage_name}.json", product)
        state = {
            "pending": pending,
            "completed_cursor": cursor,
            "consumed_budget": consumed_budget,
            "lexicon_revision": lexicon_revision,
        }
        self._write_json(staging / "state.json", state)
        manifest = {
            "snapshot_version": snapshot_version,
            "generation": generation,
            "commit_id": commit_id,
            "config_hash": identity.get("config_hash"),
            "lexicon_revision": lexicon_revision,
            "exchange_version": identity.get("exchange_version"),
            "effective_fraction_bits": identity.get("effective_fraction_bits"),
            "quantization_tier": identity.get("quantization_tier"),
            "stages": sorted(stages),
            "state_file": "state.json",
        }
        self._write_json(staging / "manifest.json", manifest)

        release.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, release)
        temp_link = self.root / f".current-tmp-{commit_id}"
        if temp_link.exists() or temp_link.is_symlink():
            temp_link.unlink()
        relative_target = os.path.relpath(release, self.root)
        temp_link.symlink_to(relative_target, target_is_directory=True)
        os.replace(temp_link, self.current_link)
        return manifest

    def simulate_forced_kill_after_staging(self, stages: dict[str, Any]) -> Path:
        """Create a complete staging directory as if SIGKILL arrived before rename."""
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / ".staging-kill-demo"
        staging.mkdir(parents=True, exist_ok=True)
        for name, product in stages.items():
            self._write_json(staging / "stages" / f"{name}.json", product)
        self._write_json(staging / "manifest.json", {"snapshot_version": 999, "kill_injected": True})
        os.kill(os.getpid(), 9)
        return staging  # pragma: no cover - os.kill never returns
