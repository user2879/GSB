"""Staged snapshots: temp-write then atomic whole-replace commit.

Each stage artifact is a single JSON file. The manifest records the global
version, config hash, source-listing hashes and the completed-sample cursor.
Commits re-check the observed manifest to detect version changes instead of
blindly overwriting files one by one.
"""

import json
import os
from pathlib import Path

STAGES = ("load", "detect", "ledger", "arbitrate")


class StaleCommit(RuntimeError):
    pass


class SnapshotStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self):
        return self.root / "manifest.json"

    def _stage_path(self, stage):
        return self.root / f"{stage}.json"

    def read_manifest(self):
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return None

    def stage_artifact(self, stage):
        path = self._stage_path(stage)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def completed_stages(self):
        manifest = self.read_manifest()
        return set(manifest.get("completed_stages", [])) if manifest else set()

    def reset(self):
        for stage in STAGES:
            path = self._stage_path(stage)
            if path.exists():
                path.unlink()
        if self.manifest_path.exists():
            self.manifest_path.unlink()

    @staticmethod
    def _atomic_write(path, payload):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    def commit(self, stage, artifact, observed, *, version, config_hash,
               source_hashes, cursor):
        """Commit one stage; raises StaleCommit if the manifest changed since
        `observed` was read (version change detected before replacing files)."""
        current = self.read_manifest()
        if (current or None) != (observed or None):
            raise StaleCommit(f"manifest changed before committing stage {stage}")
        self._atomic_write(self._stage_path(stage), artifact)
        manifest = dict(current or {})
        completed = set(manifest.get("completed_stages", []))
        completed.add(stage)
        manifest.update(
            {
                "version": version,
                "config_hash": config_hash,
                "source_hashes": source_hashes,
                "completed_stages": [s for s in STAGES if s in completed],
                "cursor": cursor,
            }
        )
        self._atomic_write(self.manifest_path, manifest)
