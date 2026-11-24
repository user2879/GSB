"""Stage snapshots: temp-dir writes, atomic replace and versioned manifests.

A stage is only ever visible as a complete, versioned directory.  Commits
compare the staged content fingerprint against the currently committed stage
version; identical content keeps the old version, changed content gets a new
monotonic version directory and the manifest is atomically swapped.
"""

import json
import os
import shutil
from pathlib import Path

from .hashing import stable_hash

STAGES = ("spectrum", "detector", "verdict")


class SnapshotStore:
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.snapshots = self.workspace / "snapshots"
        self.tmp = self.workspace / "tmp"
        self.manifest_path = self.snapshots / "manifest.json"
        for path in (self.snapshots, self.tmp):
            path.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()

    def _load_manifest(self):
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return {
            "global_version": 0,
            "config_hash": None,
            "wave_fingerprint": None,
            "budget_consumed": 0,
            "stages": {},
            "pending_queue": [],
            "committed_run": 0,
        }

    def save_manifest(self):
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self.manifest, fh, sort_keys=True, indent=2)
        os.replace(tmp, self.manifest_path)

    def write_stage_temp(self, stage, payload):
        stage_tmp = self.tmp / stage
        if stage_tmp.exists():
            shutil.rmtree(stage_tmp)
        stage_tmp.mkdir(parents=True)
        for name, content in payload.items():
            target = stage_tmp / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, str):
                target.write_text(content, encoding="utf-8")
            else:
                with target.with_suffix(target.suffix + ".tmp").open(
                        "w", encoding="utf-8") as fh:
                    json.dump(content, fh, sort_keys=True, separators=(",", ":"))
                os.replace(target.with_suffix(target.suffix + ".tmp"), target)
        fingerprint = stable_hash(payload)
        return stage_tmp, fingerprint

    def commit_stage(self, stage, payload):
        """Commit a stage only when its content actually changed."""
        stage_tmp, fingerprint = self.write_stage_temp(stage, payload)
        existing = self.manifest["stages"].get(stage)
        if existing and existing["fingerprint"] == fingerprint:
            shutil.rmtree(stage_tmp)
            return existing["version"], False
        new_version = (existing["version"] + 1) if existing else 1
        destination = self.snapshots / stage / f"v{new_version}"
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_tmp, destination)
        self.manifest["global_version"] += 1
        self.manifest["stages"][stage] = {
            "version": new_version,
            "fingerprint": fingerprint,
            "path": str(destination.relative_to(self.workspace)),
        }
        self.manifest["committed_run"] = self.manifest.get("committed_run", 0) + 1
        self.save_manifest()
        return new_version, True

    def load_stage(self, stage):
        info = self.manifest["stages"].get(stage)
        if not info:
            return None
        base = self.workspace / info["path"]
        payload = {}
        for entry in sorted(base.rglob("*")):
            if entry.is_file():
                relative = str(entry.relative_to(base))
                if entry.suffix == ".json":
                    with entry.open("r", encoding="utf-8") as fh:
                        payload[relative] = json.load(fh)
                else:
                    payload[relative] = entry.read_text(encoding="utf-8")
        return payload

    def reset_for_new_config(self, config_hash, wave_fingerprint):
        changed = (self.manifest["config_hash"] != config_hash or
                   self.manifest["wave_fingerprint"] != wave_fingerprint)
        if changed:
            self.manifest["config_hash"] = config_hash
            self.manifest["wave_fingerprint"] = wave_fingerprint
            self.manifest["budget_consumed"] = 0
            self.manifest["pending_queue"] = []
            self.manifest["stages"] = {}
            self.save_manifest()
        return changed
