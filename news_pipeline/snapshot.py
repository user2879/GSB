"""Staged atomic snapshots, manifest versioning and interrupt recovery."""

import json
import os
import shutil

from .errors import E_SNAPSHOT_STALE, PipelineError
from .hashing import fingerprint

STAGES = ("segmentation", "scoring", "selection", "rewrite", "summary")
MANIFEST_NAME = "manifest.json"


class SnapshotStore:
    def __init__(self, root, tmp_root=None):
        self.root = root
        self.tmp_root = tmp_root or os.path.join(root, "tmp")
        self.manifest_path = os.path.join(root, MANIFEST_NAME)
        self.manifest = None
        self.stale_events = []
        os.makedirs(root, exist_ok=True)
        os.makedirs(self.tmp_root, exist_ok=True)

    def load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        return self.manifest

    def is_compatible(self, config, table):
        manifest = self.load_manifest()
        if manifest is None:
            return False, None
        expected = {
            "config_fingerprint": config.fingerprint,
            "synonym_table_version": table.version,
            "table_fingerprint": table.fingerprint,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                event = {
                    "code": E_SNAPSHOT_STALE,
                    "message": "manifest %s mismatch" % key,
                    "expected": value,
                    "actual": manifest.get(key),
                }
                self.stale_events.append(event)
                return False, event
        return True, manifest

    def reset_for_rebuild(self):
        for name in os.listdir(self.root):
            path = os.path.join(self.root, name)
            if name == "tmp":
                continue
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        self.manifest = None

    def commit_stage(self, stage, stage_artifacts, manifest_updates, global_version):
        """Write stage artifacts to a temp dir then atomically replace it."""
        if stage not in STAGES:
            raise PipelineError(E_SNAPSHOT_STALE, "unknown stage %r" % stage)
        existing_payload = self.read_stage(stage)
        merged_artifacts = dict(existing_payload["artifacts"]) if existing_payload else {}
        merged_artifacts.update(stage_artifacts)
        final_stage = os.path.join(self.root, stage)
        staging = os.path.join(self.root, ".staging-%s" % stage)
        if os.path.exists(staging):
            shutil.rmtree(staging)
        os.makedirs(staging, exist_ok=True)
        payload = {
            "stage": stage,
            "global_version": global_version,
            "artifacts": merged_artifacts,
        }
        with open(os.path.join(staging, "artifacts.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        if os.path.exists(final_stage):
            obsolete = os.path.join(self.root, ".obsolete-%s" % stage)
            if os.path.exists(obsolete):
                shutil.rmtree(obsolete)
            os.rename(final_stage, obsolete)
            os.rename(staging, final_stage)
            shutil.rmtree(obsolete)
        else:
            os.rename(staging, final_stage)

        manifest = self.load_manifest() or {
            "global_version": global_version,
            "stage_versions": {},
            "completed_doc_cursor": [],
        }
        manifest["global_version"] = global_version
        manifest.update(manifest_updates)
        manifest.setdefault("stage_versions", {})[stage] = {
            "global_version": global_version,
            "artifact_fingerprint": fingerprint(stage_artifacts),
        }
        manifest["completed_doc_cursor"] = manifest_updates.get(
            "completed_doc_cursor", manifest.get("completed_doc_cursor", [])
        )
        tmp_manifest = os.path.join(self.tmp_root, MANIFEST_NAME)
        with open(tmp_manifest, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(tmp_manifest, self.manifest_path)
        self.manifest = manifest
        return manifest

    def read_stage(self, stage):
        path = os.path.join(self.root, stage, "artifacts.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
