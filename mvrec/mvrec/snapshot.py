"""Stage snapshots with whole-directory atomic replacement semantics."""

import json
import os
import shutil

from .config import camera_hash, config_hash
from .errors import E_SNAPSHOT_INVALID, PipelineError

MANIFEST_NAME = "manifest.json"
STAGE_NAMES = ("features", "matching", "depth", "merge")


def atomic_write_json(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def atomic_write_text(path, text):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


class SnapshotStore:
    """Owns the snapshot root and the monotonically versioned manifest."""

    def __init__(self, root, config, cameras):
        self.root = root
        self.config = config
        self.config_hash = config_hash(config)
        self.camera_hash = camera_hash(cameras)
        os.makedirs(root, exist_ok=True)
        self.manifest_path = os.path.join(root, MANIFEST_NAME)
        self.manifest = self._load_or_init()

    def _load_or_init(self):
        if not os.path.exists(self.manifest_path):
            return {
                "global_version": 0,
                "config_hash": self.config_hash,
                "camera_hash": self.camera_hash,
                "stage_versions": {name: 0 for name in STAGE_NAMES},
                "stage_paths": {name: None for name in STAGE_NAMES},
                "group_progress": {},
                "consumed_slots": 0,
                "consumed_points": 0,
                "deduction_sequence": [],
                "pending_queue": [],
                "failures": [],
                "last_resume_point": None,
            }
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("config_hash") != self.config_hash:
            raise PipelineError(
                E_SNAPSHOT_INVALID,
                "config hash does not match the existing snapshot",
            )
        if manifest.get("camera_hash") != self.camera_hash:
            raise PipelineError(
                E_SNAPSHOT_INVALID,
                "camera hash does not match the existing snapshot",
            )
        return manifest

    def verify_resumable(self):
        """Return a structured resume point derived only from committed data."""
        committed = sorted(
            int(key) for key, value in self.manifest["group_progress"].items()
            if value.get("status") == "committed"
        )
        resume_point = {
            "global_version": self.manifest["global_version"],
            "stage_versions": dict(self.manifest["stage_versions"]),
            "committed_groups": committed,
            "consumed_slots": self.manifest["consumed_slots"],
            "consumed_points": self.manifest["consumed_points"],
            "pending_queue": list(self.manifest["pending_queue"]),
        }
        self.manifest["last_resume_point"] = resume_point
        atomic_write_json(self.manifest_path, self.manifest)
        return resume_point

    def stage_dir(self, stage_name, version):
        return os.path.join(self.root, "%s_v%06d" % (stage_name, version))

    def commit_stage(self, stage_name, build_fn):
        """Build the new stage tree in a temp dir then atomically replace it."""
        version = self.manifest["stage_versions"][stage_name] + 1
        target = self.stage_dir(stage_name, version)
        temp_dir = os.path.join(self.root, ".tmp_%s_v%06d" % (stage_name, version))
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        build_fn(temp_dir)
        os.replace(temp_dir, target)
        previous = self.manifest["stage_paths"].get(stage_name)
        self.manifest["stage_versions"][stage_name] = version
        self.manifest["stage_paths"][stage_name] = target
        if previous and os.path.isdir(previous) and previous != target:
            shutil.rmtree(previous)
        self._bump_manifest()
        return target

    def update_manifest(self, mutation_fn):
        mutation_fn(self.manifest)
        self._bump_manifest()

    def _bump_manifest(self):
        self.manifest["global_version"] += 1
        atomic_write_json(self.manifest_path, self.manifest)
