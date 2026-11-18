"""Staged snapshot store with atomic whole-artifact replacement."""

import json
import os

from .errors import BusinessError, E_SNAPSHOT_MISMATCH
from .fingerprints import config_hash

STAGES = ("validate", "fingerprint", "gather")


class SnapshotStore:
    def __init__(self, root):
        self.root = root
        self.stages_dir = os.path.join(root, "stages")
        self.tmp_dir = os.path.join(root, "tmp")
        os.makedirs(self.stages_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    # ----------------------------------------------------------------- paths
    @property
    def manifest_path(self):
        return os.path.join(self.root, "manifest.json")

    @property
    def progress_path(self):
        return os.path.join(self.root, "progress.json")

    def stage_path(self, stage):
        return os.path.join(self.stages_dir, f"{stage}.json")

    # ------------------------------------------------------------ atomic I/O
    def _atomic_write(self, path, payload):
        tmp = os.path.join(self.tmp_dir,
                           f".{os.path.basename(path)}.{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    # -------------------------------------------------------------- manifest
    def load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def new_manifest(self, config, layout_fp):
        return {
            "global_version": 0,
            "config_hash": config_hash(config.structural_dict()),
            "quant_level": config.quant_level,
            "agg_mode": config.agg_mode,
            "layout_fp": layout_fp,
            "stages": {},
        }

    def commit_stage(self, manifest, stage, payload):
        """Write the stage artifact to tmp then atomically replace, bump
        the global version and atomically replace the manifest."""
        manifest["global_version"] += 1
        payload = dict(payload)
        payload["version"] = manifest["global_version"]
        self._atomic_write(self.stage_path(stage), payload)
        manifest["stages"][stage] = {
            "version": manifest["global_version"],
            "path": os.path.relpath(self.stage_path(stage), self.root),
        }
        self._atomic_write(self.manifest_path, manifest)
        return manifest["global_version"]

    def load_stage(self, stage):
        path = self.stage_path(stage)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # -------------------------------------------------------------- recovery
    def verify_manifest(self, manifest, config, layout_fp=None):
        """Raise E_SNAPSHOT_MISMATCH unless the manifest matches the
        current config hash, quantization level and block layout."""
        problems = []
        expected_cfg = config_hash(config.structural_dict())
        if manifest.get("config_hash") != expected_cfg:
            problems.append("config_hash")
        if manifest.get("quant_level") != config.quant_level:
            problems.append("quant_level")
        if layout_fp is not None and manifest.get("layout_fp") != layout_fp:
            problems.append("layout_fp")
        if problems:
            raise BusinessError(E_SNAPSHOT_MISMATCH, {
                "mismatched_fields": problems,
                "manifest_version": manifest.get("global_version"),
            })
        return manifest

    # -------------------------------------------------------------- progress
    def write_progress(self, progress):
        self._atomic_write(self.progress_path, progress)

    def load_progress(self):
        if not os.path.exists(self.progress_path):
            return None
        with open(self.progress_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
