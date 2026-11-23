"""Persistent cache and snapshot layout with staged atomic replacement."""

import os

from . import fingerprints as fp

SNAPSHOT_VERSION = 1
MANIFEST_NAME = "manifest.json"


class WorkStore:
    """Owns the on-disk cache and snapshot directories for one pipeline run."""

    def __init__(self, workdir):
        self.workdir = workdir
        self.cache_dir = os.path.join(workdir, "cache")
        self.snapshot_dir = os.path.join(workdir, "snapshot")
        self.stage_tmp_root = os.path.join(workdir, "stage_tmp")
        for sub in ("audio", "emission", "decode", "stitch"):
            os.makedirs(os.path.join(self.cache_dir, sub), exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.stage_tmp_root, exist_ok=True)

    def cache_path(self, stage, key):
        return os.path.join(self.cache_dir, stage, f"{key}.json")

    def has(self, stage, key):
        return os.path.exists(self.cache_path(stage, key))

    def load(self, stage, key):
        return fp.read_json(self.cache_path(stage, key))

    def store(self, stage, key, payload):
        fp.atomic_write_json(self.cache_path(stage, key), payload)

    @property
    def manifest_path(self):
        return os.path.join(self.snapshot_dir, MANIFEST_NAME)

    def read_manifest(self):
        if not os.path.exists(self.manifest_path):
            return None
        return fp.read_json(self.manifest_path)

    def write_manifest(self, manifest):
        # Stage-tmp then atomic replace, matching the snapshot commit rule.
        tmp_dir = os.path.join(self.stage_tmp_root, "snapshot")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, MANIFEST_NAME)
        fp.atomic_write_json(tmp_path, manifest)
        os.replace(tmp_path, self.manifest_path)
