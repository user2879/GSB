"""Stage snapshots committed by whole-directory atomic replacement."""

import json
import os
import shutil

SCHEMA_VERSION = 1
STAGES = ("evidence", "boundary", "trajectory", "summary")


def _atomic_write_json(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class SnapshotManager:
    """Commits per-stage snapshots into snapshots/current via temp-dir replace."""

    def __init__(self, workdir):
        self.snapshots_dir = os.path.join(workdir, "snapshots")
        self.current_dir = os.path.join(self.snapshots_dir, "current")
        self.tmp_dir = os.path.join(self.snapshots_dir, "tmp")

    def load_manifest(self):
        path = os.path.join(self.current_dir, "manifest.json")
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def commit(self, manifest):
        """Whole-directory replacement: build tmp, fsync, then os.replace."""
        os.makedirs(self.snapshots_dir, exist_ok=True)
        if os.path.isdir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir)
        _atomic_write_json(os.path.join(self.tmp_dir, "manifest.json"), manifest)
        if os.path.isdir(self.current_dir):
            old = self.current_dir + ".old"
            if os.path.isdir(old):
                shutil.rmtree(old)
            os.replace(self.current_dir, old)
            os.replace(self.tmp_dir, self.current_dir)
            shutil.rmtree(old)
        else:
            os.replace(self.tmp_dir, self.current_dir)


def build_manifest(version, config_hash, stage, stage_versions, sequences,
                   consumed_budget, segment_summaries, resident_block_keys):
    return {
        "schema_version": SCHEMA_VERSION,
        "version": int(version),
        "config_hash": config_hash,
        "stage": stage,
        "stage_versions": dict(stage_versions),
        "sequences": sequences,          # name -> {"cursor": int, "completed": bool}
        "consumed_global_frame_budget": int(consumed_budget),
        "confirmed_segment_summaries": segment_summaries,
        "resident_block_keys": list(resident_block_keys),
    }
