"""Stage snapshots with atomic whole-directory replacement."""
import json
import logging
import os
import shutil

LOG = logging.getLogger("diarization.snapshot")

STAGES = ("windowing", "embedding", "clustering", "segmentation")


class SnapshotManager:
    """Commits per-stage artifacts by building a new snapshot directory in
    a staging area and swapping it into place, never overwriting the live
    snapshot file by file. The manifest records the global version, config
    and injection hashes, the completed-audio cursor, the fixed-point
    width and per-stage artifact versions. Version changes are detected
    before each replacement. Recovery decisions after a hard kill use
    only fully swapped-in snapshots.
    """

    def __init__(self, out_dir, config_hash, injection_hash, frac_bits):
        self.live = os.path.join(out_dir, "snapshots")
        self.staging = self.live + ".new"
        self.backup = self.live + ".bak"
        self.config_hash = config_hash
        self.injection_hash = injection_hash
        self.frac_bits = frac_bits
        self.version = 0
        self.stage_versions = {s: 0 for s in STAGES}
        self.cursor = None
        self.last_manifest = None
        self.recovered = None
        mpath = os.path.join(self.live, "manifest.json")
        if os.path.isfile(mpath):
            try:
                with open(mpath, "r", encoding="utf-8") as fh:
                    self.recovered = json.load(fh)
            except (OSError, json.JSONDecodeError):
                self.recovered = {"corrupt": True}

    def commit(self, stage, artifact, cursor):
        previous = None
        mpath = os.path.join(self.live, "manifest.json")
        if os.path.isfile(mpath):
            try:
                with open(mpath, "r", encoding="utf-8") as fh:
                    previous = json.load(fh).get("version")
            except (OSError, json.JSONDecodeError):
                previous = None
        self.version += 1
        self.stage_versions[stage] += 1
        self.cursor = cursor
        manifest = {
            "version": self.version,
            "previous_version": previous,
            "config_hash": self.config_hash,
            "injection_hash": self.injection_hash,
            "completed_audio_cursor": cursor,
            "frac_bits": self.frac_bits,
            "stage_versions": dict(self.stage_versions),
            "last_stage": stage,
        }
        if previous is not None and previous != self.version - 1:
            LOG.warning("snapshot version jump detected: %s -> %s",
                        previous, self.version)
        if os.path.isdir(self.staging):
            shutil.rmtree(self.staging)
        os.makedirs(os.path.join(self.staging, "stages"))
        for name in STAGES:
            src = os.path.join(self.live, "stages", name + ".json")
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(self.staging, "stages", name + ".json"))
        tmp_art = os.path.join(self.staging, "stages", stage + ".json.tmp")
        with open(tmp_art, "w", encoding="utf-8") as fh:
            json.dump({"manifest_version": self.version, "stage": stage,
                       "artifact": artifact}, fh, indent=1)
        os.replace(tmp_art, os.path.join(self.staging, "stages", stage + ".json"))
        with open(os.path.join(self.staging, "manifest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        if os.path.isdir(self.backup):
            shutil.rmtree(self.backup)
        if os.path.isdir(self.live):
            os.rename(self.live, self.backup)
        os.rename(self.staging, self.live)
        if os.path.isdir(self.backup):
            shutil.rmtree(self.backup)
        self.last_manifest = manifest
        return manifest

    def final_manifest(self):
        return self.last_manifest
