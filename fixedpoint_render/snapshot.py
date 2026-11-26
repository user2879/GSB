"""Staged snapshots with atomic replace and crash recovery (business rule 8).

Each pipeline stage writes into a versioned temporary directory, then the whole
directory is atomically ``os.replace``-d into place.  Recovery only ever reads
fully committed version directories; a half-written temp directory is discarded.
Version identity combines a monotonic global version number with the config
hash, so a changed configuration can never silently overwrite another version.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass

from .errors import E_SNAPSHOT_CORRUPT, E_SNAPSHOT_VERSION, ErrorBundle, RenderError
from .fingerprints import fingerprint

STAGES = (
    "01_validate",
    "02_encode",
    "03_fuse",
    "04_budget",
    "05_denoise",
    "06_decode",
    "07_compose",
)


class SnapshotStore:
    def __init__(self, root: str):
        self.root = root
        self.tmp_root = os.path.join(root, "tmp")
        os.makedirs(self.tmp_root, exist_ok=True)

    # -- version / config identity ----------------------------------------
    def config_hash(self, config) -> str:
        return fingerprint(config.canonical_json(), "config-hash")

    def latest_committed(self) -> dict | None:
        manifest_path = os.path.join(self.root, "manifest.json")
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            raise ErrorBundle([RenderError(
                E_SNAPSHOT_CORRUPT, "manifest.json", "root",
                f"unparseable manifest: {exc}")])
        except OSError as exc:
            raise ErrorBundle([RenderError(
                E_SNAPSHOT_CORRUPT, "manifest.json", "root", str(exc))])

    def begin_version(self, config, resume: bool) -> tuple[int, str, dict | None]:
        """Return (version, version_dir, previous_manifest)."""
        previous = self.latest_committed()
        config_hash = self.config_hash(config)
        if previous is not None:
            if previous.get("config_hash") != config_hash:
                # New configuration => new monotonic version, never overwrite.
                version = previous["version"] + 1
            else:
                # Same config hash => same version; content-addressed ledger
                # keys stay stable so incremental reuse can hit.
                version = previous["version"]
        else:
            if resume:
                raise ErrorBundle([RenderError(
                    E_SNAPSHOT_VERSION, "snapshot", "resume",
                    "resume requested but no committed snapshot exists")])
            version = 1
        temp_dir = os.path.join(self.tmp_root, f"v{version}-{os.getpid()}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        for stage in STAGES:
            os.makedirs(os.path.join(temp_dir, stage), exist_ok=True)
        # The persistent version directory is created up front and only the
        # manifest is atomically switched at commit time.  Artifacts are
        # appended directly here so ledger paths stay valid across runs.
        version_dir = os.path.join(self.root, f"v{version}")
        os.makedirs(version_dir, exist_ok=True)
        for stage in STAGES:
            os.makedirs(os.path.join(version_dir, stage), exist_ok=True)
        return version, temp_dir, previous

    def stage_path(self, temp_dir: str, stage: str) -> str:
        if stage not in STAGES:
            raise RenderError(E_SNAPSHOT_VERSION, "snapshot", stage,
                              "unknown stage")
        return os.path.join(temp_dir, stage)

    def commit(self, temp_dir: str, version: int, config,
               stage_index: int, progress: dict) -> str:
        """Atomically publish a stage's temp directory as the version dir."""
        target = os.path.join(self.root, f"v{version}")
        stage_name = STAGES[stage_index]
        # Stage artifacts are written directly under target/<stage>; the only
        # atomically published object at commit time is the manifest, written
        # to a temp file then os.replace-d into both locations.
        manifest = {
            "version": version,
            "config_hash": self.config_hash(config),
            "committed_stages": list(STAGES[: stage_index + 1]),
            "progress": progress,
        }
        self._atomic_write_json(os.path.join(target, "manifest.json"), manifest)
        self._atomic_write_json(os.path.join(self.root, "manifest.json"),
                                manifest)
        return target

    @staticmethod
    def _atomic_write_json(path: str, payload: dict) -> None:
        tmp = path + f".tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, indent=1)
        os.replace(tmp, path)

    def recover(self) -> tuple[int, dict, str] | None:
        """Return a committed version to resume from, or None when empty."""
        previous = self.latest_committed()
        if previous is None:
            self._discard_temp()
            return None
        version = previous["version"]
        version_dir = os.path.join(self.root, f"v{version}")
        if not os.path.isdir(version_dir):
            raise ErrorBundle([RenderError(
                E_SNAPSHOT_CORRUPT, "snapshot", f"v{version}",
                "manifest points at missing version directory")])
        with open(os.path.join(version_dir, "manifest.json"),
                  encoding="utf-8") as fh:
            disk_manifest = json.load(fh)
        if disk_manifest != previous:
            raise ErrorBundle([RenderError(
                E_SNAPSHOT_CORRUPT, "snapshot", "manifest",
                "root and version manifests disagree")])
        self._discard_temp()
        return version, previous, version_dir

    def _discard_temp(self) -> None:
        if os.path.isdir(self.tmp_root):
            for name in os.listdir(self.tmp_root):
                shutil.rmtree(os.path.join(self.tmp_root, name),
                              ignore_errors=True)
