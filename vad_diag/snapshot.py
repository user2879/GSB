"""Staged snapshots with atomic whole-directory replacement and recovery.

Four pipeline stages are snapshotted independently and versioned together:

    load -> vad -> window -> posterior

Each commit writes every stage product, the pending work queue, the completed
segment cursor and the consumed recompute-window point budget into a *temp*
directory, then performs a single ``os.replace`` style swap so readers only ever
see a fully consistent snapshot.  A ``manifest.json`` records the global version
number, configuration hash, quantization level and per-stage product versions;
startup identifies a changed configuration by hash/version rather than by
comparing individual files.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .fingerprint import digest

STAGES = ("load", "vad", "window", "posterior")
MANIFEST_NAME = "manifest.json"
SNAPSHOT_CURRENT = "current"
SNAPSHOT_TMP_PREFIX = ".tmp-snap-"


@dataclass
class SnapshotState:
    version: int = 0
    config_hash: str = ""
    quant_level: str = ""
    stage_versions: Dict[str, int] = field(default_factory=dict)
    completed_segments: List[str] = field(default_factory=list)
    pending_windows: List[str] = field(default_factory=list)
    recompute_budget_used: int = 0
    products: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def manifest_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "config_hash": self.config_hash,
            "quant_level": self.quant_level,
            "stage_versions": self.stage_versions,
            "completed_segments": list(self.completed_segments),
            "pending_windows": list(self.pending_windows),
            "recompute_budget_used": self.recompute_budget_used,
        }


class SnapshotStore:
    def __init__(self, root: str):
        self.root = root
        self.current_dir = os.path.join(root, SNAPSHOT_CURRENT)

    def ensure_dirs(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.current_dir, exist_ok=True)

    def _stage_path(self, stage: str) -> str:
        return os.path.join(self.current_dir, f"{stage}.json")

    def exists(self) -> bool:
        return os.path.isfile(
            os.path.join(self.current_dir, MANIFEST_NAME))

    def save(self, state: SnapshotState, *, crash_point: Optional[str] = None
             ) -> None:
        """Atomically commit a snapshot.

        ``crash_point`` simulates a forced termination: ``"before_swap"`` leaves
        the temp directory in place without replacing ``current``; the next
        startup must recover by ignoring it.
        """
        self.ensure_dirs()
        tmp_dir = tempfile.mkdtemp(prefix=SNAPSHOT_TMP_PREFIX, dir=self.root)
        crashed = crash_point == "before_swap"
        try:
            for stage in STAGES:
                product = state.products.get(stage, {})
                with open(os.path.join(tmp_dir, f"{stage}.json"), "w",
                          encoding="utf-8") as handle:
                    json.dump(product, handle, sort_keys=True,
                              separators=(",", ":"), ensure_ascii=False)
            with open(os.path.join(tmp_dir, MANIFEST_NAME), "w",
                      encoding="utf-8") as handle:
                json.dump(state.manifest_dict(), handle, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False)

            if crash_point == "before_swap":
                # Simulate SIGKILL: temp dir remains, current is untouched.
                return

            # Whole-directory atomic replacement.
            backup = None
            if os.path.isdir(self.current_dir):
                backup = self.current_dir + ".old"
                if os.path.exists(backup):
                    shutil.rmtree(backup)
                os.replace(self.current_dir, backup)
            os.replace(tmp_dir, self.current_dir)
            if backup and os.path.isdir(backup):
                shutil.rmtree(backup)
        finally:
            # On a simulated forced termination the temp directory is
            # intentionally abandoned so recovery can prove it is ignored.
            if not crashed and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)

    def load(self) -> Optional[SnapshotState]:
        manifest_path = os.path.join(self.current_dir, MANIFEST_NAME)
        if not os.path.isfile(manifest_path):
            return None
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        state = SnapshotState(
            version=manifest["version"],
            config_hash=manifest["config_hash"],
            quant_level=manifest["quant_level"],
            stage_versions=dict(manifest["stage_versions"]),
            completed_segments=list(manifest["completed_segments"]),
            pending_windows=list(manifest["pending_windows"]),
            recompute_budget_used=manifest["recompute_budget_used"])
        for stage in STAGES:
            path = self._stage_path(stage)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    state.products[stage] = json.load(handle)
        return state

    def recover(self, expected_config_hash: str) -> Optional[SnapshotState]:
        """Decide recovery on startup using version + config hash.

        Returns the persisted state only when its configuration hash matches;
        otherwise the snapshot is treated as from a prior configuration and the
        caller must rebuild from scratch rather than overlay files.
        """
        self._discard_stale_temp()
        state = self.load()
        if state is None:
            return None
        if state.config_hash != expected_config_hash:
            return None
        return state

    def _discard_stale_temp(self) -> None:
        if not os.path.isdir(self.root):
            return
        for name in os.listdir(self.root):
            if name.startswith(SNAPSHOT_TMP_PREFIX):
                shutil.rmtree(os.path.join(self.root, name), ignore_errors=True)

    def reset(self) -> None:
        if os.path.isdir(self.root):
            shutil.rmtree(self.root)
        self.ensure_dirs()


def config_hash(config_dict: Dict[str, Any]) -> str:
    return digest(["config", config_dict])
