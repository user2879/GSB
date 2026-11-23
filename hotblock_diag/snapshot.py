"""Stage-wise snapshots with atomic directory replacement."""
from __future__ import annotations

import json
import os
import shutil
import signal
from dataclasses import dataclass
from typing import Any

from .errors import E_SNAPSHOT_STALE, FailureList
from .util import dump_json

MANIFEST_NAME = "manifest.json"
STAGE_NAMES = ("tensor_parse", "block_scoring", "selection", "hot_summary")


@dataclass
class Manifest:
    global_version: int
    config_hash: str
    tensor_fingerprint: str
    completed_rounds: int
    stage_versions: dict[str, int]
    snapshot_interval: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "tensor_fingerprint": self.tensor_fingerprint,
            "completed_rounds": self.completed_rounds,
            "stage_versions": self.stage_versions,
            "snapshot_interval": self.snapshot_interval,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Manifest":
        return cls(
            global_version=int(raw["global_version"]),
            config_hash=raw["config_hash"],
            tensor_fingerprint=raw["tensor_fingerprint"],
            completed_rounds=int(raw["completed_rounds"]),
            stage_versions={key: int(value) for key, value in raw["stage_versions"].items()},
            snapshot_interval=int(raw["snapshot_interval"]),
        )


class SnapshotStore:
    def __init__(self, root: str, failures: FailureList):
        self.root = root
        self.failures = failures
        self.staging = os.path.join(root, "staging")
        self.committed = os.path.join(root, "committed")
        os.makedirs(root, exist_ok=True)

    def reset(self) -> None:
        for path in (self.staging, self.committed):
            if os.path.exists(path):
                shutil.rmtree(path)
        os.makedirs(self.committed, exist_ok=True)

    def stage_stage(self, stage: str, payload: dict[str, Any]) -> str:
        directory = os.path.join(self.staging, stage)
        os.makedirs(directory, exist_ok=True)
        dump_json(os.path.join(directory, "data.json"), payload)
        return directory

    def commit(self, manifest: Manifest) -> None:
        dump_json(os.path.join(self.staging, MANIFEST_NAME), manifest.to_dict())
        if os.path.exists(self.committed):
            backup = self.committed + ".old"
            if os.path.exists(backup):
                shutil.rmtree(backup)
            os.replace(self.committed, backup)
            os.replace(self.staging, self.committed)
            shutil.rmtree(backup, ignore_errors=True)
        else:
            os.replace(self.staging, self.committed)
        os.makedirs(self.staging, exist_ok=True)

    def load_manifest(self) -> Manifest | None:
        path = os.path.join(self.committed, MANIFEST_NAME)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return Manifest.from_dict(json.load(handle))

    def load_stage(self, stage: str) -> dict[str, Any] | None:
        path = os.path.join(self.committed, stage, "data.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def resume_or_rebuild(self, config_hash: str, tensor_fingerprint: str) -> Manifest | None:
        manifest = self.load_manifest()
        if manifest is None:
            return None
        expected_min_version = 1
        if (manifest.global_version < expected_min_version
                or manifest.config_hash != config_hash
                or manifest.tensor_fingerprint != tensor_fingerprint):
            self.failures.add(
                E_SNAPSHOT_STALE, "snapshot_restore", self.committed,
                detail=(f"manifest config={manifest.config_hash} tensor={manifest.tensor_fingerprint} "
                        f"vs current config={config_hash} tensor={tensor_fingerprint}"),
            )
            self.reset()
            return None
        return manifest

    def tamper(self) -> None:
        """Test helper: corrupt the committed manifest version."""
        path = os.path.join(self.committed, MANIFEST_NAME)
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        raw["global_version"] = -999
        raw["config_hash"] = "tampered"
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(raw, handle, sort_keys=True, indent=2)
            handle.write("\n")


class SignalFlag:
    """Sets a flag on SIGTERM/SIGINT; the current round always finishes."""

    def __init__(self) -> None:
        self.requested = False
        self._handlers: dict[int, Any] = {}

    def install(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._handlers[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass

    def _handle(self, signum: int, frame: Any) -> None:
        self.requested = True

    def restore(self) -> None:
        for sig, handler in self._handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
