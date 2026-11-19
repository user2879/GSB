"""Content-addressed snapshot store with atomic manifest replacement."""

from __future__ import annotations

from pathlib import Path

from .errors import ConfigMismatchError
from .utils import (
    ALGORITHM_VERSION,
    atomic_write_bytes,
    atomic_write_json,
    load_json,
    load_npy,
    save_npy,
)

MANIFEST_NAME = "manifest.json"


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.objects_dir = self.root / "objects"
        self.tmp_dir = self.root / "tmp"
        self.manifest_path = self.root / MANIFEST_NAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    # -- manifest -----------------------------------------------------------

    def fresh_manifest(self, config_fingerprint: str) -> dict:
        return {
            "manifest_version": 1,
            "global_version": 0,
            "algorithm_version": ALGORITHM_VERSION,
            "config_fingerprint": config_fingerprint,
            "completed_samples": [],
            "failures": [],
            "samples": {},
            "runs": [],
        }

    def load_manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        return load_json(self.manifest_path)

    def resume_manifest(self, config_fingerprint: str) -> dict:
        manifest = self.load_manifest()
        if manifest is None:
            return self.fresh_manifest(config_fingerprint)
        if manifest.get("config_fingerprint") != config_fingerprint:
            raise ConfigMismatchError(
                "snapshot config fingerprint does not match current config",
                {
                    "snapshot_fingerprint": manifest.get("config_fingerprint"),
                    "current_fingerprint": config_fingerprint,
                },
            )
        if manifest.get("algorithm_version") != ALGORITHM_VERSION:
            raise ConfigMismatchError(
                "snapshot algorithm version is unsupported",
                {
                    "snapshot_algorithm": manifest.get("algorithm_version"),
                    "current_algorithm": ALGORITHM_VERSION,
                },
            )
        return manifest

    def commit_manifest(self, manifest: dict) -> None:
        # Version comparison happens before commit: callers only bump changed
        # entries, so this single atomic swap is the whole-store replacement.
        atomic_write_json(self.manifest_path, manifest)

    # -- immutable objects --------------------------------------------------

    def object_path(self, key: str, suffix: str) -> Path:
        prefix = key[:2]
        directory = self.objects_dir / prefix
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{key}.{suffix}"

    def has_object(self, key: str, suffix: str) -> bool:
        return self.object_path(key, suffix).exists()

    def write_array(self, key: str, array) -> Path:
        path = self.object_path(key, "npy")
        if not path.exists():  # objects are immutable; never overwrite
            save_npy(path, array)
        return path

    def read_array(self, key: str):
        return load_npy(self.object_path(key, "npy"))

    def write_json_object(self, key: str, payload) -> Path:
        path = self.object_path(key, "json")
        if not path.exists():
            atomic_write_json(path, payload)
        return path

    def read_json_object(self, key: str):
        return load_json(self.object_path(key, "json"))

    def write_png(self, key: str, data: bytes) -> Path:
        path = self.object_path(key, "png")
        if not path.exists():
            atomic_write_bytes(path, data)
        return path

    def prune_manifest(self, manifest: dict, keep_names: set) -> int:
        """Drop in-memory manifest entries for samples outside active scope."""
        removed = 0
        for name in list(manifest["samples"].keys()):
            if name not in keep_names:
                del manifest["samples"][name]
                removed += 1
        manifest["completed_samples"] = [
            name for name in manifest["completed_samples"] if name in keep_names
        ]
        manifest["failures"] = [
            failure for failure in manifest["failures"] if failure["sample"] in keep_names
        ]
        return removed
