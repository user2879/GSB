"""Atomic snapshot store, manifest bookkeeping and recovery decisions."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ConfigMismatch

MANIFEST_NAME = "manifest.json"


@dataclass
class ArtifactState:
    node_id: str
    cache_key: str
    version: int
    payload_path: str
    fingerprint: int


@dataclass
class Manifest:
    global_version: int
    config_hash: str
    fractional_bits: int
    blend_mode: str
    completed_cursor: list[str]
    artifacts: dict[str, ArtifactState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "fractional_bits": self.fractional_bits,
            "blend_mode": self.blend_mode,
            "completed_cursor": list(self.completed_cursor),
            "artifacts": {
                node_id: {
                    "cache_key": state.cache_key,
                    "version": state.version,
                    "payload_path": state.payload_path,
                    "fingerprint": state.fingerprint,
                }
                for node_id, state in sorted(self.artifacts.items())
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Manifest":
        return cls(
            global_version=int(raw["global_version"]),
            config_hash=raw["config_hash"],
            fractional_bits=int(raw["fractional_bits"]),
            blend_mode=raw["blend_mode"],
            completed_cursor=list(raw.get("completed_cursor", [])),
            artifacts={
                node_id: ArtifactState(
                    node_id=node_id, cache_key=item["cache_key"], version=int(item["version"]),
                    payload_path=item["payload_path"], fingerprint=int(item["fingerprint"]),
                )
                for node_id, item in raw.get("artifacts", {}).items()
            },
        )


class SnapshotStore:
    """Artifact directory where every commit is a temp-file + atomic rename."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / MANIFEST_NAME

    def load_manifest(self) -> Manifest | None:
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return Manifest.from_dict(json.load(handle))

    def commit_manifest(self, manifest: Manifest) -> None:
        self._atomic_write_bytes(
            self.manifest_path,
            json.dumps(manifest.to_dict(), sort_keys=True, indent=2).encode("utf-8"),
        )

    def _atomic_write_bytes(self, destination: Path, payload: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(destination.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, destination)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    def save_npz(self, relative_path: str, **arrays: np.ndarray) -> Path:
        destination = self.root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".npz", dir=str(destination.parent))
        os.close(fd)
        try:
            np.savez_compressed(tmp_name, **arrays)
            os.replace(tmp_name, destination)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise
        return destination

    def save_png_bytes(self, relative_path: str, payload: bytes) -> Path:
        destination = self.root / relative_path
        return self._atomic_write_bytes(destination, payload)

    def load_npz(self, relative_path: str) -> dict[str, np.ndarray]:
        with np.load(self.root / relative_path) as data:
            return {key: data[key] for key in data.files}

    def artifact_exists(self, relative_path: str) -> bool:
        return (self.root / relative_path).exists()

    def check_resume(self, expected_config_hash: str, *, strict: bool) -> Manifest | None:
        manifest = self.load_manifest()
        if manifest is None:
            return None
        if manifest.config_hash != expected_config_hash:
            if strict:
                raise ConfigMismatch(
                    "stored manifest config hash does not match current inputs",
                    details={"stored": manifest.config_hash, "current": expected_config_hash},
                )
        return manifest

    def recovery_point(self, manifest: Manifest) -> dict[str, Any]:
        return {
            "global_version": manifest.global_version,
            "completed_batches": list(manifest.completed_cursor),
            "artifact_count": len(manifest.artifacts),
            "fractional_bits": manifest.fractional_bits,
            "blend_mode": manifest.blend_mode,
        }
