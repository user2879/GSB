"""Versioned snapshots, atomic replacement, signals and crash recovery."""

from __future__ import annotations

import json
import os
import signal
import tempfile
from pathlib import Path

from .fingerprint import sha256_text, stable_json_dumps


CURRENT_POINTER = "current.json"
MANIFEST_NAME = "manifest.json"


class GracefulExit:
    """Raised between batches after SIGTERM/SIGINT so the current batch and
    its manifest replacement finish before exiting with code 0."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, _frame) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshots_dir = self.root / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        pointer = self.root / CURRENT_POINTER
        self.current = json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else None

    def current_manifest(self) -> dict | None:
        if not self.current:
            return None
        manifest_path = Path(self.current["snapshot_dir"]) / MANIFEST_NAME
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def begin_snapshot(self, manifest: dict) -> Path:
        """Build a fully isolated temp staging directory first."""
        staging = Path(
            tempfile.mkdtemp(prefix=f"snap-v{manifest['global_version']}-", dir=self.snapshots_dir)
        )
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return staging

    def commit_snapshot(self, staging: Path, manifest: dict) -> dict:
        """Verify manifest version drift, then atomically swap the pointer."""
        prior = self.current_manifest()
        if prior is not None and prior["global_version"] >= manifest["global_version"]:
            raise RuntimeError(
                f"refuse to swap: manifest version {manifest['global_version']} "
                f"is not newer than committed {prior['global_version']}"
            )
        manifest_path = staging / MANIFEST_NAME
        written = json.loads(manifest_path.read_text(encoding="utf-8"))
        written["artifact_hashes"] = _hash_artifacts(staging)
        manifest_path.write_text(
            json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        final_dir = self.snapshots_dir / f"v{manifest['global_version']:06d}"
        if final_dir.exists():
            raise RuntimeError(f"snapshot directory already exists: {final_dir}")
        os.replace(staging, final_dir)

        manifest_path = final_dir / MANIFEST_NAME
        pointer = self.root / CURRENT_POINTER
        pointer_payload = {
            "snapshot_dir": str(final_dir),
            "global_version": manifest["global_version"],
            "manifest_hash": sha256_text(
                manifest_path.read_text(encoding="utf-8")
            ),
        }
        tmp_pointer = pointer.with_suffix(".json.tmp")
        tmp_pointer.write_text(
            json.dumps(pointer_payload, ensure_ascii=False, indent=2,
                       sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_pointer, pointer)
        self.current = pointer_payload
        return pointer_payload

    def recovery_point(self) -> dict:
        """Recovery trusts only fully replaced (pointer-visible) snapshots."""
        prior = self.current_manifest()
        if prior is None:
            return {
                "recoverable": False,
                "reason": "no_committed_snapshot",
                "committed_snapshot": None,
            }
        snapshot_dir = Path(self.current["snapshot_dir"])
        missing = [
            name
            for name, version in prior.get("stage_versions", {}).items()
            if not (snapshot_dir / name).exists()
        ]
        return {
            "recoverable": not missing,
            "reason": "complete_snapshot" if not missing else "partial_artifacts",
            "committed_snapshot": str(snapshot_dir),
            "global_version": prior["global_version"],
            "config_hash": prior.get("config_hash"),
            "metadata_hash": prior.get("metadata_hash"),
            "source_fingerprints": prior.get("source_fingerprints", {}),
            "stage_versions": prior.get("stage_versions", {}),
            "missing_artifacts": missing,
        }


def _hash_artifacts(directory: Path) -> dict:
    hashes: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != MANIFEST_NAME:
            rel = path.relative_to(directory).as_posix()
            hashes[rel] = sha256_text(path.read_text(encoding="utf-8"))
    return hashes


def build_manifest(
    global_version: int,
    config_hash: str,
    metadata_hash: str,
    source_fingerprints: dict,
    stage_versions: dict,
) -> dict:
    manifest = {
        "global_version": global_version,
        "config_hash": config_hash,
        "metadata_hash": metadata_hash,
        "source_fingerprints": source_fingerprints,
        "stage_versions": stage_versions,
    }
    manifest["manifest_body_hash"] = sha256_text(stable_json_dumps(manifest))
    return manifest
