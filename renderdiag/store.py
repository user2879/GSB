from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .errors import E_CACHE_CORRUPT, E_SNAPSHOT_CORRUPT, fail
from .util import atomic_write_bytes, atomic_write_json, canonical_json, fingerprint, load_json


class ArtifactStore:
    """Content-addressed, atomically published stage artifacts."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    def _artifact_dir(self, key: str) -> Path:
        return self.objects / key[:2] / key

    def _load_artifact_file(self, path: Path, expected_key: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            artifact = load_json(path, "cache.artifact")
        except Exception as exc:
            if isinstance(exc, OSError):
                fail(E_CACHE_CORRUPT, "cache.artifact", "cache artifact cannot be read", detail={"path": str(path)})
            raise
        if artifact.get("format") != "renderdiag-artifact-v1":
            fail(E_CACHE_CORRUPT, "cache.artifact.format", "unknown cache artifact format")
        if artifact.get("key") != expected_key:
            fail(
                E_CACHE_CORRUPT,
                "cache.artifact.key",
                "cache artifact key does not match its location",
                detail={"expected": expected_key, "actual": artifact.get("key")},
            )
        payload = artifact.get("payload")
        if not isinstance(payload, dict):
            fail(E_CACHE_CORRUPT, "cache.artifact.payload", "cache artifact payload is missing")
        computed = fingerprint(payload, "artifact-payload-v1")
        if computed != artifact.get("payload_fingerprint"):
            fail(
                E_CACHE_CORRUPT,
                "cache.artifact.payload_fingerprint",
                "cache artifact payload fingerprint mismatch",
                detail={"expected": artifact.get("payload_fingerprint"), "actual": computed},
            )
        return artifact

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._artifact_dir(key) / "artifact.json"
        artifact = self._load_artifact_file(path, key)
        if artifact is None:
            return None
        return artifact

    def put(
        self,
        *,
        stage: str,
        key: str,
        upstream_fingerprint: str,
        parameter_fingerprint: str,
        quantization_level: int,
        snapshot_version: int,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        path = self._artifact_dir(key)
        if path.exists():
            existing = self.get(key)
            if existing is None:
                fail(E_CACHE_CORRUPT, "cache.artifact", "artifact directory exists without artifact.json")
            if existing.get("payload_fingerprint") != fingerprint(payload, "artifact-payload-v1"):
                fail(
                    E_CACHE_CORRUPT,
                    "cache.artifact.payload_fingerprint",
                    "existing content-addressed artifact conflicts with the new payload",
                    detail={"key": key},
                )
            return existing, True

        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=str(parent)))
        try:
            artifact = {
                "format": "renderdiag-artifact-v1",
                "key": key,
                "stage": stage,
                "upstream_fingerprint": upstream_fingerprint,
                "parameter_fingerprint": parameter_fingerprint,
                "quantization_level": int(quantization_level),
                "snapshot_version": int(snapshot_version),
                "payload_fingerprint": fingerprint(payload, "artifact-payload-v1"),
                "payload": payload,
            }
            artifact_path = temp_dir / "artifact.json"
            atomic_write_bytes(artifact_path, (canonical_json(artifact) + "\n").encode("utf-8"))
            try:
                os.rename(temp_dir, path)
            except FileExistsError:
                existing = self.get(key)
                if existing is None:
                    fail(E_CACHE_CORRUPT, "cache.artifact", "concurrent artifact publication failed")
                return existing, True
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        loaded = self.get(key)
        if loaded is None:
            fail(E_CACHE_CORRUPT, "cache.artifact", "published artifact disappeared", detail={"key": key})
        return loaded, False

    def count(self) -> int:
        return sum(1 for _ in self.objects.glob("*/*/artifact.json"))


class SnapshotStore:
    """Whole-snapshot publication with version-change detection."""

    def __init__(self, root: str | os.PathLike[str], artifact_version: int) -> None:
        self.root = Path(root)
        self.snapshots = self.root / "snapshots"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.current_path = self.root / "CURRENT"
        self.artifact_version = int(artifact_version)

    def _read_current_name(self) -> str | None:
        try:
            value = self.current_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if not value:
            return None
        if "/" in value or value.startswith("."):
            fail(E_SNAPSHOT_CORRUPT, "snapshot.CURRENT", "invalid current snapshot pointer", detail={"value": value})
        return value

    def load_current(self) -> dict[str, Any] | None:
        current_name = self._read_current_name()
        if current_name is None:
            return None
        snapshot_path = self.snapshots / current_name / "snapshot.json"
        if not snapshot_path.exists():
            fail(
                E_SNAPSHOT_CORRUPT,
                "snapshot.CURRENT",
                "CURRENT points to a missing snapshot",
                detail={"current": current_name},
            )
        try:
            manifest = load_json(snapshot_path, "snapshot.manifest")
        except Exception:
            raise
        if not isinstance(manifest, dict):
            fail(E_SNAPSHOT_CORRUPT, "snapshot.manifest", "snapshot manifest must be an object")
        state = manifest.get("state")
        computed = fingerprint(state, "snapshot-state-v1")
        if computed != manifest.get("state_digest"):
            fail(
                E_SNAPSHOT_CORRUPT,
                "snapshot.state_digest",
                "snapshot state digest mismatch",
                detail={"expected": manifest.get("state_digest"), "actual": computed},
            )
        return manifest

    def current_version(self) -> int:
        manifest = self.load_current()
        return 0 if manifest is None else int(manifest["commit_version"])

    def commit(self, state: dict[str, Any]) -> tuple[int, bool, dict[str, Any]]:
        state_digest = fingerprint(state, "snapshot-state-v1")
        current = self.load_current()
        if current is not None and current.get("state_digest") == state_digest:
            return int(current["commit_version"]), False, current

        commit_version = 1 if current is None else int(current["commit_version"]) + 1
        snapshot_name = f"v{commit_version:06d}"
        final_dir = self.snapshots / snapshot_name
        if final_dir.exists():
            fail(
                E_SNAPSHOT_CORRUPT,
                "snapshot.commit_version",
                "snapshot commit path already exists",
                detail={"path": str(final_dir)},
            )
        temp_dir = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=str(self.snapshots)))
        try:
            manifest = {
                "format": "renderdiag-snapshot-v1",
                "artifact_version": self.artifact_version,
                "commit_version": commit_version,
                "state_digest": state_digest,
                "state": state,
            }
            manifest_path = temp_dir / "snapshot.json"
            atomic_write_bytes(
                manifest_path,
                (canonical_json(manifest) + "\n").encode("utf-8"),
            )
            os.rename(temp_dir, final_dir)
            atomic_write_bytes(self.current_path, (snapshot_name + "\n").encode("utf-8"))
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        loaded = self.load_current()
        if loaded is None or int(loaded.get("commit_version", -1)) != commit_version:
            fail(E_SNAPSHOT_CORRUPT, "snapshot.commit", "snapshot did not become current")
        return commit_version, True, loaded

    def versions(self) -> list[int]:
        result: list[int] = []
        for path in sorted(self.snapshots.glob("v*/snapshot.json")):
            try:
                manifest = load_json(path, "snapshot.manifest")
                result.append(int(manifest["commit_version"]))
            except Exception:
                continue
        return result
