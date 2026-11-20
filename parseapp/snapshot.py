"""Whole-generation snapshots with an atomic current-version pointer.

Each commit materializes a complete generation directory, then atomically
replaces a small pointer file. A process killed before the pointer replacement
therefore leaves the previous complete generation as the recovery point.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


class SnapshotError(RuntimeError):
    code = "E_SNAPSHOT_INVALID"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    generation: str
    version: int
    manifest: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    path: Path


@dataclass(frozen=True)
class SnapshotLoad:
    snapshot: Snapshot | None
    warnings: list[dict[str, Any]]


class SnapshotStore:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.snapshot_root = self.state_dir / "snapshots"
        self.generations_dir = self.snapshot_root / "generations"
        self.current_path = self.snapshot_root / "CURRENT.json"
        self.generations_dir.mkdir(parents=True, exist_ok=True)

    def load_current(self) -> SnapshotLoad:
        warnings: list[dict[str, Any]] = []
        if not self.current_path.exists():
            return SnapshotLoad(snapshot=None, warnings=warnings)
        try:
            pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
            generation = pointer["generation"]
            expected_manifest_hash = pointer["manifest_sha256"]
            version = int(pointer["global_version"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            warnings.append(
                {
                    "code": "E_SNAPSHOT_POINTER_INVALID",
                    "message": f"ignoring current pointer: {exc}",
                }
            )
            return SnapshotLoad(snapshot=None, warnings=warnings)

        if (
            not isinstance(generation, str)
            or "/" in generation
            or "\\" in generation
            or generation.startswith(".")
        ):
            warnings.append(
                {"code": "E_SNAPSHOT_POINTER_INVALID", "message": "unsafe generation name"}
            )
            return SnapshotLoad(snapshot=None, warnings=warnings)
        generation_dir = self.generations_dir / generation
        manifest_path = generation_dir / "manifest.json"
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(
                {
                    "code": "E_SNAPSHOT_INCOMPLETE",
                    "message": f"generation {generation} has no valid manifest: {exc}",
                }
            )
            return SnapshotLoad(snapshot=None, warnings=warnings)
        if _sha256_bytes(manifest_bytes) != expected_manifest_hash:
            warnings.append(
                {
                    "code": "E_SNAPSHOT_MANIFEST_HASH",
                    "message": f"generation {generation} manifest hash mismatch",
                }
            )
            return SnapshotLoad(snapshot=None, warnings=warnings)
        if int(manifest.get("global_version", -1)) != version:
            warnings.append(
                {
                    "code": "E_SNAPSHOT_VERSION",
                    "message": "pointer and manifest versions disagree",
                }
            )
            return SnapshotLoad(snapshot=None, warnings=warnings)

        artifacts: dict[str, Any] = {}
        artifact_hashes = manifest.get("artifact_files", {})
        if not isinstance(artifact_hashes, Mapping):
            warnings.append(
                {"code": "E_SNAPSHOT_INCOMPLETE", "message": "artifact_files is invalid"}
            )
            return SnapshotLoad(snapshot=None, warnings=warnings)
        for relative, expected_hash in artifact_hashes.items():
            try:
                safe_relative = self._safe_relative(relative)
            except SnapshotError as exc:
                warnings.append({"code": exc.code, "message": str(exc)})
                return SnapshotLoad(snapshot=None, warnings=warnings)
            artifact_path = generation_dir / safe_relative
            try:
                payload = artifact_path.read_bytes()
            except OSError as exc:
                warnings.append(
                    {
                        "code": "E_SNAPSHOT_INCOMPLETE",
                        "message": f"missing artifact {relative}: {exc}",
                    }
                )
                return SnapshotLoad(snapshot=None, warnings=warnings)
            if _sha256_bytes(payload) != expected_hash:
                warnings.append(
                    {
                        "code": "E_SNAPSHOT_ARTIFACT_HASH",
                        "message": f"artifact hash mismatch: {relative}",
                    }
                )
                return SnapshotLoad(snapshot=None, warnings=warnings)
            try:
                artifacts[relative] = json.loads(payload)
            except json.JSONDecodeError as exc:
                warnings.append(
                    {
                        "code": "E_SNAPSHOT_ARTIFACT_JSON",
                        "message": f"artifact is not JSON: {relative}: {exc}",
                    }
                )
                return SnapshotLoad(snapshot=None, warnings=warnings)

        return SnapshotLoad(
            snapshot=Snapshot(
                generation=generation,
                version=version,
                manifest=manifest,
                artifacts=artifacts,
                path=generation_dir,
            ),
            warnings=warnings,
        )

    @staticmethod
    def _safe_relative(value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise SnapshotError("artifact path must be a non-empty string")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or path.parts[0] in {".", ""}:
            raise SnapshotError(f"unsafe artifact path: {value}")
        return str(path)

    def commit(
        self,
        manifest_payload: Mapping[str, Any],
        artifacts: Mapping[str, Any],
        expected_parent_version: int,
    ) -> Snapshot:
        current = self.load_current()
        current_snapshot = current.snapshot
        current_version = current_snapshot.version if current_snapshot else 0
        if current_version != expected_parent_version:
            raise SnapshotError(
                "E_SNAPSHOT_VERSION_CONFLICT: "
                f"expected parent {expected_parent_version}, found {current_version}"
            )
        version = current_version + 1
        generation = f"v{version:06d}"
        final_dir = self.generations_dir / generation
        if final_dir.exists():
            raise SnapshotError(f"generation already exists: {generation}")

        temp_dir = Path(
            tempfile.mkdtemp(prefix=f".tmp-{generation}-", dir=self.generations_dir)
        )
        try:
            artifact_files: dict[str, str] = {}
            for relative, payload in artifacts.items():
                safe_relative = self._safe_relative(relative)
                destination = temp_dir / safe_relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                encoded = _canonical_json(payload).encode("utf-8")
                destination.write_bytes(encoded)
                artifact_files[safe_relative] = _sha256_bytes(encoded)

            manifest = dict(manifest_payload)
            manifest["schema_version"] = "1.0"
            manifest["global_version"] = version
            manifest["parent_version"] = current_version
            manifest["artifact_files"] = artifact_files
            manifest_bytes = _canonical_json(manifest).encode("utf-8")
            (temp_dir / "manifest.json").write_bytes(manifest_bytes)

            # Generation directories are never modified in place. The final
            # rename publishes a complete, immutable tree.
            os.replace(temp_dir, final_dir)
            pointer_payload = {
                "generation": generation,
                "global_version": version,
                "manifest_sha256": _sha256_bytes(manifest_bytes),
                "parent_version": current_version,
            }
            self._atomic_replace_json(self.current_path, pointer_payload)
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        loaded = self.load_current()
        if loaded.snapshot is None or loaded.snapshot.version != version:
            raise SnapshotError("snapshot commit did not produce a readable generation")
        return loaded.snapshot

    @staticmethod
    def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(_canonical_json(payload).encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
