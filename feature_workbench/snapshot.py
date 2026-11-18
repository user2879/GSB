from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .canonical import fingerprint


SCHEMA_VERSION = 1


class SnapshotError(RuntimeError):
    pass


class SnapshotMismatch(SnapshotError):
    pass


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _write_file(path: Path, payload: Any) -> str:
    data = _json_bytes(payload)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync; directory handles are not portable to Windows."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class SnapshotStore:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.manifest_path = state_dir / "manifest.json"

    def read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotMismatch(f"snapshot manifest is unreadable: {exc}") from exc

    def snapshot_path(self, manifest: dict[str, Any]) -> Path:
        relative = manifest.get("snapshot_dir")
        if not isinstance(relative, str) or not relative:
            raise SnapshotMismatch("snapshot manifest has no snapshot_dir")
        path = (self.state_dir / relative).resolve()
        root = self.state_dir.resolve()
        if root not in path.parents and path != root:
            raise SnapshotMismatch("snapshot_dir escapes the state directory")
        return path

    def load_stages(
        self, manifest: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        directory = self.snapshot_path(manifest)
        hashes = manifest.get("artifact_hashes", {})
        payloads: dict[str, Any] = {}
        for name in ("load", "features", "scores", "progress"):
            path = directory / f"{name}.json"
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise SnapshotMismatch(f"snapshot artifact {name} is missing: {exc}") from exc
            actual = hashlib.sha256(data).hexdigest()
            if hashes.get(name) != actual:
                raise SnapshotMismatch(f"snapshot artifact {name} hash mismatch")
            try:
                payloads[name] = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SnapshotMismatch(f"snapshot artifact {name} is corrupt: {exc}") from exc
        progress = payloads["progress"]
        if progress.get("global_version") != manifest.get("global_version"):
            raise SnapshotMismatch("snapshot progress global_version does not match manifest")
        if progress.get("applied_batches") != manifest.get("applied_batches"):
            raise SnapshotMismatch("snapshot progress applied_batches does not match manifest")
        if progress.get("event_cursor") != manifest.get("event_cursor"):
            raise SnapshotMismatch("snapshot progress event_cursor does not match manifest")
        if progress.get("config_hash") != manifest.get("config_hash"):
            raise SnapshotMismatch("snapshot progress config_hash does not match manifest")
        return payloads["load"], payloads["features"], payloads["scores"]

    def commit(
        self,
        *,
        global_version: int,
        config_hash: str,
        base_definitions_hash: str,
        base_events_hash: str,
        definitions_hash: str,
        plan_hash: str,
        event_cursor: int,
        applied_batches: int,
        load_stage: dict[str, Any],
        feature_stage: dict[str, Any],
        score_stage: dict[str, Any],
        status: str = "committed",
    ) -> dict[str, Any]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        stages_root = self.state_dir / "stages"
        stages_root.mkdir(parents=True, exist_ok=True)
        unique = uuid.uuid4().hex[:12]
        directory_name = f"v{global_version}-{unique}"
        target = stages_root / directory_name
        temporary = self.state_dir / f".tmp-{unique}"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            hashes: dict[str, str] = {}
            hashes["load"] = _write_file(temporary / "load.json", load_stage)
            hashes["features"] = _write_file(temporary / "features.json", feature_stage)
            hashes["scores"] = _write_file(temporary / "scores.json", score_stage)
            progress = {
                "status": status,
                "global_version": global_version,
                "applied_batches": applied_batches,
                "event_cursor": event_cursor,
                "config_hash": config_hash,
                "definitions_hash": definitions_hash,
                "plan_hash": plan_hash,
                "stage_versions": {
                    "load": global_version,
                    "features": global_version,
                    "scores": global_version,
                },
            }
            hashes["progress"] = _write_file(temporary / "progress.json", progress)
            os.replace(temporary, target)
            _fsync_directory(target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "global_version": global_version,
            "config_hash": config_hash,
            "base_definitions_hash": base_definitions_hash,
            "base_events_hash": base_events_hash,
            "definitions_hash": definitions_hash,
            "plan_hash": plan_hash,
            "event_cursor": event_cursor,
            "applied_batches": applied_batches,
            "snapshot_dir": f"stages/{directory_name}",
            "artifact_versions": {
                "load": global_version,
                "features": global_version,
                "scores": global_version,
                "progress": global_version,
            },
            "artifact_hashes": hashes,
            "status": status,
        }
        try:
            with (self.state_dir / "manifest.json.tmp").open("wb") as handle:
                data = _json_bytes(manifest)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(self.state_dir / "manifest.json.tmp", self.manifest_path)
            _fsync_directory(self.state_dir)
        except Exception:
            raise
        return manifest


def definitions_hash(raw_definitions: list[dict[str, Any]]) -> str:
    return fingerprint("definition-document-v1", {"features": raw_definitions})
