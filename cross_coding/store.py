from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .constants import CODE_VERSION_CONFLICT
from .hashing import fingerprint, stable_json


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(stage: str, combo_fingerprint: str, strategy_fingerprint: str, profile: dict[str, Any]) -> str:
    return fingerprint(
        "cache-key",
        {
            "stage": stage,
            "combo_fingerprint": combo_fingerprint,
            "strategy_fingerprint": strategy_fingerprint,
            "width_profile": profile,
        },
    )


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.cache_root = self.root / "cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def path_for(self, stage: str, key: str) -> Path:
        safe_stage = stage.replace(":", "_").replace("/", "_")
        return self.cache_root / safe_stage / f"{key.replace(':', '_')}.json"

    def get(self, stage: str, key: str) -> dict[str, Any] | None:
        path = self.path_for(stage, key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if payload.get("cache_key") != key:
            return None
        digest = fingerprint("cached-artifact", payload.get("artifact"))
        if payload.get("artifact_digest") != digest:
            return None
        return payload.get("artifact")

    def put(self, stage: str, key: str, artifact: dict[str, Any]) -> None:
        path = self.path_for(stage, key)
        envelope = {
            "cache_key": key,
            "artifact_digest": fingerprint("cached-artifact", artifact),
            "artifact": artifact,
        }
        atomic_write_json(path, envelope)


class SnapshotVersionConflict(RuntimeError):
    code = CODE_VERSION_CONFLICT


class SnapshotStore:
    STAGES = ("enumeration", "coding", "cardinality", "adjudication", "pending_queue", "ledger")

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshot_root = self.root / "snapshots"
        self.manifest_path = self.root / "manifest.json"
        self.ledger_path = self.root / "ledger.jsonl"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.artifacts = ArtifactStore(self.root)

    def current_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def expected_version(self) -> int:
        manifest = self.current_manifest()
        return int(manifest["global_version"]) if manifest else 0

    def _verify_manifest(self, manifest: dict[str, Any]) -> bool:
        base = self.root
        if manifest.get("snapshot_dir"):
            candidate = Path(manifest["snapshot_dir"])
            snapshot_dir = candidate if candidate.is_absolute() else self.root / candidate
            if snapshot_dir.exists():
                base = snapshot_dir
        for stage, info in (manifest.get("stages") or {}).items():
            path_value = info.get("path")
            if not path_value:
                return False
            path = Path(path_value)
            if not path.is_absolute():
                path = base / path
            if not path.exists() or file_sha256(path) != info.get("sha256"):
                return False
        return True

    def recover(self) -> dict[str, Any]:
        current = self.current_manifest()
        if current and self._verify_manifest(current):
            return {"status": "current_valid", "global_version": current["global_version"]}

        candidates: list[tuple[int, Path]] = []
        for manifest_file in self.snapshot_root.glob("v*/manifest.json"):
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            version = int(manifest.get("global_version", 0))
            if self._verify_manifest({**manifest, "snapshot_dir": str(manifest_file.parent)}):
                candidates.append((version, manifest_file))
        if not candidates:
            return {"status": "fresh", "global_version": 0}

        version, manifest_file = max(candidates, key=lambda item: item[0])
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest["recovery_note"] = "root manifest was missing or stale; restored immutable snapshot"
        atomic_write_json(self.manifest_path, manifest)
        return {
            "status": "recovered_from_snapshot",
            "global_version": version,
            "manifest_path": str(self.manifest_path),
            "snapshot_dir": str(manifest_file.parent),
        }

    def commit(
        self,
        *,
        stage_payloads: dict[str, Any],
        policy_hash: str,
        width_profile: dict[str, Any],
        command_cursor: int,
        expected_version: int | None = None,
        extra_manifest: dict[str, Any] | None = None,
        ledger_entries: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        current = self.current_manifest()
        current_version = int(current["global_version"]) if current else 0
        if expected_version is not None and int(expected_version) != current_version:
            raise SnapshotVersionConflict(
                f"manifest version changed: expected {expected_version}, found {current_version}"
            )

        new_version = current_version + 1
        final_dir = self.snapshot_root / f"v{new_version:06d}"
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".v{new_version:06d}.tmp-", dir=str(self.snapshot_root))
        )
        try:
            stage_info: dict[str, Any] = {}
            for stage in self.STAGES:
                payload = stage_payloads.get(stage)
                if payload is None:
                    payload = {} if stage != "ledger" else {"entries": ledger_entries or []}
                relative_name = f"{stage}.json"
                target = temporary_dir / relative_name
                atomic_write_json(target, payload)
                stage_info[stage] = {
                    "path": relative_name,
                    "artifact_version": fingerprint("stage-version", payload),
                    "sha256": file_sha256(target),
                    "bytes": target.stat().st_size,
                }

            manifest: dict[str, Any] = {
                "global_version": new_version,
                "parent_version": current_version,
                "policy_config_hash": policy_hash,
                "width_profile": width_profile,
                "completed_command_cursor": int(command_cursor),
                "snapshot_commit_interval": (extra_manifest or {}).get("snapshot_commit_interval"),
                "stages": stage_info,
                "status": (extra_manifest or {}).get("status", "committed"),
            }
            if extra_manifest:
                for key, value in extra_manifest.items():
                    if key not in manifest:
                        manifest[key] = value
            manifest["manifest_digest"] = fingerprint("manifest", {k: v for k, v in manifest.items() if k != "manifest_digest"})
            atomic_write_json(temporary_dir / "manifest.json", manifest)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.rename(temporary_dir, final_dir)

            committed_manifest = dict(manifest)
            committed_manifest["snapshot_dir"] = os.path.relpath(final_dir, self.root)
            atomic_write_json(self.manifest_path, committed_manifest)

            if ledger_entries:
                with self.ledger_path.open("a", encoding="utf-8") as handle:
                    for entry in ledger_entries:
                        handle.write(stable_json(entry))
                        handle.write("\n")
            result_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            result_manifest["snapshot_path"] = str(final_dir)
            return result_manifest
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)


def load_previous_index(store: SnapshotStore | None) -> dict[str, Any]:
    if store is None:
        return {}
    manifest = store.current_manifest()
    if not manifest:
        return {}
    enum_info = (manifest.get("stages") or {}).get("enumeration") or {}
    relative = enum_info.get("path")
    if not relative:
        return {}
    enum_path = Path(relative)
    if not enum_path.is_absolute():
        snapshot_dir = manifest.get("snapshot_dir")
        base = store.root / snapshot_dir if snapshot_dir else store.root
        enum_path = base / relative
    if not enum_path.exists():
        return {}
    try:
        payload = json.loads(enum_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {item["canonical_key"]: item for item in payload.get("combinations", [])}
