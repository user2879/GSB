from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .errors import BusinessError, E_SNAPSHOT_VERSION


class CacheStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hit_count = 0
        self.miss_count = 0

    def path_for(self, key: str) -> Path:
        safe = "".join(ch for ch in key if ch.isalnum() or ch in {"_", "-"})
        return self.root / f"{safe}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.exists():
            self.miss_count += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.miss_count += 1
            return None
        self.hit_count += 1
        return payload

    def put(self, key: str, payload: dict[str, Any]) -> Path:
        path = self.path_for(key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(canonical_json({"key": key, "payload": payload}) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.versions = self.root / "versions"
        self.versions.mkdir(parents=True, exist_ok=True)
        self.pointer = self.root / "current.json"

    def current_metadata(self) -> dict[str, Any] | None:
        if not self.pointer.exists():
            return None
        return json.loads(self.pointer.read_text(encoding="utf-8"))

    def decide_restore(self, config_hash: str) -> dict[str, Any]:
        metadata = self.current_metadata()
        if metadata is None:
            return {"decision": "start", "reason": "no_snapshot", "config_hash": config_hash}
        if metadata.get("config_hash") == config_hash:
            return {
                "decision": "resume",
                "reason": "config_hash_match",
                "global_version": metadata.get("global_version"),
                "config_hash": config_hash,
                "snapshot_dir": metadata.get("snapshot_dir"),
            }
        return {
            "decision": "rebuild",
            "reason": "config_hash_changed",
            "stale_global_version": metadata.get("global_version"),
            "stale_config_hash": metadata.get("config_hash"),
            "config_hash": config_hash,
        }

    def commit(self, state: dict[str, Any], *, expected_global_version: int | None = None) -> dict[str, Any]:
        metadata = self.current_metadata()
        if (
            expected_global_version is not None
            and metadata is not None
            and metadata.get("global_version") != expected_global_version
        ):
            raise BusinessError(
                E_SNAPSHOT_VERSION,
                "snapshot global version changed before commit",
                field="global_version",
                details={
                    "expected": expected_global_version,
                    "current": metadata.get("global_version"),
                },
            )

        global_version = int(state["global_version"])
        config_hash = str(state["config_hash"])
        self.versions.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".tmp-v{global_version}-", dir=self.versions))
        try:
            artifacts = {
                "manifest.json": {
                    "global_version": global_version,
                    "config_hash": config_hash,
                    "format_version": state.get("snapshot_format_version", 1),
                },
                "documents.json": state.get("documents", []),
                "config.json": state.get("config", {}),
                "shard-receipts.json": state.get("receipts", {}),
                "ledger.json": state.get("ledger", {}),
                "pending.json": state.get("pending", []),
                "budget.json": state.get("budget", {}),
                "cache-manifest.json": state.get("cache_manifest", {}),
                "code-counts.json": state.get("code_counts", {}),
                "progress.json": state.get("progress", {}),
            }
            for name, value in artifacts.items():
                (staging / name).write_text(canonical_json(value) + "\n", encoding="utf-8")
            _fsync_directory(staging)
            unique_suffix = staging.name.rsplit("-", 1)[-1]
            final_dir = self.versions / (
                f"v{global_version}-{config_hash[:12]}-{unique_suffix}"
            )
            os.replace(staging, final_dir)
            _fsync_directory(self.versions)
            pointer_payload = {
                "global_version": global_version,
                "config_hash": config_hash,
                "snapshot_dir": str(final_dir),
                "files": sorted(artifacts),
            }
            pointer_tmp = self.pointer.with_suffix(".json.tmp")
            pointer_tmp.write_text(canonical_json(pointer_payload) + "\n", encoding="utf-8")
            os.replace(pointer_tmp, self.pointer)
            _fsync_directory(self.root)
            return pointer_payload
        finally:
            if staging.exists():
                import shutil

                shutil.rmtree(staging, ignore_errors=True)

    def load_state(self) -> dict[str, Any] | None:
        metadata = self.current_metadata()
        if metadata is None:
            return None
        directory = Path(metadata["snapshot_dir"])
        state: dict[str, Any] = {"__metadata__": metadata}
        for path in sorted(directory.glob("*.json")):
            state[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        return state
