from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .codes import E_SNAPSHOT_CORRUPT, E_VERSION_MISMATCH
from .utils import fingerprint, read_json, write_json


class ContentLedger:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.artifact_dir = self.root / "ledger" / "artifacts"
        self.snapshot_dir = self.root / "ledger" / "snapshots"
        self.event_path = self.root / "ledger" / "events.jsonl"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, cache_key: str) -> Path:
        return self.artifact_dir / f"{cache_key}.json"

    def has_artifact(self, cache_key: str) -> bool:
        path = self.artifact_path(cache_key)
        if not path.is_file():
            return False
        try:
            data = read_json(path)
            return isinstance(data, dict) and data.get("cache_key") == cache_key
        except Exception:
            return False

    def load_artifact(self, cache_key: str) -> dict[str, Any] | None:
        if not self.has_artifact(cache_key):
            return None
        return read_json(self.artifact_path(cache_key))

    def store_artifact(self, artifact: dict[str, Any]) -> str:
        cache_key = str(artifact["cache_key"])
        write_json(self.artifact_path(cache_key), artifact)
        return cache_key

    def append_event(self, event: dict[str, Any]) -> None:
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.event_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not self.event_path.exists():
            return []
        events = []
        with open(self.event_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(json.loads(line))
        return events


class SnapshotManager:
    def __init__(self, ledger: ContentLedger):
        self.ledger = ledger
        self.root = ledger.snapshot_dir
        self.current_path = self.root / "current.json"

    def load_current_pointer(self) -> dict[str, Any] | None:
        try:
            pointer = read_json(self.current_path)
            target = Path(str(pointer["path"]))
            if not target.is_absolute():
                target = self.root / target
            pointer["resolved_path"] = str(target)
            return pointer
        except Exception:
            return None

    def recover(self) -> dict[str, Any]:
        stale_staging = sorted(str(path) for path in self.root.glob(".staging-*") if path.is_dir())
        pointer = self.load_current_pointer()
        if pointer is None:
            return {
                "decision": "rebuild",
                "code": E_SNAPSHOT_CORRUPT,
                "reason": "missing or unreadable committed current.json",
                "stale_staging": stale_staging,
            }
        target = Path(pointer["resolved_path"])
        manifest_path = target / "manifest.json"
        state_path = target / "state.json"
        try:
            manifest = read_json(manifest_path)
            state = read_json(state_path)
            state_hash = fingerprint(state)
            if manifest.get("state_hash") != state_hash:
                return {
                    "decision": "rebuild",
                    "code": E_SNAPSHOT_CORRUPT,
                    "reason": "committed snapshot state hash mismatch",
                    "stale_staging": stale_staging,
                }
            if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
                return {
                    "decision": "rebuild",
                    "code": E_VERSION_MISMATCH,
                    "reason": "schema version changed",
                    "stale_staging": stale_staging,
                }
            return {
                "decision": "resume_with_snapshot",
                "code": "OK",
                "global_version": manifest.get("global_version"),
                "config_hash": manifest.get("config_hash"),
                "manifest": manifest,
                "state": state,
                "snapshot_path": str(target),
                "stale_staging": stale_staging,
            }
        except Exception as exc:
            return {
                "decision": "rebuild",
                "code": E_SNAPSHOT_CORRUPT,
                "reason": f"committed snapshot unreadable: {type(exc).__name__}: {exc}",
                "stale_staging": stale_staging,
            }

    def commit(self, state: dict[str, Any], config_hash: str) -> dict[str, Any]:
        recovery = self.recover()
        previous_version = int(recovery.get("global_version", 0)) if recovery["decision"] == "resume_with_snapshot" else 0
        global_version = previous_version + 1
        staging = Path(tempfile.mkdtemp(prefix=f".staging-v{global_version}-", dir=self.root))
        try:
            write_json(staging / "state.json", state)
            state_hash = fingerprint(state)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "global_version": global_version,
                "config_hash": config_hash,
                "state_hash": state_hash,
                "status": "committed",
                "pending_count": len(state.get("pending", [])),
                "business_code_counts": state.get("business_code_counts", {}),
            }
            write_json(staging / "manifest.json", manifest)
            target = self.root / f"snapshot-v{global_version:06d}-{config_hash[:12]}"
            if target.exists():
                shutil.rmtree(target)
            os_replace_dir(staging, target)
            pointer = {
                "global_version": global_version,
                "config_hash": config_hash,
                "path": target.name,
                "state_hash": state_hash,
            }
            tmp_pointer = self.root / f".current-{global_version}.tmp"
            write_json(tmp_pointer, pointer)
            tmp_pointer.replace(self.current_path)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)


def os_replace_dir(source: Path, target: Path) -> None:
    source.rename(target)
