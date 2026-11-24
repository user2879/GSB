from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .hashing import fingerprint


class SnapshotStore:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.root = self.output_dir / "snapshots"
        self.versions_dir = self.root / "versions"
        self.cache_dir = self.output_dir / "cache" / "stages"
        self.current_pointer = self.root / "CURRENT.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_current_pointer(self) -> dict[str, Any] | None:
        if not self.current_pointer.exists():
            return None
        return json.loads(self.current_pointer.read_text(encoding="utf-8"))

    def load_manifest(self) -> dict[str, Any] | None:
        pointer = self.load_current_pointer()
        if not pointer:
            return None
        manifest_path = Path(pointer["manifest_path"])
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def load_pending(self) -> list[dict[str, Any]]:
        manifest = self.load_manifest()
        if not manifest:
            return []
        pending_path = Path(manifest["pending_path"])
        if not pending_path.exists():
            return []
        return json.loads(pending_path.read_text(encoding="utf-8"))

    def recovery_decision(
        self, config_hash: str, sample_fingerprint: str | None = None
    ) -> dict[str, Any]:
        manifest = self.load_manifest()
        if manifest is None:
            return {"decision": "full_rebuild", "reason": "no_committed_snapshot", "manifest": None}
        if manifest.get("manifest_version") != 1:
            return {"decision": "full_rebuild", "reason": "unsupported_manifest_version", "manifest": manifest}
        if manifest.get("config_hash") != config_hash:
            return {
                "decision": "recompute_dag",
                "reason": "config_hash_changed_content_cache_still_eligible",
                "manifest": manifest,
            }
        if (
            sample_fingerprint is not None
            and manifest.get("sample_fingerprint") != sample_fingerprint
        ):
            return {
                "decision": "incremental_refresh",
                "reason": "sample_fingerprint_changed_stage_cache_evaluated_by_content_keys",
                "sample_fingerprint_match": False,
                "manifest": manifest,
            }
        return {
            "decision": "resume",
            "reason": "version_config_and_sample_fingerprints_match",
            "sample_fingerprint_match": True if sample_fingerprint is not None else None,
            "manifest": manifest,
        }

    def stage_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def load_stage_cache(self, cache_key: str) -> dict[str, Any] | None:
        path = self.stage_cache_path(cache_key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_stage_cache(self, cache_key: str, product: dict[str, Any]) -> Path:
        path = self.stage_cache_path(cache_key)
        atomic_write_json(path, product)
        return path

    def commit(
        self,
        manifest: dict[str, Any],
        products: dict[str, dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> dict[str, Any]:
        global_version = int(manifest.get("global_version", 1))
        temp_dir = Path(tempfile.mkdtemp(prefix=f".tmp-v{global_version:08d}-", dir=self.versions_dir))
        try:
            stage_dir = temp_dir / "stages"
            stage_dir.mkdir(parents=True, exist_ok=True)
            for stage_name, product in products.items():
                atomic_write_json(stage_dir / f"{stage_name}.json", product)
            pending_path = temp_dir / "pending.json"
            atomic_write_json(pending_path, pending)
            committed_manifest = dict(manifest)
            committed_manifest["manifest_version"] = 1
            final_dir = self.versions_dir / f"v{global_version:08d}"
            committed_manifest["pending_path"] = str((final_dir / "pending.json").resolve())
            committed_manifest["manifest_path"] = str((final_dir / "manifest.json").resolve())
            committed_manifest["pending_fingerprint"] = fingerprint(pending)
            manifest_path = temp_dir / "manifest.json"
            atomic_write_json(manifest_path, committed_manifest)

            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(temp_dir, final_dir)
            final_manifest = final_dir / "manifest.json"
            pointer = {
                "global_version": global_version,
                "manifest_path": str(final_manifest.resolve()),
                "committed_dir": str(final_dir.resolve()),
            }
            atomic_write_json(self.current_pointer, pointer)
            return pointer
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_report(output_dir: str | Path, name: str, payload: Any) -> Path:
    report_dir = Path(output_dir) / "reports"
    path = report_dir / name
    atomic_write_json(path, payload)
    return path
