"""Staged, whole-snapshot publication and recovery diagnostics."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json, fingerprint, read_json, write_json
from .codes import (
    E_RECOVERY_CONFIG_CHANGED,
    E_RECOVERY_CURRENT,
    E_RECOVERY_MANIFEST_INVALID,
    E_RECOVERY_NO_SNAPSHOT,
    E_RECOVERY_ORPHAN_TEMP,
    E_RECOVERY_RESUME,
    E_RECOVERY_VERSION_CHANGED,
    E_SNAPSHOT_COMMIT,
    Issue,
)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass
class StagedSnapshot:
    staging_path: Path
    snapshot_id: str
    manifest: dict[str, Any]


@dataclass
class LoadedSnapshot:
    snapshot_id: str
    path: Path
    manifest: dict[str, Any]
    summaries: dict[str, dict[str, Any]]
    ledger: dict[str, Any]
    recall: list[dict[str, Any]]
    report: dict[str, Any]


class SnapshotStore:
    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.root = self.workdir / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True)
        self.pointer_path = self.root / "CURRENT.json"

    def _directory_snapshot_id(self, path: Path) -> str:
        return path.name

    def orphan_directories(self) -> list[str]:
        active_id = self.current_snapshot_id()
        active_version: int | None = None
        if active_id is not None:
            try:
                active_manifest = read_json(self.root / active_id / "manifest.json")
                if isinstance(active_manifest.get("global_version"), int):
                    active_version = int(active_manifest["global_version"])
            except (OSError, ValueError, TypeError, AttributeError):
                active_version = None
        orphans: list[str] = []
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            if path.name.startswith(".staging-"):
                orphans.append(path.name)
                continue
            if not path.name.startswith("snap-") or path.name == active_id:
                continue
            try:
                manifest = read_json(path / "manifest.json")
                version = manifest.get("global_version")
            except (OSError, ValueError, TypeError, AttributeError):
                orphans.append(path.name)
                continue
            if not isinstance(version, int) or active_version is None or version > active_version:
                orphans.append(path.name)
        return sorted(orphans)

    def current_snapshot_id(self) -> str | None:
        if not self.pointer_path.exists():
            return None
        try:
            raw = read_json(self.pointer_path)
        except (OSError, ValueError, TypeError):
            return None
        value = raw.get("snapshot_id") if isinstance(raw, dict) else None
        return str(value) if value else None

    def load_current(self) -> LoadedSnapshot | None:
        snapshot_id = self.current_snapshot_id()
        if snapshot_id is None:
            return None
        path = self.root / snapshot_id
        if not path.is_dir():
            return None
        try:
            manifest = read_json(path / "manifest.json")
            summaries: dict[str, dict[str, Any]] = {}
            for shard_id in manifest.get("shard_ids", []):
                summaries[str(shard_id)] = read_json(path / "shards" / f"{shard_id}.json")
            ledger = read_json(path / "ledger.json")
            recall = read_json(path / "recall.json")
            report = read_json(path / "report.json")
        except (OSError, ValueError, TypeError):
            return None
        return LoadedSnapshot(
            snapshot_id=snapshot_id,
            path=path,
            manifest=manifest,
            summaries=summaries,
            ledger=ledger,
            recall=recall,
            report=report,
        )

    def _next_snapshot_id(self, global_version: int, config_hash: str) -> str:
        base = f"snap-{global_version:08d}-{config_hash[:10]}"
        candidate = base
        counter = 1
        while (self.root / candidate).exists():
            counter += 1
            candidate = f"{base}-{counter:02d}"
        return candidate

    def stage(
        self,
        *,
        global_version: int,
        config_hash: str,
        manifest_fields: Mapping[str, Any],
        summaries: Mapping[str, Mapping[str, Any]],
        ledger: Mapping[str, Any],
        recall: list[dict[str, Any]],
        report: Mapping[str, Any],
    ) -> StagedSnapshot:
        snapshot_id = self._next_snapshot_id(global_version, config_hash)
        staging_name = f".staging-{snapshot_id}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        staging = self.root / staging_name
        staging.mkdir(parents=True, exist_ok=False)
        shard_ids = sorted(str(shard_id) for shard_id in summaries)
        shard_dir = staging / "shards"
        shard_dir.mkdir()
        written: list[Path] = []
        for shard_id in shard_ids:
            path = shard_dir / f"{shard_id}.json"
            write_json(path, summaries[shard_id])
            written.append(path)
        ledger_path = staging / "ledger.json"
        recall_path = staging / "recall.json"
        report_path = staging / "report.json"
        write_json(ledger_path, ledger)
        write_json(recall_path, recall)
        write_json(report_path, report)
        written.extend([ledger_path, recall_path, report_path])
        manifest = dict(manifest_fields)
        manifest.update(
            {
                "schema_version": 1,
                "snapshot_id": snapshot_id,
                "status": "prepared",
                "global_version": global_version,
                "config_hash": config_hash,
                "shard_ids": shard_ids,
                "files": {
                    "ledger": "ledger.json",
                    "recall": "recall.json",
                    "report": "report.json",
                    "shards": [f"shards/{shard_id}.json" for shard_id in shard_ids],
                },
                "content_fingerprint": fingerprint(
                    {
                        "summaries": summaries,
                        "ledger": ledger,
                        "recall": recall,
                    }
                ),
            }
        )
        manifest_path = staging / "manifest.json"
        write_json(manifest_path, manifest)
        written.append(manifest_path)
        for path in written:
            _fsync_file(path)
        _fsync_directory(staging)
        _fsync_directory(shard_dir)
        return StagedSnapshot(staging, snapshot_id, manifest)

    def commit(self, staged: StagedSnapshot) -> LoadedSnapshot:
        final_path = self.root / staged.snapshot_id
        try:
            os.replace(staged.staging_path, final_path)
            _fsync_directory(self.root)
            pointer_temp = self.root / f".CURRENT.json.tmp-{os.getpid()}"
            pointer_temp.write_text(
                canonical_json(
                    {
                        "snapshot_id": staged.snapshot_id,
                        "global_version": staged.manifest["global_version"],
                        "config_hash": staged.manifest["config_hash"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _fsync_file(pointer_temp)
            os.replace(pointer_temp, self.pointer_path)
            _fsync_directory(self.root)
        except OSError as exc:
            raise RuntimeError(f"{E_SNAPSHOT_COMMIT}: {exc}") from exc
        loaded = self.load_current()
        if loaded is None or loaded.snapshot_id != staged.snapshot_id:
            raise RuntimeError(f"{E_SNAPSHOT_COMMIT}: active snapshot could not be reloaded")
        return loaded

    def recovery_decision(
        self,
        *,
        requested_global_version: int,
        expected_config_hash: str,
        expected_document_fingerprint: str,
        mode: str,
    ) -> dict[str, Any]:
        orphan_dirs = self.orphan_directories()
        loaded = self.load_current()
        issues: list[dict[str, Any]] = []
        if orphan_dirs:
            issues.append(
                Issue(
                    code=E_RECOVERY_ORPHAN_TEMP,
                    message="uncommitted snapshot directories were found and ignored",
                    object_type="snapshot",
                    object_id="staging",
                    details={"directories": orphan_dirs},
                ).to_dict()
            )
        if loaded is None:
            issues.append(
                Issue(
                    code=E_RECOVERY_NO_SNAPSHOT,
                    message="no committed snapshot is available; rebuilding from blank",
                    object_type="snapshot",
                    object_id="current",
                ).to_dict()
            )
            return {
                "decision": "rebuild",
                "code": E_RECOVERY_NO_SNAPSHOT,
                "issues": issues,
                "orphan_directories": orphan_dirs,
                "active_snapshot": None,
                "requested_global_version": requested_global_version,
                "expected_config_hash": expected_config_hash,
            }
        manifest = loaded.manifest
        version = manifest.get("global_version")
        config_hash = manifest.get("config_hash")
        if not isinstance(version, int) or not isinstance(config_hash, str):
            issues.append(
                Issue(
                    code=E_RECOVERY_MANIFEST_INVALID,
                    message="active manifest lacks a valid version or config hash",
                    object_type="snapshot",
                    object_id=loaded.snapshot_id,
                ).to_dict()
            )
            decision = "rebuild"
        elif config_hash != expected_config_hash:
            issues.append(
                Issue(
                    code=E_RECOVERY_CONFIG_CHANGED,
                    message="configuration hash differs from the active snapshot",
                    object_type="snapshot",
                    object_id=loaded.snapshot_id,
                    details={"snapshot_config_hash": config_hash, "expected_config_hash": expected_config_hash},
                ).to_dict()
            )
            decision = "rebuild"
        elif mode == "full":
            issues.append(
                Issue(
                    code=E_RECOVERY_VERSION_CHANGED,
                    message="full mode deliberately rebuilds from blank despite an active snapshot",
                    object_type="snapshot",
                    object_id=loaded.snapshot_id,
                ).to_dict()
            )
            decision = "rebuild"
        elif requested_global_version != version:
            issues.append(
                Issue(
                    code=E_RECOVERY_VERSION_CHANGED,
                    message="requested global version differs; continuing from the active manifest",
                    object_type="snapshot",
                    object_id=loaded.snapshot_id,
                    details={"snapshot_global_version": version, "requested_global_version": requested_global_version},
                ).to_dict()
            )
            decision = "resume"
        elif manifest.get("document_fingerprint") != expected_document_fingerprint:
            issues.append(
                Issue(
                    code=E_RECOVERY_RESUME,
                    message="document content changed; resume from snapshot using fingerprint invalidation",
                    object_type="snapshot",
                    object_id=loaded.snapshot_id,
                ).to_dict()
            )
            decision = "resume"
        elif manifest.get("completed_doc_cursor", 0) < manifest.get("document_count", 0):
            issues.append(
                Issue(
                    code=E_RECOVERY_RESUME,
                    message="snapshot has pending documents; continue from its cursor",
                    object_type="snapshot",
                    object_id=loaded.snapshot_id,
                    details={
                        "completed_doc_cursor": manifest.get("completed_doc_cursor"),
                        "document_count": manifest.get("document_count"),
                    },
                ).to_dict()
            )
            decision = "resume"
        else:
            decision = "current"
        if decision == "current":
            issues.append(
                Issue(
                    code=E_RECOVERY_CURRENT,
                    message="active snapshot matches version and configuration hash",
                    object_type="snapshot",
                    object_id=loaded.snapshot_id,
                ).to_dict()
            )
        return {
            "decision": decision,
            "code": issues[-1]["code"] if issues else E_RECOVERY_CURRENT,
            "issues": issues,
            "orphan_directories": orphan_dirs,
            "active_snapshot": loaded.snapshot_id,
            "active_global_version": version,
            "active_config_hash": config_hash,
            "requested_global_version": requested_global_version,
            "expected_config_hash": expected_config_hash,
        }
