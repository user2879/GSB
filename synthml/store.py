"""Content-addressed, staged snapshot storage with atomic publication."""

from __future__ import annotations

import copy
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import SnapshotMismatchError
from .util import atomic_write_bytes, atomic_write_json, read_json, sha256_bytes, stable_hash


STAGES = ("load", "feature", "score", "decision", "metric")
SNAPSHOT_RETENTION = 5


@dataclass(frozen=True)
class CurrentSnapshot:
    name: str
    version: int
    path: Path
    manifest: dict[str, Any]


def _link_or_copy(source: str, target: str) -> str:
    try:
        os.link(source, target)
        return target
    except OSError:
        return shutil.copy2(source, target)


def copy_tree_linked(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for root, directories, files in os.walk(source):
        root_path = Path(root)
        relative = root_path.relative_to(source)
        destination_root = target / relative
        destination_root.mkdir(parents=True, exist_ok=True)
        for directory in sorted(directories):
            (destination_root / directory).mkdir(parents=True, exist_ok=True)
        for filename in sorted(files):
            source_file = root_path / filename
            destination_file = destination_root / filename
            if destination_file.exists():
                continue
            _link_or_copy(str(source_file), str(destination_file))


class SnapshotStore:
    def __init__(self, root: Path, schema_version: int) -> None:
        self.root = root
        self.schema_version = schema_version
        self.snapshots_dir = root / "snapshots"
        self.current_path = root / "CURRENT"

    def _snapshot_name(self, version: int) -> str:
        return f"v{version:06d}"

    def _versioned_snapshot_dirs(self) -> list[tuple[int, Path]]:
        if not self.snapshots_dir.exists():
            return []
        result: list[tuple[int, Path]] = []
        for path in self.snapshots_dir.iterdir():
            if path.is_dir() and path.name.startswith("v") and path.name[1:].isdigit():
                result.append((int(path.name[1:]), path))
        return sorted(result)

    def max_version(self) -> int:
        versions = self._versioned_snapshot_dirs()
        return versions[-1][0] if versions else 0

    def prune(self, keep: int = SNAPSHOT_RETENTION) -> list[str]:
        versions = self._versioned_snapshot_dirs()
        if len(versions) <= keep:
            return []
        current_name = None
        if self.current_path.exists():
            current_name = self.current_path.read_text(encoding="utf-8").strip()
        retained = {path.name for _version, path in versions[-keep:]}
        if current_name:
            retained.add(current_name)
        removed: list[str] = []
        for _version, path in versions:
            if path.name in retained:
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
        return removed

    def load_current(self, *, ignore_schema: bool = False) -> CurrentSnapshot | None:
        if not self.current_path.exists():
            return None
        try:
            name = self.current_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SnapshotMismatchError(f"cannot read snapshot pointer: {exc}") from exc
        if not name:
            raise SnapshotMismatchError("snapshot pointer is empty")
        path = self.snapshots_dir / name
        manifest_path = path / "manifest.json"
        if not path.is_dir() or not manifest_path.exists():
            raise SnapshotMismatchError(f"snapshot {name} is missing or incomplete")
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError) as exc:
            raise SnapshotMismatchError(f"cannot read snapshot manifest {name}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise SnapshotMismatchError(f"snapshot manifest {name} is not an object")
        if not ignore_schema and manifest.get("schema_version") != self.schema_version:
            raise SnapshotMismatchError(
                "snapshot schema version mismatch",
                {
                    "snapshot_schema_version": manifest.get("schema_version"),
                    "expected_schema_version": self.schema_version,
                    "snapshot": name,
                },
            )
        try:
            version = int(manifest["snapshot_version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotMismatchError(f"snapshot manifest {name} has an invalid version") from exc
        return CurrentSnapshot(name=name, version=version, path=path, manifest=manifest)

    def begin(
        self,
        *,
        prior: CurrentSnapshot | None,
        reuse: bool,
        higher_version: int | None = None,
    ) -> "SnapshotSession":
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        staging = self.root / f".staging-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        artifacts: dict[str, dict[str, Any]] = {}
        prior_keys: set[str] = set()
        if reuse and prior is not None:
            copy_tree_linked(prior.path, staging)
            copied = prior.manifest.get("artifacts", {})
            if isinstance(copied, dict):
                for key, record in copied.items():
                    if isinstance(record, dict):
                        artifact_path = staging / str(record.get("path", ""))
                        if artifact_path.is_file():
                            artifacts[str(key)] = copy.deepcopy(record)
                            prior_keys.add(str(key))
            report_path = staging / "report.json"
            if report_path.exists():
                report_path.unlink()
        start_version = max(self.max_version(), prior.version if prior else 0, higher_version or 0) + 1
        return SnapshotSession(
            store=self,
            staging=staging,
            next_version=start_version,
            artifacts=artifacts,
            prior_keys=prior_keys,
        )


class SnapshotSession:
    def __init__(
        self,
        *,
        store: SnapshotStore,
        staging: Path,
        next_version: int,
        artifacts: dict[str, dict[str, Any]],
        prior_keys: set[str],
    ) -> None:
        self.store = store
        self.staging = staging
        self.next_version = next_version
        self.artifacts = artifacts
        self.prior_keys = prior_keys
        self.audit_entries: list[dict[str, Any]] = []
        self._closed = False

    def artifact_path(self, key: str) -> Path:
        record = self.artifacts.get(key)
        if not record:
            return self.staging / "__missing__"
        return self.staging / str(record["path"])

    def sidecar_path(self, key: str) -> Path:
        path = self.artifact_path(key)
        return path.with_suffix(".bin")

    def has_artifact(self, key: str, *, require_sidecar: bool = False) -> bool:
        if key not in self.artifacts:
            return False
        if not self.artifact_path(key).is_file():
            return False
        if require_sidecar and not self.sidecar_path(key).is_file():
            return False
        return True

    def was_prior(self, key: str) -> bool:
        return key in self.prior_keys

    def read_artifact(self, key: str) -> dict[str, Any]:
        path = self.artifact_path(key)
        try:
            artifact = read_json(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read cached artifact {key}: {exc}") from exc
        if not isinstance(artifact, dict) or artifact.get("cache_key") != key:
            raise ValueError(f"cached artifact {key} is malformed")
        return artifact

    def read_sidecar(self, key: str) -> bytes:
        return self.sidecar_path(key).read_bytes()

    def drop_artifact(self, key: str) -> None:
        record = self.artifacts.pop(key, None)
        self.prior_keys.discard(key)
        if not record:
            return
        path = self.staging / str(record.get("path", ""))
        for candidate in (path, path.with_suffix(".bin")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass

    def write_artifact(
        self,
        *,
        stage: str,
        key: str,
        tier: str | None,
        sample: str | None,
        classifier: str | None,
        payload: dict[str, Any],
        metadata: dict[str, Any],
        input_refs: list[str],
        sidecar: bytes | None = None,
    ) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        if self.has_artifact(key):
            return self.artifacts[key]
        relative = Path("stages") / stage / f"{key}.json"
        path = self.staging / relative
        sidecar_hash = None
        if sidecar is not None:
            sidecar_hash = sha256_bytes(sidecar)
            atomic_write_bytes(path.with_suffix(".bin"), sidecar)
        artifact = {
            "cache_key": key,
            "stage": stage,
            "tier": tier,
            "sample": sample,
            "classifier": classifier,
            "input_refs": list(input_refs),
            "metadata": metadata,
            "payload": payload,
            "sidecar_sha256": sidecar_hash,
        }
        artifact_version = stable_hash({"artifact": artifact, "sidecar": sidecar_hash})
        artifact["artifact_version"] = artifact_version
        atomic_write_json(path, artifact)
        record = {
            "key": key,
            "stage": stage,
            "tier": tier,
            "sample": sample,
            "classifier": classifier,
            "path": relative.as_posix(),
            "input_refs": list(input_refs),
            "artifact_version": artifact_version,
            "sidecar": sidecar_hash is not None,
        }
        self.artifacts[key] = record
        return record

    def audit(
        self,
        *,
        stage: str,
        key: str,
        status: str,
        tier: str | None,
        sample: str | None,
        classifier: str | None,
        reason: str,
        input_refs: list[str] | None = None,
    ) -> None:
        self.audit_entries.append(
            {
                "stage": stage,
                "cache_key": key,
                "status": status,
                "tier": tier,
                "sample": sample,
                "classifier": classifier,
                "reason": reason,
                "input_refs": list(input_refs or []),
            }
        )

    def cache_hit_counts(self) -> dict[str, int]:
        counts = {stage: 0 for stage in STAGES}
        for entry in self.audit_entries:
            if entry["status"] == "reused":
                counts[entry["stage"]] = counts.get(entry["stage"], 0) + 1
        return counts

    def audit_sorted(self) -> list[dict[str, Any]]:
        stage_order = {stage: index for index, stage in enumerate(STAGES)}
        return sorted(
            self.audit_entries,
            key=lambda entry: (
                entry.get("tier") or "",
                stage_order.get(entry["stage"], 99),
                entry.get("classifier") or "",
                entry.get("sample") or "",
                entry["cache_key"],
            ),
        )

    def invalidations_sorted(self) -> list[dict[str, Any]]:
        return [
            entry
            for entry in self.audit_sorted()
            if entry["status"] in {"recomputed", "failed"}
        ]

    def stage_versions(self, active_keys: set[str] | None = None) -> dict[str, str]:
        versions: dict[str, str] = {}
        keys = active_keys if active_keys is not None else set(self.artifacts)
        for stage in STAGES:
            stage_records = [
                {
                    "key": key,
                    "version": self.artifacts[key]["artifact_version"],
                }
                for key in sorted(keys)
                if key in self.artifacts and self.artifacts[key]["stage"] == stage
            ]
            versions[stage] = stable_hash({"stage": stage, "artifacts": stage_records})
        return versions

    def write_report(self, report: dict[str, Any]) -> str:
        path = self.staging / "report.json"
        atomic_write_json(path, report)
        return sha256_bytes(path.read_bytes())

    def commit(
        self,
        *,
        config_hash: str,
        plan_hash: str,
        completed_cursor: dict[str, str],
        tiers: list[dict[str, Any]],
        in_progress: bool,
        active_keys: set[str] | None = None,
        report_sha256: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        version = self.next_version
        name = self.store._snapshot_name(version)
        self.next_version += 1
        stage_versions = self.stage_versions(active_keys)
        global_version = f"v{version:06d}-{config_hash[:12]}"
        manifest = {
            "schema_version": self.store.schema_version,
            "snapshot_version": version,
            "global_version": global_version,
            "config_hash": config_hash,
            "plan_hash": plan_hash,
            "stage_versions": stage_versions,
            "completed_cursor": dict(completed_cursor),
            "in_progress": bool(in_progress),
            "tiers": copy.deepcopy(tiers),
            "artifacts": copy.deepcopy(self.artifacts),
            "report_sha256": report_sha256,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "extra": copy.deepcopy(extra or {}),
        }
        atomic_write_json(self.staging / "manifest.json", manifest)
        destination = self.store.snapshots_dir / name
        if destination.exists():
            raise SnapshotMismatchError(f"snapshot destination already exists: {name}")
        copy_tree_linked(self.staging, destination)
        temporary_pointer = self.store.current_path.with_name(
            f"CURRENT.{uuid.uuid4().hex}.tmp"
        )
        atomic_write_bytes(temporary_pointer, (name + "\n").encode("utf-8"))
        os.replace(temporary_pointer, self.store.current_path)
        return {
            "snapshot_version": version,
            "snapshot_name": name,
            "global_version": global_version,
            "stage_versions": stage_versions,
            "in_progress": bool(in_progress),
            "report_sha256": report_sha256,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self.staging, ignore_errors=True)
