"""Versioned repair-session snapshots, whole replacement and replay comparison."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .audit import utc_now
from .canonical import atomic_write_text, sha256_object
from .errors import ReplayError, SnapshotError
from .models import Spec
from .packing import PackStore
from .registry import InjectionRecord, ReplicaRecord, ReplicaRegistry
from .scheduler import ScheduleResult
from .spec import spec_manifest_digest


class SnapshotManager:
    def __init__(self, snapshot_path: Path) -> None:
        self.path = snapshot_path
        self.backup_dir = snapshot_path.parent / "backups"

    def save(
        self,
        *,
        spec: Spec,
        registry: ReplicaRegistry,
        pack_store: PackStore,
        schedule: ScheduleResult | dict[str, Any],
        repair_report: dict[str, Any],
        injections: list[InjectionRecord | dict[str, Any]],
        budget: dict[str, Any],
        spec_digest: str,
        replay: dict[str, Any] | None = None,
        block_replica_digest: str | None = None,
    ) -> dict[str, Any]:
        previous_version = 0
        if self.path.exists():
            try:
                old = json.loads(self.path.read_text(encoding="utf-8"))
                previous_version = int(old.get("snapshot_version", 0))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                previous_version = 0
            self._backup_existing(self.path, previous_version, suffix="before-overwrite")
        schedule_value = schedule.as_dict() if isinstance(schedule, ScheduleResult) else schedule
        injection_values = [
            injection.as_dict() if isinstance(injection, InjectionRecord) else dict(injection)
            for injection in injections
        ]
        payload: dict[str, Any] = {
            "format": "shardrep-snapshot",
            "format_version": 1,
            "snapshot_version": previous_version + 1,
            "saved_at": utc_now(),
            "group_id": spec.group_id,
            "group_version": spec.group_version,
            "mirror_graph_version": spec.mirror_graph_version,
            "spec_digest": spec_digest,
            "registry_digest": registry.state_digest(),
            "disk_digest": registry.disk_digest(),
            "registry": {"records": [record.as_dict() for record in registry.records]},
            "mirror_graph": spec.mirror_graph(),
            "pack_manifest": pack_store.manifest(),
            "schedule": schedule_value,
            "repair_report": repair_report,
            "injections": injection_values,
            "budget": dict(budget),
            "replay": dict(replay or {}),
            "block_replica_digest": block_replica_digest,
        }
        payload["content_digest"] = sha256_object(payload)
        atomic_write_text(
            self.path,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        return payload

    def load(self, *, verify: bool = True) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SnapshotError("snapshot file does not exist", field_path="$.path") from exc
        except json.JSONDecodeError as exc:
            raise SnapshotError(f"invalid snapshot JSON: {exc.msg}", field_path="$") from exc
        if not isinstance(payload, dict):
            raise SnapshotError("snapshot must be a JSON object", field_path="$")
        if payload.get("format") != "shardrep-snapshot":
            raise SnapshotError("unexpected snapshot format", field_path="format")
        format_version = payload.get("format_version")
        if not isinstance(format_version, int) or isinstance(format_version, bool) or format_version != 1:
            raise SnapshotError(
                "snapshot format_version must be integer 1",
                field_path="format_version",
                version=payload.get("snapshot_version") if isinstance(payload.get("snapshot_version"), int) else None,
            )
        snapshot_version = payload.get("snapshot_version")
        if not isinstance(snapshot_version, int) or isinstance(snapshot_version, bool) or snapshot_version < 1:
            raise SnapshotError("snapshot_version must be a positive integer", field_path="snapshot_version")
        expected = payload.get("content_digest")
        if not isinstance(expected, str):
            raise SnapshotError("content_digest is missing", field_path="content_digest", version=snapshot_version)
        unsigned = {key: value for key, value in payload.items() if key != "content_digest"}
        if verify and sha256_object(unsigned) != expected:
            raise SnapshotError(
                "snapshot content digest mismatch",
                field_path="content_digest",
                version=snapshot_version,
                details={"expected": expected, "actual": sha256_object(unsigned)},
            )
        return payload

    def restore(
        self,
        *,
        spec: Spec,
        registry: ReplicaRegistry,
        state_path: Path,
        logger: Any | None = None,
    ) -> dict[str, Any]:
        snapshot = self.load(verify=True)
        differences: list[dict[str, Any]] = []
        if snapshot.get("group_version") != spec.group_version:
            differences.append(
                {
                    "field": "group_version",
                    "snapshot": snapshot.get("group_version"),
                    "current": spec.group_version,
                }
            )
        if snapshot.get("mirror_graph_version") != spec.mirror_graph_version:
            differences.append(
                {
                    "field": "mirror_graph_version",
                    "snapshot": snapshot.get("mirror_graph_version"),
                    "current": spec.mirror_graph_version,
                }
            )
        current_spec_digest = spec_manifest_digest(spec)
        if snapshot.get("spec_digest") != current_spec_digest:
            differences.append(
                {
                    "field": "spec_digest",
                    "snapshot": snapshot.get("spec_digest"),
                    "current": current_spec_digest,
                }
            )
        if snapshot.get("disk_digest") != registry.disk_digest():
            differences.append(
                {
                    "field": "disk_digest",
                    "snapshot": snapshot.get("disk_digest"),
                    "current": registry.disk_digest(),
                }
            )

        replaced = bool(differences)
        backup_path: Path | None = None
        if replaced:
            backup_path = self._backup_existing(self.path, int(snapshot["snapshot_version"]), suffix="restore")
            self._replace_registry(spec, registry, snapshot)
            state_payload = {
                "format": "shardrep-restored-state-v1",
                "mode": "whole_replacement",
                "restored_at": utc_now(),
                "restored_from": str(self.path),
                "readonly_backup": str(backup_path),
                "snapshot_version": snapshot["snapshot_version"],
                "registry_digest": snapshot.get("registry_digest"),
                "mirror_graph_version": snapshot.get("mirror_graph_version"),
                "registry": snapshot.get("registry", {"records": []}),
                "mirror_graph": snapshot.get("mirror_graph", {}),
                "pack_manifest": snapshot.get("pack_manifest", {}),
                "schedule": snapshot.get("schedule", {}),
                "repair_report": snapshot.get("repair_report", {}),
                "injections": snapshot.get("injections", []),
                "budget": snapshot.get("budget", {}),
            }
            atomic_write_text(
                state_path,
                json.dumps(state_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            )
        result = {
            "code": "E_SNAPSHOT_OK",
            "snapshot_version": snapshot["snapshot_version"],
            "replaced": replaced,
            "differences": differences,
            "backup_path": str(backup_path) if backup_path else None,
            "registry_digest": registry.state_digest(),
            "state_path": str(state_path) if replaced else None,
        }
        if logger is not None:
            logger.emit(
                "E_SNAPSHOT_RESTORE",
                shard_id=None,
                version=snapshot["snapshot_version"],
                replaced=replaced,
                differences=differences,
                backup_path=result["backup_path"],
            )
        return result

    def apply_recorded_injections(
        self,
        snapshot: dict[str, Any],
        registry: ReplicaRegistry,
    ) -> list[InjectionRecord]:
        applied: list[InjectionRecord] = []
        for expected in sorted(snapshot.get("injections", []), key=lambda item: int(item.get("sequence", 0))):
            record = registry.inject(
                shard_id=expected["shard_id"],
                node_id=expected.get("node_id"),
                kind=expected["kind"],
                offset=int(expected.get("offset") or 0),
                keep_bytes=expected.get("keep_bytes"),
                replacement_hex=expected.get("replacement_hex"),
                sequence=int(expected.get("sequence", len(applied))),
            )
            if record.after_digest != expected.get("after_digest"):
                raise ReplayError(
                    "recorded injection did not reproduce its digest",
                    shard_id=expected["shard_id"],
                    field_path="injections.after_digest",
                    details={"expected": expected.get("after_digest"), "actual": record.after_digest},
                )
            applied.append(record)
        return applied

    def _backup_existing(self, path: Path, version: int, *, suffix: str) -> Path:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        candidate = self.backup_dir / f"session.v{version}.{suffix}.json"
        counter = 1
        while candidate.exists():
            candidate = self.backup_dir / f"session.v{version}.{suffix}.{counter}.json"
            counter += 1
        shutil.copy2(path, candidate)
        candidate.chmod(0o444)
        return candidate

    @staticmethod
    def _replace_registry(spec: Spec, registry: ReplicaRegistry, snapshot: dict[str, Any]) -> None:
        restored: list[ReplicaRecord] = []
        for value in snapshot.get("registry", {}).get("records", []):
            shard_id = value.get("shard_id")
            if shard_id not in spec.shard_by_id:
                continue
            shard = spec.shard_by_id[shard_id]
            restored.append(
                ReplicaRecord(
                    node_id=shard.node_id,
                    shard_id=shard.shard_id,
                    logical_id=shard.logical_id,
                    seq=shard.seq,
                    path=shard.path,
                    expected_bytes=int(value.get("expected_bytes", shard.byte_count)),
                    expected_digest=str(value.get("expected_digest", shard.digest)),
                    actual_bytes=value.get("actual_bytes"),
                    actual_digest=value.get("actual_digest"),
                    status=str(value.get("status", "missing")),
                    pending_repair=bool(value.get("pending_repair", False)),
                    injected_kind=value.get("injected_kind"),
                )
            )
        if restored:
            registry.records = tuple(
                sorted(restored, key=lambda item: (item.node_id, item.seq, item.logical_id, item.shard_id))
            )
            registry._reindex()


def compare_replay(snapshot: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    """Compare stable replay facts; monotonic wall-clock samples are intentionally excluded."""

    checks: list[tuple[str, Any, Any]] = [
        ("affected", snapshot.get("replay", {}).get("affected"), observed.get("affected")),
        (
            "pack_manifest.artifacts",
            _stable_pack_entries(snapshot.get("pack_manifest", {})),
            _stable_pack_entries(observed.get("pack_manifest", {})),
        ),
        (
            "schedule.entries",
            _stable_schedule(snapshot.get("schedule", {})),
            _stable_schedule(observed.get("schedule", {})),
        ),
        (
            "repair_report.counts",
            snapshot.get("repair_report", {}).get("counts"),
            observed.get("repair_report", {}).get("counts"),
        ),
        (
            "repair_report.categories",
            snapshot.get("repair_report", {}).get("categories"),
            observed.get("repair_report", {}).get("categories"),
        ),
        ("registry_digest", snapshot.get("registry_digest"), observed.get("registry_digest")),
        ("block_replica_digest", snapshot.get("block_replica_digest"), observed.get("block_replica_digest")),
    ]
    for field, expected, actual in checks:
        if expected != actual:
            shard_id = _first_difference_shard(expected, actual)
            raise ReplayError(
                f"snapshot replay differs at {field}",
                shard_id=shard_id,
                field_path=field,
                details={"expected": expected, "actual": actual},
            )
    return {"ok": True, "checked_fields": [field for field, _expected, _actual in checks]}


def _stable_pack_entries(manifest: dict[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts", {})
    stable: dict[str, Any] = {}
    for logical_id, entry in sorted(artifacts.items()):
        stable[logical_id] = {
            "seq": entry.get("seq"),
            "bytes": entry.get("bytes"),
            "expected_digest": entry.get("expected_digest"),
            "content_digest": entry.get("content_digest"),
            "version": entry.get("version"),
        }
    return stable


def _stable_schedule(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for entry in schedule.get("entries", []):
        values.append(
            {
                "logical_id": entry.get("logical_id"),
                "attempt": entry.get("attempt"),
                "node_id": entry.get("node_id"),
                "outcome": entry.get("outcome"),
                "completed_bytes": entry.get("completed_bytes"),
                "content_digest": entry.get("content_digest"),
                "version": entry.get("version"),
                "skip_reason": entry.get("skip_reason"),
            }
        )
    return sorted(values, key=lambda item: (item["logical_id"] or "", item["attempt"] or 0, item["node_id"] or ""))


def _first_difference_shard(expected: Any, actual: Any) -> str | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if expected.get(key) != actual.get(key):
                return key
    if isinstance(expected, list) and isinstance(actual, list):
        for left, right in zip(expected, actual):
            if left != right:
                if isinstance(left, dict):
                    return left.get("logical_id")
                if isinstance(right, dict):
                    return right.get("logical_id")
    return None


def tamper_snapshot_copy(source: Path, target: Path) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["content_digest"] = "sha256:" + "0" * 64
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return target
