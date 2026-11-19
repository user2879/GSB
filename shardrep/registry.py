"""Physical replica registry, integrity checks and corruption injection."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .canonical import atomic_write_bytes, sha256_file, sha256_object
from .content import content_digest, deterministic_replacement
from .models import ShardDecl, Spec


@dataclass(frozen=True)
class ReplicaRecord:
    node_id: str
    shard_id: str
    logical_id: str
    seq: int
    path: Path
    expected_bytes: int
    expected_digest: str
    actual_bytes: int | None
    actual_digest: str | None
    status: str
    pending_repair: bool = False
    injected_kind: str | None = None

    @property
    def usable(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "shard_id": self.shard_id,
            "logical_id": self.logical_id,
            "seq": self.seq,
            "path": str(self.path),
            "expected_bytes": self.expected_bytes,
            "expected_digest": self.expected_digest,
            "actual_bytes": self.actual_bytes,
            "actual_digest": self.actual_digest,
            "status": self.status,
            "pending_repair": self.pending_repair,
            "injected_kind": self.injected_kind,
        }

    def disk_dict(self) -> dict[str, Any]:
        value = self.as_dict()
        value.pop("path", None)
        value.pop("pending_repair", None)
        value.pop("injected_kind", None)
        return value


@dataclass(frozen=True)
class InjectionRecord:
    sequence: int
    shard_id: str
    node_id: str
    logical_id: str
    kind: str
    before_bytes: int
    before_digest: str | None
    after_bytes: int | None
    after_digest: str | None
    offset: int | None = None
    keep_bytes: int | None = None
    replacement_hex: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "logical_id": self.logical_id,
            "kind": self.kind,
            "before_bytes": self.before_bytes,
            "before_digest": self.before_digest,
            "after_bytes": self.after_bytes,
            "after_digest": self.after_digest,
            "offset": self.offset,
            "keep_bytes": self.keep_bytes,
            "replacement_hex": self.replacement_hex,
        }


class ReplicaRegistry:
    def __init__(self, spec: Spec, records: Iterable[ReplicaRecord], logger: Any | None = None) -> None:
        self.spec = spec
        self.logger = logger
        self.records: tuple[ReplicaRecord, ...] = tuple(
            sorted(records, key=lambda item: (item.node_id, item.seq, item.logical_id, item.shard_id))
        )
        self._by_shard = {record.shard_id: record for record in self.records}
        self._by_logical: dict[str, list[ReplicaRecord]] = {logical_id: [] for logical_id in spec.logical_shards}
        for record in self.records:
            self._by_logical[record.logical_id].append(record)
        for values in self._by_logical.values():
            values.sort(key=lambda item: (item.node_id, item.shard_id))
        self._pending_by_shard: set[str] = {record.shard_id for record in self.records if record.pending_repair}
        self.integrity_issues: list[dict[str, Any]] = []

    @classmethod
    def load(
        cls,
        spec: Spec,
        *,
        logger: Any | None = None,
        injected_kinds: dict[str, str] | None = None,
    ) -> "ReplicaRegistry":
        injected_kinds = injected_kinds or {}
        records: list[ReplicaRecord] = []
        issues: list[dict[str, Any]] = []
        for node in spec.nodes:
            for shard in node.shards:
                record = _inspect_replica(shard, injected_kind=injected_kinds.get(shard.shard_id))
                records.append(record)
                if record.status != "ok":
                    issue = {
                        "code": "E_INTEGRITY",
                        "shard_id": shard.shard_id,
                        "node_id": node.node_id,
                        "status": record.status,
                        "expected_digest": record.expected_digest,
                        "actual_digest": record.actual_digest,
                        "expected_bytes": record.expected_bytes,
                        "actual_bytes": record.actual_bytes,
                    }
                    issues.append(issue)
        registry = cls(spec, records, logger)
        registry.integrity_issues = issues
        if logger is not None:
            for issue in issues:
                payload = {key: value for key, value in issue.items() if key != "code"}
                logger.emit("E_INTEGRITY", version=spec.group_version, **payload)
        return registry

    def by_shard(self, shard_id: str) -> ReplicaRecord:
        return self._by_shard[shard_id]

    def records_for_logical(self, logical_id: str) -> tuple[ReplicaRecord, ...]:
        return tuple(self._by_logical[logical_id])

    def invalid_records(self) -> tuple[ReplicaRecord, ...]:
        return tuple(record for record in self.records if record.status != "ok")

    def pending_records(self) -> tuple[ReplicaRecord, ...]:
        return tuple(record for record in self.records if record.pending_repair)

    def mark_affected(self, logical_ids: Iterable[str]) -> None:
        affected = set(logical_ids)
        self.records = tuple(
            replace(record, pending_repair=True) if record.logical_id in affected else record
            for record in self.records
        )
        self._reindex()

    def clear_pending(self, logical_ids: Iterable[str] | None = None) -> None:
        affected = set(logical_ids) if logical_ids is not None else None
        self.records = tuple(
            replace(record, pending_repair=False)
            if affected is None or record.logical_id in affected
            else record
            for record in self.records
        )
        self._reindex()

    def replace_record(self, record: ReplicaRecord) -> None:
        records = list(self.records)
        for index, existing in enumerate(records):
            if existing.shard_id == record.shard_id:
                records[index] = record
                break
        else:
            records.append(record)
        self.records = tuple(
            sorted(records, key=lambda item: (item.node_id, item.seq, item.logical_id, item.shard_id))
        )
        self._reindex()
        self.refresh_integrity_issues()

    def refresh_integrity_issues(self) -> None:
        self.integrity_issues = [
            {
                "code": "E_INTEGRITY",
                "shard_id": record.shard_id,
                "node_id": record.node_id,
                "status": record.status,
                "expected_digest": record.expected_digest,
                "actual_digest": record.actual_digest,
                "expected_bytes": record.expected_bytes,
                "actual_bytes": record.actual_bytes,
            }
            for record in self.records
            if record.status != "ok"
        ]

    def refresh(self, *, preserve_pending: bool = True) -> None:
        pending = {record.shard_id for record in self.records if record.pending_repair}
        injected = {record.shard_id: record.injected_kind for record in self.records if record.injected_kind}
        refreshed: list[ReplicaRecord] = []
        new_issues: list[dict[str, Any]] = []
        for record in self.records:
            current = _inspect_replica(self.spec.shard_by_id[record.shard_id], injected_kind=injected.get(record.shard_id))
            if preserve_pending and record.shard_id in pending:
                current = replace(current, pending_repair=True)
            refreshed.append(current)
            if current.status != "ok":
                issue = {
                    "code": "E_INTEGRITY",
                    "shard_id": current.shard_id,
                    "node_id": current.node_id,
                    "status": current.status,
                    "expected_digest": current.expected_digest,
                    "actual_digest": current.actual_digest,
                    "expected_bytes": current.expected_bytes,
                    "actual_bytes": current.actual_bytes,
                }
                new_issues.append(issue)
        self.records = tuple(refreshed)
        self.integrity_issues = new_issues
        self._reindex()

    def _reindex(self) -> None:
        self._by_shard = {record.shard_id: record for record in self.records}
        self._by_logical = {logical_id: [] for logical_id in self.spec.logical_shards}
        for record in self.records:
            self._by_logical[record.logical_id].append(record)
        for values in self._by_logical.values():
            values.sort(key=lambda item: (item.node_id, item.shard_id))

    def disk_digest(self) -> str:
        return sha256_object([record.disk_dict() for record in self.records])

    def state_digest(self) -> str:
        portable: list[dict[str, Any]] = []
        for record in self.records:
            value = record.as_dict()
            value.pop("path", None)
            portable.append(value)
        return sha256_object(portable)

    def summary(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for record in self.records:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
        return {
            "spec_digest": sha256_object(self.spec.mirror_graph() | {
                "group_id": self.spec.group_id,
                "group_version": self.spec.group_version,
                "nodes": [node.as_dict() for node in self.spec.nodes],
                "blocks": [block.as_dict() for block in self.spec.blocks],
            }),
            "disk_digest": self.disk_digest(),
            "state_digest": self.state_digest(),
            "status_counts": status_counts,
            "pending_repair": len(self.pending_records()),
            "integrity_issues": len(self.integrity_issues),
        }

    def inject(
        self,
        *,
        shard_id: str,
        kind: str,
        node_id: str | None = None,
        offset: int = 0,
        keep_bytes: int | None = None,
        replacement: bytes | None = None,
        replacement_hex: str | None = None,
        sequence: int = 0,
    ) -> InjectionRecord:
        if shard_id not in self._by_shard:
            raise KeyError(f"unknown shard_id {shard_id}")
        record = self._by_shard[shard_id]
        if node_id is not None and record.node_id != node_id:
            raise KeyError(f"shard {shard_id} is not on node {node_id}")
        if kind not in {"truncate", "flip", "replace", "delete"}:
            raise ValueError("kind must be truncate, flip, replace or delete")
        path = record.path
        original = path.read_bytes() if path.exists() else b""
        before_bytes = len(original)
        before_digest = content_digest(original) if path.exists() else None
        actual_offset: int | None = None
        actual_keep: int | None = None
        actual_replacement: bytes | None = None

        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "truncate":
            actual_keep = keep_bytes if keep_bytes is not None else max(0, before_bytes // 2)
            if actual_keep < 0 or actual_keep > before_bytes:
                raise ValueError("keep_bytes must be within the current shard length")
            atomic_write_bytes(path, original[:actual_keep])
        elif kind == "flip":
            if before_bytes == 0:
                data = b"\x01"
                actual_offset = 0
            else:
                actual_offset = offset % before_bytes
                mutable = bytearray(original)
                mutable[actual_offset] ^= 0xFF
                data = bytes(mutable)
            atomic_write_bytes(path, data)
        elif kind == "replace":
            if replacement is not None:
                actual_replacement = replacement
            elif replacement_hex is not None:
                actual_replacement = bytes.fromhex(replacement_hex)
            else:
                actual_replacement = deterministic_replacement(
                    self.spec.seed, shard_id, max(1, record.expected_bytes)
                )
            if actual_replacement == original:
                actual_replacement = bytes([actual_replacement[0] ^ 0xFF]) + actual_replacement[1:]
            atomic_write_bytes(path, actual_replacement)
        else:
            path.unlink(missing_ok=True)

        updated = _inspect_replica(self.spec.shard_by_id[shard_id], injected_kind=kind)
        updated = replace(updated, pending_repair=record.pending_repair)
        self.replace_record(updated)
        injection = InjectionRecord(
            sequence=sequence,
            shard_id=shard_id,
            node_id=record.node_id,
            logical_id=record.logical_id,
            kind=kind,
            before_bytes=before_bytes,
            before_digest=before_digest,
            after_bytes=updated.actual_bytes,
            after_digest=updated.actual_digest,
            offset=actual_offset,
            keep_bytes=actual_keep,
            replacement_hex=actual_replacement.hex() if actual_replacement is not None else replacement_hex,
        )
        if self.logger is not None:
            self.logger.emit(
                "E_INTEGRITY",
                shard_id=shard_id,
                version=self.spec.group_version,
                node_id=record.node_id,
                kind=kind,
                expected_digest=record.expected_digest,
                actual_digest=updated.actual_digest,
                expected_bytes=record.expected_bytes,
                actual_bytes=updated.actual_bytes,
            )
        return injection


def _inspect_replica(shard: ShardDecl, *, injected_kind: str | None = None) -> ReplicaRecord:
    if not shard.path.exists() or not shard.path.is_file():
        status = "missing"
        actual_bytes = None
        actual_digest = None
    else:
        try:
            actual_bytes = shard.path.stat().st_size
            actual_digest = sha256_file(shard.path)
        except OSError:
            status = "unreadable"
            actual_bytes = None
            actual_digest = None
        else:
            if actual_bytes < shard.byte_count:
                status = "truncated"
            elif actual_bytes != shard.byte_count:
                status = "length_mismatch"
            elif actual_digest != shard.digest:
                status = "digest_mismatch"
            else:
                status = "ok"
    return ReplicaRecord(
        node_id=shard.node_id,
        shard_id=shard.shard_id,
        logical_id=shard.logical_id,
        seq=shard.seq,
        path=shard.path,
        expected_bytes=shard.byte_count,
        expected_digest=shard.digest,
        actual_bytes=actual_bytes,
        actual_digest=actual_digest,
        status=status,
        injected_kind=injected_kind,
    )


def apply_repair_task(task: Any, registry: ReplicaRegistry) -> dict[str, Any]:
    """Copy a source replica into a target replica and refresh its record."""

    source = registry.by_shard(task.source_shard_id)
    target = registry.by_shard(task.target_shard_id)
    if source.status != "ok":
        return {"ok": False, "task_id": task.task_id, "reason": "source_not_usable"}
    try:
        data = source.path.read_bytes()
        if len(data) != task.bytes or content_digest(data) != task.expected_digest:
            return {"ok": False, "task_id": task.task_id, "reason": "source_changed"}
        atomic_write_bytes(target.path, data)
        updated = _inspect_replica(registry.spec.shard_by_id[target.shard_id])
        self_pending = target.pending_repair
        updated = replace(updated, pending_repair=self_pending)
        registry.replace_record(updated)
        return {
            "ok": updated.status == "ok",
            "task_id": task.task_id,
            "target_shard_id": target.shard_id,
            "status": updated.status,
            "digest": updated.actual_digest,
            "bytes": updated.actual_bytes,
        }
    except OSError as exc:
        return {"ok": False, "task_id": task.task_id, "reason": f"I/O error: {exc}"}


def remove_readonly(path: Path) -> None:
    if path.exists():
        os.chmod(path, 0o600)
