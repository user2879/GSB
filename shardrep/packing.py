"""Pack task construction, artifact versioning and atomic publication."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import atomic_write_text, safe_artifact_name, sha256_object
from .models import Spec


@dataclass(frozen=True)
class PackTask:
    task_id: str
    node_id: str
    logical_id: str
    seq: int
    bytes: int
    timeout_ms: int
    simulate_ms: int
    max_retries: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "logical_id": self.logical_id,
            "seq": self.seq,
            "bytes": self.bytes,
            "timeout_ms": self.timeout_ms,
            "simulate_ms": self.simulate_ms,
            "max_retries": self.max_retries,
        }


@dataclass(frozen=True)
class PreparedArtifact:
    task_id: str
    node_id: str
    logical_id: str
    seq: int
    byte_count: int
    expected_digest: str
    content_digest: str
    version: int
    staging_path: Path
    final_path: Path
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "logical_id": self.logical_id,
            "seq": self.seq,
            "bytes": self.byte_count,
            "expected_digest": self.expected_digest,
            "content_digest": self.content_digest,
            "version": self.version,
            "artifact_path": str(self.final_path),
        }


def build_pack_tasks(
    spec: Spec,
    logical_ids: Iterable[str],
    *,
    timeout_ms: int = 2_000,
    simulate_ms: int = 0,
    max_retries: int = 1,
    quotas: dict[str, int] | None = None,
) -> tuple[PackTask, ...]:
    """Create one deterministic task per affected logical shard.

    Tasks are distributed by least assigned load, then node id, among nodes that
    hold a replica and have a positive quota. This makes node quotas observable
    without giving the scheduler freedom to mutate task identity.
    """

    target_set = set(logical_ids)
    ordered = [logical_id for logical_id in spec.topo_order if logical_id in target_set]
    quota_map = quotas if quotas is not None else {node.node_id: node.pack_quota for node in spec.nodes}
    load = {node.node_id: 0 for node in spec.nodes}
    tasks: list[PackTask] = []
    for logical_id in ordered:
        logical = spec.logical_shards[logical_id]
        candidates = [
            node.node_id
            for node in spec.nodes
            if quota_map.get(node.node_id, 0) > 0
            and any(shard.logical_id == logical_id for shard in node.shards)
        ]
        if not candidates:
            candidates = [node.node_id for node in spec.nodes]
        candidates.sort(key=lambda node_id: (load[node_id], node_id))
        chosen = candidates[0]
        load[chosen] += 1
        tasks.append(
            PackTask(
                task_id=f"pack:{logical_id}",
                node_id=chosen,
                logical_id=logical_id,
                seq=logical.seq,
                bytes=logical.byte_count,
                timeout_ms=timeout_ms,
                simulate_ms=simulate_ms,
                max_retries=max_retries,
            )
        )
    return tuple(sorted(tasks, key=lambda task: (task.seq, task.logical_id)))


class PackStore:
    """One canonical artifact file per logical shard; unaffected files are untouched."""

    def __init__(self, root: Path, spec: Spec) -> None:
        self.root = root
        self.spec = spec
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".staging").mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.RLock()
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            try:
                raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            if isinstance(raw, dict):
                artifacts = raw.get("artifacts", {})
                if isinstance(artifacts, dict):
                    return {"format_version": 1, "artifacts": artifacts}
        return {"format_version": 1, "artifacts": {}}

    def _payload(self, logical_id: str) -> dict[str, Any]:
        logical = self.spec.logical_shards[logical_id]
        return {
            "format": "shardrep-pack-v1",
            "group_id": self.spec.group_id,
            "group_version": self.spec.group_version,
            "logical_id": logical_id,
            "seq": logical.seq,
            "bytes": logical.byte_count,
            "expected_digest": logical.digest,
        }

    def content_digest(self, logical_id: str) -> str:
        return sha256_object(self._payload(logical_id))

    def artifact_info(self, logical_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._manifest["artifacts"].get(logical_id)
            return dict(value) if isinstance(value, dict) else None

    def next_version(self, logical_id: str) -> int:
        with self._lock:
            value = self._manifest["artifacts"].get(logical_id, {})
            try:
                return int(value.get("version", 0)) + 1
            except (TypeError, ValueError):
                return 1

    def prepare(self, task: PackTask) -> PreparedArtifact:
        logical = self.spec.logical_shards[task.logical_id]
        content_digest = self.content_digest(task.logical_id)
        with self._lock:
            version = self.next_version(task.logical_id)
            payload = {
                **self._payload(task.logical_id),
                "content_digest": content_digest,
                "version": version,
                "packed_by_node": task.node_id,
            }
            safe_task = safe_artifact_name(task.task_id.replace(":", "_"))
            staging_path = self.root / ".staging" / safe_task
            final_path = self.root / safe_artifact_name(task.logical_id)
            atomic_write_text(staging_path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            return PreparedArtifact(
                task_id=task.task_id,
                node_id=task.node_id,
                logical_id=task.logical_id,
                seq=logical.seq,
                byte_count=logical.byte_count,
                expected_digest=logical.digest,
                content_digest=content_digest,
                version=version,
                staging_path=staging_path,
                final_path=final_path,
                payload=payload,
            )

    def publish(self, prepared: PreparedArtifact) -> dict[str, Any]:
        with self._lock:
            prepared.staging_path.replace(prepared.final_path)
            entry = {
                "logical_id": prepared.logical_id,
                "seq": prepared.seq,
                "bytes": prepared.byte_count,
                "expected_digest": prepared.expected_digest,
                "content_digest": prepared.content_digest,
                "version": prepared.version,
                "artifact": prepared.final_path.name,
            }
            self._manifest["artifacts"][prepared.logical_id] = entry
            self._write_manifest()
            return dict(entry)

    def promote_staged(self, task_id: str, logical_id: str) -> dict[str, Any] | None:
        """Recovery helper for a committed task whose publish was interrupted."""

        with self._lock:
            safe_task = safe_artifact_name(task_id.replace(":", "_"))
            staging_path = self.root / ".staging" / safe_task
            if not staging_path.exists():
                return None
            payload = json.loads(staging_path.read_text(encoding="utf-8"))
            final_path = self.root / safe_artifact_name(logical_id)
            staging_path.replace(final_path)
            entry = {
                "logical_id": logical_id,
                "seq": payload["seq"],
                "bytes": payload["bytes"],
                "expected_digest": payload["expected_digest"],
                "content_digest": payload["content_digest"],
                "version": payload["version"],
                "artifact": final_path.name,
            }
            self._manifest["artifacts"][logical_id] = entry
            self._write_manifest()
            return dict(entry)

    def repack_all_direct(self) -> dict[str, Any]:
        for logical_id in self.spec.topo_order:
            task = PackTask(
                task_id=f"full:{logical_id}",
                node_id=min(node.node_id for node in self.spec.nodes),
                logical_id=logical_id,
                seq=self.spec.logical_shards[logical_id].seq,
                bytes=self.spec.logical_shards[logical_id].byte_count,
                timeout_ms=1_000_000,
                simulate_ms=0,
                max_retries=0,
            )
            prepared = self.prepare(task)
            self.publish(prepared)
        return self.manifest()

    def _write_manifest(self) -> None:
        value = {
            "format": "shardrep-pack-manifest-v1",
            "format_version": 1,
            "group_id": self.spec.group_id,
            "group_version": self.spec.group_version,
            "artifacts": {key: self._manifest["artifacts"][key] for key in sorted(self._manifest["artifacts"])},
        }
        atomic_write_text(
            self.manifest_path,
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )

    def manifest(self) -> dict[str, Any]:
        with self._lock:
            return {
                "format": "shardrep-pack-manifest-v1",
                "format_version": 1,
                "group_id": self.spec.group_id,
                "group_version": self.spec.group_version,
                "artifacts": {
                    key: dict(value) for key, value in sorted(self._manifest["artifacts"].items())
                },
            }

    def artifact_file_digest(self, logical_id: str) -> str | None:
        with self._lock:
            entry = self._manifest["artifacts"].get(logical_id)
            if not isinstance(entry, dict):
                return None
            from .canonical import sha256_file

            path = self.root / str(entry.get("artifact", safe_artifact_name(logical_id)))
            return sha256_file(path) if path.exists() else None


def compare_artifact_sets(
    incremental: dict[str, Any],
    full: dict[str, Any],
    *,
    incremental_registry_digest: str,
    full_registry_digest: str,
) -> dict[str, Any]:
    from .errors import IncrementalMismatch

    left = incremental.get("artifacts", {})
    right = full.get("artifacts", {})
    for logical_id in sorted(set(left) | set(right)):
        if logical_id not in left:
            raise IncrementalMismatch(
                "incremental manifest is missing a shard present in the full rebuild",
                shard_id=logical_id,
                field="artifacts",
            )
        if logical_id not in right:
            raise IncrementalMismatch(
                "full rebuild is missing a shard present in the incremental result",
                shard_id=logical_id,
                field="artifacts",
            )
        for field in ("content_digest", "bytes", "expected_digest", "seq"):
            if left[logical_id].get(field) != right[logical_id].get(field):
                raise IncrementalMismatch(
                    f"incremental and full rebuild differ at {field}",
                    shard_id=logical_id,
                    field=field,
                    details={"incremental": left[logical_id].get(field), "full": right[logical_id].get(field)},
                )
    if incremental_registry_digest != full_registry_digest:
        raise IncrementalMismatch(
            "block replica digest differs between incremental and full rebuild",
            shard_id=None,
            field="registry_digest",
            details={
                "incremental": incremental_registry_digest,
                "full": full_registry_digest,
            },
        )
    return {
        "ok": True,
        "compared_shards": sorted(left),
        "incremental_registry_digest": incremental_registry_digest,
        "full_registry_digest": full_registry_digest,
    }

