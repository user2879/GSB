"""Deterministic compilation of cluster/task/shard distribution plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote

from .errors import E_DIVERGE, E_MANIFEST, PipelineError
from .manifest import Manifest, Revision, revision_dependency_closure
from .util import canonical_json, digest_json


PLAN_FORMAT_VERSION = 1
SAMPLE_COUNT = 64
GROUP_COUNT = 8


def stable_shard_id(cluster_id: str, task_id: str, shard_index: int) -> str:
    # Percent-encoding keeps user identifiers from creating an ambiguous
    # composite id while preserving simple ids verbatim.
    return (
        f"{quote(cluster_id, safe='')}::"
        f"{quote(task_id, safe='')}::s{shard_index:06d}"
    )


@dataclass(frozen=True)
class ShardPlan:
    stable_id: str
    cluster_id: str
    task_id: str
    shard_index: int
    revision_version: int
    config_keys: tuple[str, ...]
    config_values: dict[str, int]
    config_relations: dict[str, tuple[str, ...]]
    resource_units: int
    task_concurrency: int
    cluster_seed: int
    cluster_fingerprint: str
    sample_count: int = SAMPLE_COUNT
    group_count: int = GROUP_COUNT

    @property
    def compile_payload(self) -> dict[str, Any]:
        return {
            "format_version": PLAN_FORMAT_VERSION,
            "stable_id": self.stable_id,
            "cluster_id": self.cluster_id,
            "cluster_seed": self.cluster_seed,
            "cluster_fingerprint": self.cluster_fingerprint,
            "task_id": self.task_id,
            "shard_index": self.shard_index,
            "revision_version": self.revision_version,
            "config_keys": list(self.config_keys),
            "config_values": {
                key: self.config_values[key] for key in self.config_keys
            },
            "config_relations": {
                key: list(self.config_relations[key]) for key in self.config_keys
            },
            "resource_units": self.resource_units,
            "task_concurrency": self.task_concurrency,
            "sample_count": self.sample_count,
            "group_count": self.group_count,
        }

    @property
    def compile_digest(self) -> str:
        return digest_json(self.compile_payload)

    def canonical(self) -> dict[str, Any]:
        payload = self.compile_payload
        payload["compile_digest"] = self.compile_digest
        return payload

    @property
    def order_key(self) -> tuple[str, str, int]:
        return (self.cluster_id, self.task_id, self.shard_index)

    @property
    def dispatch_request(self) -> dict[str, Any]:
        return {
            "type": "dispatch",
            "shard_id": self.stable_id,
            "cluster_id": self.cluster_id,
            "task_id": self.task_id,
            "shard_index": self.shard_index,
            "revision_version": self.revision_version,
            "config_keys": list(self.config_keys),
            "config_values": {
                key: self.config_values[key] for key in self.config_keys
            },
            "seed": self.cluster_seed,
            "sample_count": self.sample_count,
            "group_count": self.group_count,
        }


@dataclass(frozen=True)
class Plan:
    manifest_digest: str
    shards: tuple[ShardPlan, ...]
    plan_digest: str

    @property
    def by_id(self) -> dict[str, ShardPlan]:
        return {shard.stable_id: shard for shard in self.shards}

    @property
    def digest_by_id(self) -> dict[str, str]:
        return {shard.stable_id: shard.compile_digest for shard in self.shards}

    def canonical_without_digest(self) -> dict[str, Any]:
        return {
            "format_version": PLAN_FORMAT_VERSION,
            "manifest_digest": self.manifest_digest,
            "shards": [shard.canonical() for shard in self.shards],
        }

    def canonical(self) -> dict[str, Any]:
        payload = self.canonical_without_digest()
        payload["plan_digest"] = self.plan_digest
        return payload

    def serialized(self) -> str:
        return canonical_json(self.canonical())


def _revision_for(
    manifest: Manifest,
    stable_id: str,
    preserved_revisions: Mapping[str, int] | None,
) -> Revision:
    if preserved_revisions is None or stable_id not in preserved_revisions:
        return manifest.latest_revision
    version = preserved_revisions[stable_id]
    if not isinstance(version, int) or isinstance(version, bool):
        raise PipelineError(
            E_MANIFEST,
            "preserved revision must be an integer",
            field="preserved_revisions",
            shard_id=stable_id,
        )
    revision = manifest.revision_by_version.get(version)
    if revision is None:
        raise PipelineError(
            E_MANIFEST,
            "preserved revision does not exist in the manifest",
            field="preserved_revisions",
            shard_id=stable_id,
            details={"revision_version": version},
        )
    return revision


def compile_plan(
    manifest: Manifest,
    preserved_revisions: Mapping[str, int] | None = None,
) -> Plan:
    """Compile all logical shards.

    ``preserved_revisions`` maps a stable shard id to the old revision that the
    shard must continue to use.  Shards absent from the mapping compile against
    the latest revision.  This is the primitive used by selective recompilation.
    """

    clusters = manifest.cluster_by_id
    shards: list[ShardPlan] = []
    for task in manifest.tasks:
        for cluster_id in task.target_clusters:
            cluster = clusters[cluster_id]
            for shard_index in range(task.shards):
                stable_id = stable_shard_id(cluster_id, task.id, shard_index)
                revision = _revision_for(manifest, stable_id, preserved_revisions)
                missing = [
                    key
                    for key in task.config_keys
                    if key not in revision.effective_values
                ]
                if missing:
                    raise PipelineError(
                        E_MANIFEST,
                        "task references config keys unavailable in the selected revision",
                        field=f"revisions[{revision.version}].values",
                        shard_id=stable_id,
                        details={"missing_keys": missing},
                    )
                shards.append(
                    ShardPlan(
                        stable_id=stable_id,
                        cluster_id=cluster_id,
                        task_id=task.id,
                        shard_index=shard_index,
                        revision_version=revision.version,
                        config_keys=task.config_keys,
                        config_values={
                            key: revision.effective_values[key]
                            for key in task.config_keys
                        },
                        config_relations={
                            key: revision_dependency_closure(revision, key)
                            for key in task.config_keys
                        },
                        resource_units=task.resource_units,
                        task_concurrency=task.concurrency_limit,
                        cluster_seed=cluster.seed,
                        cluster_fingerprint=digest_json(cluster.canonical()),
                    )
                )

    shards.sort(key=lambda item: item.order_key)
    without_digest = {
        "format_version": PLAN_FORMAT_VERSION,
        "manifest_digest": manifest.manifest_digest,
        "shards": [shard.canonical() for shard in shards],
    }
    plan_digest = digest_json(without_digest)
    return Plan(
        manifest_digest=manifest.manifest_digest,
        shards=tuple(shards),
        plan_digest=plan_digest,
    )


def _first_field_difference(left: Any, right: Any, prefix: str = "") -> tuple[str, Any, Any]:
    if type(left) is not type(right):
        return (prefix or "<value>", left, right)
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys | right_keys):
            field = f"{prefix}.{key}" if prefix else key
            if key not in left:
                return (field, "<missing>", right[key])
            if key not in right:
                return (field, left[key], "<missing>")
            if left[key] != right[key]:
                return _first_field_difference(left[key], right[key], field)
        return (prefix or "<value>", left, right)
    if isinstance(left, list):
        common = min(len(left), len(right))
        for index in range(common):
            if left[index] != right[index]:
                field = f"{prefix}[{index}]"
                return _first_field_difference(left[index], right[index], field)
        if len(left) != len(right):
            field = f"{prefix}[{common}]"
            return (field, left[common:] if len(left) > common else "<missing>", right[common:] if len(right) > common else "<missing>")
        return (prefix or "<value>", left, right)
    return (prefix or "<value>", left, right)


def compare_plans(expected: Plan, actual: Plan) -> None:
    """Raise ``E_DIVERGE`` at the first differing stable shard/field."""

    expected_map = expected.by_id
    actual_map = actual.by_id
    for stable_id in sorted(set(expected_map) | set(actual_map)):
        if stable_id not in expected_map:
            raise PipelineError(
                E_DIVERGE,
                "compiled plan contains an unexpected shard",
                shard_id=stable_id,
                field="shard",
                details={"expected": "<missing>", "actual": "present"},
            )
        if stable_id not in actual_map:
            raise PipelineError(
                E_DIVERGE,
                "compiled plan is missing a shard",
                shard_id=stable_id,
                field="shard",
                details={"expected": "present", "actual": "<missing>"},
            )
        # Compare the source payload before derived hashes, so the reported
        # field is actionable (for example ``config_values.base``).
        left = expected_map[stable_id].compile_payload
        right = actual_map[stable_id].compile_payload
        if left == right:
            continue
        field, left_value, right_value = _first_field_difference(left, right)
        raise PipelineError(
            E_DIVERGE,
            "compiled shard differs at the first observable field",
            shard_id=stable_id,
            field=field,
            details={"expected": left_value, "actual": right_value},
        )


def stable_shard_order(shard_ids: list[str]) -> list[str]:
    """Stable lexical order for already encoded shard ids."""

    return sorted(shard_ids)


def logical_key(shard: ShardPlan) -> tuple[str, str, int]:
    return shard.order_key
