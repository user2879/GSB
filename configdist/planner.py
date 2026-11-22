from __future__ import annotations

from typing import Any

from .canonical import canonical_bytes, sha256_bytes
from .models import Spec


def shard_identifier(cluster_id: str, task_id: str, shard_index: int) -> str:
    return f"{cluster_id}/{task_id}/{shard_index:04d}"


def compile_plan(spec: Spec) -> dict[str, Any]:
    current = spec.current_revisions()
    clusters = spec.cluster_map()
    shards: list[dict[str, Any]] = []
    task_meta: list[dict[str, Any]] = []

    for task in sorted(spec.tasks, key=lambda item: item.id):
        task_meta.append(
            {
                "task_id": task.id,
                "config_keys": sorted(task.config_keys),
                "target_clusters": sorted(task.target_clusters),
                "shards": task.shards,
                "resource_units": task.resource_units,
                "max_parallel": task.max_parallel,
            }
        )
        for cluster_id in sorted(task.target_clusters):
            cluster = clusters[cluster_id]
            revision = current[cluster_id]
            selected_config = {key: revision.config[key] for key in sorted(task.config_keys)}
            for shard_index in range(task.shards):
                shards.append(
                    {
                        "shard_id": shard_identifier(cluster_id, task.id, shard_index),
                        "cluster_id": cluster_id,
                        "cluster_port": cluster.port,
                        "nodes": list(cluster.nodes),
                        "cluster_quota": cluster.concurrency,
                        "task_id": task.id,
                        "shard_index": shard_index,
                        "config_keys": sorted(task.config_keys),
                        "config_values": selected_config,
                        "resource_units": task.resource_units,
                        "revision_version": revision.version,
                        "max_parallel": task.max_parallel,
                    }
                )

    shards.sort(key=lambda item: (item["cluster_id"], item["task_id"], item["shard_index"]))
    body: dict[str, Any] = {
        "format_version": 1,
        "simulation_seed": spec.simulation_seed,
        "versions": spec.versions.to_dict(),
        "global_resource_units": spec.budgets.global_resource_units,
        "default_task_parallel": spec.budgets.default_task_parallel,
        "tasks": sorted(task_meta, key=lambda item: item["task_id"]),
        "shards": shards,
    }
    body["plan_digest"] = sha256_bytes(canonical_bytes(body))
    return body


def plan_bytes(plan: dict[str, Any]) -> bytes:
    return canonical_bytes(plan)


def shard_fingerprint(shard: dict[str, Any]) -> str:
    fields = {
        "shard_id": shard["shard_id"],
        "cluster_id": shard["cluster_id"],
        "task_id": shard["task_id"],
        "shard_index": shard["shard_index"],
        "config_keys": shard["config_keys"],
        "config_values": shard["config_values"],
        "resource_units": shard["resource_units"],
        "revision_version": shard["revision_version"],
        "max_parallel": shard["max_parallel"],
        "cluster_quota": shard["cluster_quota"],
    }
    return sha256_bytes(canonical_bytes(fields))


def compare_plans(old_plan: dict[str, Any], new_plan: dict[str, Any]) -> dict[str, Any]:
    old_shards = {shard["shard_id"]: shard for shard in old_plan.get("shards", [])}
    new_shards = {shard["shard_id"]: shard for shard in new_plan.get("shards", [])}
    changed: set[str] = set()
    for shard_id in sorted(set(old_shards) | set(new_shards)):
        old = old_shards.get(shard_id)
        new = new_shards.get(shard_id)
        if old is None or new is None or shard_fingerprint(old) != shard_fingerprint(new):
            changed.add(shard_id)

    reasons: list[str] = []
    old_clusters = {shard["cluster_id"] for shard in old_shards.values()}
    new_clusters = {shard["cluster_id"] for shard in new_shards.values()}
    for cluster_id in sorted(old_clusters | new_clusters):
        old_revisions = {shard["revision_version"] for shard in old_shards.values() if shard["cluster_id"] == cluster_id}
        new_revisions = {shard["revision_version"] for shard in new_shards.values() if shard["cluster_id"] == cluster_id}
        if old_revisions and new_revisions and old_revisions != new_revisions:
            reasons.append(
                f"revision changed for {cluster_id}: {sorted(old_revisions)} -> {sorted(new_revisions)}"
            )
    old_tasks = {task["task_id"]: task for task in old_plan.get("tasks", [])}
    new_tasks = {task["task_id"]: task for task in new_plan.get("tasks", [])}
    for task_id in sorted(set(old_tasks) | set(new_tasks)):
        old_task = old_tasks.get(task_id)
        new_task = new_tasks.get(task_id)
        if old_task is None or new_task is None:
            reasons.append(f"task membership changed for {task_id}")
            continue
        if old_task["target_clusters"] != new_task["target_clusters"]:
            reasons.append(
                f"target clusters changed for {task_id}: "
                f"{old_task['target_clusters']} -> {new_task['target_clusters']}"
            )
        if old_task["config_keys"] != new_task["config_keys"]:
            reasons.append(f"config keys changed for {task_id}")
        if old_task["shards"] != new_task["shards"]:
            reasons.append(f"shard count changed for {task_id}: {old_task['shards']} -> {new_task['shards']}")
    if not reasons and changed:
        reasons.append("plan content changed")
    affected = set(old_shards) | set(new_shards)
    reasons.append("whole generation replacement invalidates every prior conclusion")
    return {
        "affected_shards": sorted(affected),
        "changed_shards": sorted(changed),
        "reasons": reasons,
        "old_plan_digest": old_plan.get("plan_digest"),
        "new_plan_digest": new_plan.get("plan_digest"),
    }
