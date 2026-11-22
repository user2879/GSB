from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClusterSpec:
    id: str
    host: str
    port: int
    nodes: tuple[str, ...]
    concurrency: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "nodes": list(self.nodes),
            "concurrency": self.concurrency,
        }


@dataclass(frozen=True)
class Revision:
    cluster_id: str
    version: int
    parent: int | None
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "version": self.version,
            "parent": self.parent,
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class TaskSpec:
    id: str
    config_keys: tuple[str, ...]
    target_clusters: tuple[str, ...]
    shards: int
    resource_units: int
    max_parallel: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "config_keys": list(self.config_keys),
            "target_clusters": list(self.target_clusters),
            "shards": self.shards,
            "resource_units": self.resource_units,
            "max_parallel": self.max_parallel,
        }


@dataclass(frozen=True)
class Budgets:
    global_resource_units: int
    default_task_parallel: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_resource_units": self.global_resource_units,
            "default_task_parallel": self.default_task_parallel,
        }


@dataclass(frozen=True)
class Versions:
    clusters: int
    revisions: int
    budgets: int

    def to_dict(self) -> dict[str, int]:
        return {
            "clusters": self.clusters,
            "revisions": self.revisions,
            "budgets": self.budgets,
        }


@dataclass(frozen=True)
class Spec:
    format_version: int
    simulation_seed: int
    clusters: tuple[ClusterSpec, ...]
    revisions: tuple[Revision, ...]
    tasks: tuple[TaskSpec, ...]
    budgets: Budgets
    versions: Versions

    def cluster_map(self) -> dict[str, ClusterSpec]:
        return {cluster.id: cluster for cluster in self.clusters}

    def revision_map(self) -> dict[tuple[str, int], Revision]:
        return {(revision.cluster_id, revision.version): revision for revision in self.revisions}

    def current_revisions(self) -> dict[str, Revision]:
        current: dict[str, Revision] = {}
        for revision in self.revisions:
            previous = current.get(revision.cluster_id)
            if previous is None or revision.version > previous.version:
                current[revision.cluster_id] = revision
        return current

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "simulation_seed": self.simulation_seed,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "revisions": [revision.to_dict() for revision in self.revisions],
            "tasks": [task.to_dict() for task in self.tasks],
            "budgets": self.budgets.to_dict(),
            "versions": self.versions.to_dict(),
        }
