"""Cluster / revision-chain / dispatch-task data model with strict validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DiagError, raise_spec

ALLOWED_STRATEGIES = ("canary", "rolling", "full")
ALLOWED_CALIBERS = (1, 2, 3)


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    host: str
    port: int
    nodes: tuple[str, ...]
    seed: int
    sample_rows: int


@dataclass(frozen=True)
class Task:
    task_id: str
    keys: tuple[str, ...]
    targets: tuple[str, ...]
    caliber: int


@dataclass(frozen=True)
class Revision:
    version: int
    parent: int | None
    values: dict[str, int]
    strategy: str
    derives: tuple[int, ...]


@dataclass
class Spec:
    clusters: tuple[Cluster, ...]
    tasks: tuple[Task, ...]
    revisions: tuple[Revision, ...]
    source_files: dict[str, str] = field(default_factory=dict)

    @property
    def cluster_map(self) -> dict[str, Cluster]:
        return {c.cluster_id: c for c in self.clusters}

    @property
    def task_map(self) -> dict[str, Task]:
        return {t.task_id: t for t in self.tasks}

    def revision(self, version: int) -> Revision:
        for rev in self.revisions:
            if rev.version == version:
                return rev
        raise KeyError(version)

    def ancestors_including(self, version: int) -> list[int]:
        """Return the root->self version path for a linear revision chain."""
        path: list[int] = []
        cur: int | None = version
        seen = set()
        while cur is not None:
            if cur in seen:
                raise_spec("revision chain contains a cycle", {"revision": cur})
            seen.add(cur)
            path.append(cur)
            cur = self.revision(cur).parent
        path.reverse()
        return path


def _read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_spec(clusters_path, tasks_path, revisions_path) -> Spec:
    """Load and strictly validate the three declaration files.

    The whole declaration is rejected on the first violation with ``E_SPEC``,
    including the offending revision/version and the field path.
    """
    clusters_raw = _read_json(clusters_path)
    tasks_raw = _read_json(tasks_path)
    revisions_raw = _read_json(revisions_path)
    spec = _build_spec(clusters_raw, tasks_raw, revisions_raw)
    spec.source_files = {
        "clusters": str(clusters_path),
        "tasks": str(tasks_path),
        "revisions": str(revisions_path),
    }
    return spec


def _build_spec(clusters_raw, tasks_raw, revisions_raw) -> Spec:
    clusters = _validate_clusters(clusters_raw)
    tasks = _validate_tasks(tasks_raw, clusters)
    revisions = _validate_revisions(revisions_raw)
    _validate_task_keys(tasks, revisions)
    return Spec(clusters=clusters, tasks=tasks, revisions=revisions)


def _validate_clusters(raw) -> tuple[Cluster, ...]:
    if not isinstance(raw, dict) or "clusters" not in raw:
        raise_spec("clusters file must be an object with a 'clusters' array",
                   {"field": "clusters"})
    items = raw["clusters"]
    if not isinstance(items, list) or not items:
        raise_spec("'clusters' must be a non-empty array", {"field": "clusters"})
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    result: list[Cluster] = []
    for index, item in enumerate(items):
        loc = {"field": f"clusters[{index}]"}
        if not isinstance(item, dict):
            raise_spec("cluster declaration must be an object", loc)
        cid = item.get("cluster_id")
        loc = {"field": f"clusters[{index}].cluster_id", "cluster": cid}
        if not isinstance(cid, str) or not cid:
            raise_spec("cluster_id must be a non-empty string", loc)
        if cid in seen_ids:
            raise_spec(f"duplicate cluster_id {cid!r}", loc)
        seen_ids.add(cid)
        host = item.get("host")
        if host != "127.0.0.1":
            raise_spec("host must be the loopback literal 127.0.0.1",
                       {"field": f"clusters[{index}].host", "cluster": cid})
        port = item.get("port")
        if not isinstance(port, int) or isinstance(port, bool) or not (1 <= port <= 65535):
            raise_spec("port must be an integer in [1,65535]",
                       {"field": f"clusters[{index}].port", "cluster": cid})
        if port in seen_ports:
            raise_spec(f"loopback port {port} is declared more than once",
                       {"field": f"clusters[{index}].port", "cluster": cid})
        seen_ports.add(port)
        nodes = item.get("nodes")
        if not isinstance(nodes, list) or not nodes or \
                any(not isinstance(n, str) or not n for n in nodes):
            raise_spec("nodes must be a non-empty array of non-empty strings",
                       {"field": f"clusters[{index}].nodes", "cluster": cid})
        seed = item.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise_spec("seed must be an integer",
                       {"field": f"clusters[{index}].seed", "cluster": cid})
        sample_rows = item.get("sample_rows", 64)
        if not isinstance(sample_rows, int) or isinstance(sample_rows, bool) or sample_rows <= 0:
            raise_spec("sample_rows must be a positive integer",
                       {"field": f"clusters[{index}].sample_rows", "cluster": cid})
        result.append(Cluster(cid, host, port, tuple(nodes), seed, sample_rows))
    return tuple(result)


def _validate_tasks(raw, clusters: tuple[Cluster, ...]) -> tuple[Task, ...]:
    if not isinstance(raw, dict) or "tasks" not in raw:
        raise_spec("tasks file must be an object with a 'tasks' array",
                   {"field": "tasks"})
    items = raw["tasks"]
    if not isinstance(items, list) or not items:
        raise_spec("'tasks' must be a non-empty array", {"field": "tasks"})
    cluster_ids = {c.cluster_id for c in clusters}
    seen: set[str] = set()
    result: list[Task] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise_spec("task declaration must be an object",
                       {"field": f"tasks[{index}]"})
        tid = item.get("task_id")
        loc = {"field": f"tasks[{index}].task_id", "task": tid}
        if not isinstance(tid, str) or not tid:
            raise_spec("task_id must be a non-empty string", loc)
        if tid in seen:
            raise_spec(f"duplicate task_id {tid!r}", loc)
        seen.add(tid)
        keys = item.get("keys")
        if not isinstance(keys, list) or not keys or \
                any(not isinstance(k, str) or not k for k in keys):
            raise_spec("keys must be a non-empty array of non-empty strings",
                       {"field": f"tasks[{index}].keys", "task": tid})
        if len(set(keys)) != len(keys):
            raise_spec("task keys must be unique",
                       {"field": f"tasks[{index}].keys", "task": tid})
        targets = item.get("targets")
        if not isinstance(targets, list) or not targets:
            raise_spec("targets must be a non-empty array",
                       {"field": f"tasks[{index}].targets", "task": tid})
        for target in targets:
            if target not in cluster_ids:
                raise_spec(f"target cluster {target!r} is not declared",
                           {"field": f"tasks[{index}].targets", "task": tid,
                            "cluster": target})
        caliber = item.get("caliber")
        if caliber not in ALLOWED_CALIBERS:
            raise_spec(f"caliber must be one of {ALLOWED_CALIBERS}",
                       {"field": f"tasks[{index}].caliber", "task": tid})
        result.append(Task(tid, tuple(keys), tuple(targets), caliber))
    return tuple(result)


def _validate_revisions(raw) -> tuple[Revision, ...]:
    if not isinstance(raw, dict) or "revisions" not in raw:
        raise_spec("revisions file must be an object with a 'revisions' array",
                   {"field": "revisions"})
    items = raw["revisions"]
    if not isinstance(items, list) or not items:
        raise_spec("'revisions' must be a non-empty array", {"field": "revisions"})
    versions: set[int] = set()
    result: list[Revision] = []
    previous_version: int | None = None
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise_spec("revision declaration must be an object",
                       {"field": f"revisions[{index}]"})
        version = item.get("version")
        loc = {"field": f"revisions[{index}].version", "revision": version}
        if not isinstance(version, int) or isinstance(version, bool):
            raise_spec("revision version must be an integer", loc)
        if version in versions:
            raise_spec(f"duplicate revision version {version}", loc)
        versions.add(version)
        if previous_version is not None and version <= previous_version:
            raise_spec(
                f"revision versions must be strictly increasing; "
                f"{version} follows {previous_version}", loc)
        parent = item.get("parent")
        if not isinstance(parent, int) or isinstance(parent, bool):
            if parent is not None:
                raise_spec("parent must be an integer or null",
                           {"field": f"revisions[{index}].parent",
                            "revision": version})
        values = item.get("values", {})
        if not isinstance(values, dict) or any(
                not isinstance(k, str) or not isinstance(v, int) or isinstance(v, bool)
                for k, v in values.items()):
            raise_spec("values must map string keys to integer values",
                       {"field": f"revisions[{index}].values", "revision": version})
        strategy = item.get("strategy", "rolling")
        if strategy not in ALLOWED_STRATEGIES:
            raise_spec(f"strategy must be one of {ALLOWED_STRATEGIES}",
                       {"field": f"revisions[{index}].strategy",
                        "revision": version})
        derives = item.get("derives", [])
        if not isinstance(derives, list) or any(
                not isinstance(d, int) or isinstance(d, bool) for d in derives):
            raise_spec("derives must be an array of integer revision versions",
                       {"field": f"revisions[{index}].derives",
                        "revision": version})
        if len(set(derives)) != len(derives):
            raise_spec("derives entries must be unique",
                       {"field": f"revisions[{index}].derives",
                        "revision": version})
        if parent is not None and parent not in versions:
            raise_spec(
                f"parent revision {parent} does not exist or appears later",
                {"field": f"revisions[{index}].parent", "revision": version})
        for derived in derives:
            if derived not in versions:
                raise_spec(
                    f"derived reference {derived} does not exist or appears later",
                    {"field": f"revisions[{index}].derives",
                     "revision": version})
            if derived >= version:
                raise_spec(
                    f"derived reference {derived} must point to an earlier revision",
                    {"field": f"revisions[{index}].derives",
                     "revision": version})
        result.append(Revision(version, parent, dict(values), strategy, tuple(derives)))
        previous_version = version
    # Chain shape: exactly one root (parent == null) and every other parent
    # must reference the immediately preceding version (linear chain, no cycle).
    roots = [r.version for r in result if r.parent is None]
    if len(roots) != 1:
        raise_spec(f"revision chain must have exactly one root, found {roots}",
                   {"field": "revisions.parent"})
    for rev in result:
        if rev.parent is None:
            continue
        expected = result[result.index(rev) - 1].version
        if rev.parent != expected:
            raise_spec(
                f"revision {rev.version} parent {rev.parent} does not link the "
                f"immediate predecessor {expected} (non-linear chain or cycle)",
                {"field": "revisions.parent", "revision": rev.version})
    return tuple(result)


def _validate_task_keys(tasks: tuple[Task, ...], revisions: tuple[Revision, ...]) -> None:
    chain_keys: set[str] = set()
    for rev in revisions:
        chain_keys.update(rev.values.keys())
    for task in tasks:
        for key in task.keys:
            if key not in chain_keys:
                raise_spec(
                    f"task {task.task_id!r} references key {key!r} that never "
                    "appears in the revision chain",
                    {"field": "tasks.keys", "task": task.task_id, "key": key})
