from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import DiagError, E_REV, E_SPEC, SpecIssue
from .util import read_json


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
AGGREGATIONS = frozenset({"sum", "mean", "min", "max", "count"})
MAX_SAMPLE_COUNT = 1_000_000


@dataclass(frozen=True)
class ClusterSpec:
    id: str
    host: str
    port: int
    nodes: tuple[str, ...]


@dataclass(frozen=True)
class TaskSpec:
    id: str
    config_keys: tuple[str, ...]
    targets: tuple[str, ...]
    aggregation: str
    sample_count: int


@dataclass(frozen=True)
class RevisionSpec:
    version: int
    parent: int | None
    config: dict[str, Any]
    task_ids: tuple[str, ...]


@dataclass(frozen=True)
class SystemSpec:
    schema_version: int
    seed: int
    clusters: tuple[ClusterSpec, ...]
    tasks: tuple[TaskSpec, ...]
    revisions: tuple[RevisionSpec, ...]
    description: str
    source_path: Path

    @property
    def cluster_ids(self) -> tuple[str, ...]:
        return tuple(cluster.id for cluster in self.clusters)

    @property
    def clusters_by_id(self) -> dict[str, ClusterSpec]:
        return {cluster.id: cluster for cluster in self.clusters}

    @property
    def tasks_by_id(self) -> dict[str, TaskSpec]:
        return {task.id: task for task in self.tasks}

    @property
    def revisions_by_version(self) -> dict[int, RevisionSpec]:
        return {revision.version: revision for revision in self.revisions}

    @property
    def latest_state(self) -> RevisionSpec:
        return self.revisions[-1]

    def revision(self, version: int) -> RevisionSpec:
        try:
            return self.revisions_by_version[version]
        except KeyError as exc:
            raise DiagError(
                E_SPEC,
                f"unknown revision {version}",
                field_path="$.revisions",
            ) from exc

    def targeted_clusters(self, revision: RevisionSpec) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for task_id in revision.task_ids:
            task = self.tasks_by_id[task_id]
            for cluster_id in task.targets:
                if cluster_id not in seen:
                    seen.add(cluster_id)
                    result.append(cluster_id)
        return tuple(sorted(result))

    def task_for_cluster(self, revision: RevisionSpec, cluster_id: str) -> TaskSpec:
        matches = [
            self.tasks_by_id[task_id]
            for task_id in revision.task_ids
            if cluster_id in self.tasks_by_id[task_id].targets
        ]
        if len(matches) != 1:
            raise DiagError(
                E_SPEC,
                "revision must contain exactly one task for each targeted cluster",
                field_path=f"$.revisions[version={revision.version}].task_ids",
                details={"cluster_id": cluster_id, "match_count": len(matches)},
            )
        return matches[0]


def _issue(issues: list[SpecIssue], path: str, message: str) -> None:
    issues.append(SpecIssue(path, message))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_keys(
    value: dict[str, Any],
    *,
    path: str,
    allowed: Iterable[str],
    issues: list[SpecIssue],
) -> None:
    allowed_set = set(allowed)
    for key in value:
        if key not in allowed_set:
            _issue(issues, f"{path}.{key}", "unknown field")


def _nonempty_string(
    value: Any,
    *,
    path: str,
    issues: list[SpecIssue],
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        _issue(issues, path, "must be a non-empty string")
        return None
    if value != value.strip() or len(value) > 128:
        _issue(issues, path, "must have no surrounding whitespace and be at most 128 characters")
        return None
    return value


def _string_list(
    value: Any,
    *,
    path: str,
    issues: list[SpecIssue],
    nonempty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        _issue(issues, path, "must be a list")
        return ()
    if nonempty and not value:
        _issue(issues, path, "must not be empty")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        checked = _nonempty_string(item, path=item_path, issues=issues)
        if checked is None:
            continue
        if checked in seen:
            _issue(issues, item_path, "duplicate value")
            continue
        seen.add(checked)
        result.append(checked)
    return tuple(result)


def _apply_cycle_injection(data: dict[str, Any]) -> None:
    revisions = data.get("revisions")
    if not isinstance(revisions, list) or not revisions:
        return
    last = revisions[-1]
    if isinstance(last, dict) and _is_int(last.get("version")):
        last["parent"] = last["version"]


def load_spec(
    path: str | Path,
    *,
    cycle_code: str = E_SPEC,
    inject_cycle: bool = False,
) -> SystemSpec:
    """Load and validate a complete specification, rejecting it as a unit."""

    source_path = Path(path).resolve()
    try:
        raw = read_json(source_path)
    except OSError as exc:
        raise DiagError(E_SPEC, f"cannot read specification: {exc}", field_path="$") from exc
    except Exception as exc:  # json.JSONDecodeError is intentionally normalized
        raise DiagError(E_SPEC, f"invalid JSON specification: {exc}", field_path="$") from exc

    if not isinstance(raw, dict):
        raise DiagError(E_SPEC, "specification root must be an object", field_path="$")

    if inject_cycle:
        _apply_cycle_injection(raw)

    issues: list[SpecIssue] = []
    _check_keys(
        raw,
        path="$",
        allowed={"schema_version", "seed", "clusters", "tasks", "revisions", "description"},
        issues=issues,
    )

    schema_version = raw.get("schema_version")
    if not _is_int(schema_version) or schema_version != 1:
        _issue(issues, "$.schema_version", "must be the integer 1")

    seed = raw.get("seed")
    if not _is_int(seed) or not 0 <= seed <= 2**63 - 1:
        _issue(issues, "$.seed", "must be a non-negative signed 64-bit integer")

    description = raw.get("description", "")
    if not isinstance(description, str):
        _issue(issues, "$.description", "must be a string")
        description = ""

    clusters_raw = raw.get("clusters")
    if not isinstance(clusters_raw, list) or not clusters_raw:
        _issue(issues, "$.clusters", "must be a non-empty list")
        clusters_raw = []

    clusters: list[ClusterSpec] = []
    cluster_ids: set[str] = set()
    ports: set[int] = set()
    for index, item in enumerate(clusters_raw):
        path_prefix = f"$.clusters[{index}]"
        if not isinstance(item, dict):
            _issue(issues, path_prefix, "must be an object")
            continue
        _check_keys(item, path=path_prefix, allowed={"id", "host", "port", "nodes"}, issues=issues)
        cluster_id = _nonempty_string(item.get("id"), path=f"{path_prefix}.id", issues=issues)
        host = item.get("host")
        if host not in LOOPBACK_HOSTS:
            _issue(issues, f"{path_prefix}.host", "must be the loopback literal 127.0.0.1 or ::1")
        port = item.get("port")
        if not _is_int(port) or not 1024 <= port <= 65535:
            _issue(issues, f"{path_prefix}.port", "must be an integer from 1024 through 65535")
        nodes = _string_list(item.get("nodes"), path=f"{path_prefix}.nodes", issues=issues, nonempty=True)
        if cluster_id is not None:
            if cluster_id in cluster_ids:
                _issue(issues, f"{path_prefix}.id", f"duplicate cluster id {cluster_id!r}")
            else:
                cluster_ids.add(cluster_id)
        if _is_int(port):
            if port in ports:
                _issue(issues, f"{path_prefix}.port", f"port {port} is not globally unique")
            else:
                ports.add(port)
        if cluster_id is not None and host in LOOPBACK_HOSTS and _is_int(port) and nodes:
            clusters.append(ClusterSpec(cluster_id, host, port, nodes))

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        _issue(issues, "$.tasks", "must be a non-empty list")
        tasks_raw = []

    tasks: list[TaskSpec] = []
    task_ids: set[str] = set()
    task_index_by_id: dict[str, int] = {}
    for index, item in enumerate(tasks_raw):
        path_prefix = f"$.tasks[{index}]"
        if not isinstance(item, dict):
            _issue(issues, path_prefix, "must be an object")
            continue
        _check_keys(
            item,
            path=path_prefix,
            allowed={"id", "config_keys", "targets", "aggregation", "sample_count"},
            issues=issues,
        )
        task_id = _nonempty_string(item.get("id"), path=f"{path_prefix}.id", issues=issues)
        config_keys = _string_list(
            item.get("config_keys"),
            path=f"{path_prefix}.config_keys",
            issues=issues,
            nonempty=True,
        )
        targets = _string_list(
            item.get("targets"),
            path=f"{path_prefix}.targets",
            issues=issues,
            nonempty=True,
        )
        aggregation = item.get("aggregation")
        if aggregation not in AGGREGATIONS:
            _issue(
                issues,
                f"{path_prefix}.aggregation",
                f"must be one of {sorted(AGGREGATIONS)}",
            )
        sample_count = item.get("sample_count")
        if not _is_int(sample_count) or not 1 <= sample_count <= MAX_SAMPLE_COUNT:
            _issue(
                issues,
                f"{path_prefix}.sample_count",
                f"must be an integer from 1 through {MAX_SAMPLE_COUNT}",
            )
        if task_id is not None:
            if task_id in task_ids:
                _issue(issues, f"{path_prefix}.id", f"duplicate task id {task_id!r}")
            else:
                task_ids.add(task_id)
                task_index_by_id[task_id] = index
        for target_index, target in enumerate(targets):
            if target not in cluster_ids:
                _issue(
                    issues,
                    f"{path_prefix}.targets[{target_index}]",
                    f"unknown target cluster {target!r}",
                )
        if (
            task_id is not None
            and config_keys
            and isinstance(aggregation, str)
            and aggregation in AGGREGATIONS
            and _is_int(sample_count)
            and 1 <= sample_count <= MAX_SAMPLE_COUNT
            and targets
        ):
            tasks.append(TaskSpec(task_id, config_keys, targets, aggregation, sample_count))

    revisions_raw = raw.get("revisions")
    if not isinstance(revisions_raw, list) or not revisions_raw:
        _issue(issues, "$.revisions", "must be a non-empty list")
        revisions_raw = []

    revisions: list[RevisionSpec] = []
    versions: list[int] = []
    parents: dict[int, int | None] = {}
    version_to_index: dict[int, int] = {}
    for index, item in enumerate(revisions_raw):
        path_prefix = f"$.revisions[{index}]"
        if not isinstance(item, dict):
            _issue(issues, path_prefix, "must be an object")
            continue
        _check_keys(
            item,
            path=path_prefix,
            allowed={"version", "parent", "config", "task_ids"},
            issues=issues,
        )
        version = item.get("version")
        if not _is_int(version) or version < 1:
            _issue(issues, f"{path_prefix}.version", "must be a positive integer")
            version = None
        parent = item.get("parent")
        if parent is not None and (not _is_int(parent) or parent < 1):
            _issue(issues, f"{path_prefix}.parent", "must be null or a positive integer")
            parent = None
        config = item.get("config")
        if not isinstance(config, dict):
            _issue(issues, f"{path_prefix}.config", "must be an object")
            config = {}
        else:
            for key in config:
                if not isinstance(key, str) or not key:
                    _issue(issues, f"{path_prefix}.config", "configuration keys must be non-empty strings")
        revision_task_ids = _string_list(
            item.get("task_ids"),
            path=f"{path_prefix}.task_ids",
            issues=issues,
            nonempty=True,
        )
        for task_index, task_id in enumerate(revision_task_ids):
            if task_id not in task_ids:
                _issue(
                    issues,
                    f"{path_prefix}.task_ids[{task_index}]",
                    f"unknown task id {task_id!r}",
                )
        if version is not None:
            if version in version_to_index:
                _issue(issues, f"{path_prefix}.version", f"duplicate revision version {version}")
            else:
                version_to_index[version] = index
                versions.append(version)
                parents[version] = parent
        if version is not None and revision_task_ids:
            revisions.append(
                RevisionSpec(version=version, parent=parent, config=dict(config), task_ids=revision_task_ids)
            )

    # Cycle detection is intentionally separate from the linear-chain check so an injected
    # revision cycle is reported as E_REV while ordinary malformed input remains E_SPEC.
    cycle_path: str | None = None
    for start in parents:
        seen: set[int] = set()
        current: int | None = start
        while current is not None and current in parents:
            if current in seen:
                cycle_path = f"$.revisions[{version_to_index[start]}].parent"
                break
            seen.add(current)
            current = parents[current]
        if cycle_path is not None:
            break
    if cycle_path is not None:
        raise DiagError(
            cycle_code,
            "revision parent references form a cycle",
            field_path=cycle_path,
            issues=[SpecIssue(cycle_path, "revision parent cycle")],
        )

    for index, version in enumerate(versions):
        if index == 0:
            if parents[version] is not None:
                _issue(
                    issues,
                    f"$.revisions[{version_to_index[version]}].parent",
                    "first revision parent must be null",
                )
        elif parents[version] != versions[index - 1]:
            _issue(
                issues,
                f"$.revisions[{version_to_index[version]}].parent",
                f"parent must reference the preceding revision {versions[index - 1]}",
            )

    if versions != sorted(versions) or len(versions) != len(set(versions)):
        _issue(issues, "$.revisions", "revision versions must be strictly increasing in file order")
    if revisions and [revision.version for revision in revisions] != versions:
        _issue(issues, "$.revisions", "every revision must have a unique valid version")

    task_map = {task.id: task for task in tasks}
    all_declared_config_keys = {
        key for revision in revisions for key in revision.config
    }
    for task in tasks:
        for key_index, key in enumerate(task.config_keys):
            if key not in all_declared_config_keys:
                _issue(
                    issues,
                    f"$.tasks[{task_index_by_id[task.id]}].config_keys[{key_index}]",
                    f"configuration key {key!r} is not declared by any revision",
                )
    for revision in revisions:
        missing_keys: dict[str, list[str]] = {}
        targets_seen: dict[str, str] = {}
        for task_id in revision.task_ids:
            task = task_map.get(task_id)
            if task is None:
                continue
            missing = [key for key in task.config_keys if key not in revision.config]
            if missing:
                missing_keys[task_id] = missing
            for cluster_id in task.targets:
                previous_task = targets_seen.get(cluster_id)
                if previous_task is not None and previous_task != task_id:
                    _issue(
                        issues,
                        f"$.revisions[version={revision.version}].task_ids",
                        f"cluster {cluster_id!r} is targeted by multiple tasks in one revision",
                    )
                targets_seen[cluster_id] = task_id
        for task_id, missing in missing_keys.items():
            for key in missing:
                _issue(
                    issues,
                    f"$.revisions[version={revision.version}].config.{key}",
                    f"task {task_id!r} references a missing configuration key",
                )

    if revisions and clusters:
        first_revision = revisions[0]
        first_targets: set[str] = set()
        for task_id in first_revision.task_ids:
            task = task_map.get(task_id)
            if task is not None:
                first_targets.update(task.targets)
        for cluster_id in sorted(cluster_ids - first_targets):
            _issue(
                issues,
                "$.revisions[0].task_ids",
                f"first revision must target every cluster; missing {cluster_id!r}",
            )

    if issues:
        raise DiagError(
            E_SPEC,
            "specification rejected as a unit",
            field_path=issues[0].field_path,
            issues=issues,
        )

    return SystemSpec(
        schema_version=1,
        seed=int(seed),
        clusters=tuple(clusters),
        tasks=tuple(tasks),
        revisions=tuple(revisions),
        description=description,
        source_path=source_path,
    )
