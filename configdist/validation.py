from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .canonical import read_json
from .errors import AppError
from .models import Budgets, ClusterSpec, Revision, Spec, TaskSpec, Versions


FORMAT_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _fail(message: str, field: str, location: str, code: str = "E_SPEC", **details: Any) -> None:
    raise AppError(code=code, message=message, field=field, location=location, details=details)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_mapping(value: Any, field: str, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("必须为 JSON 对象", field, location)
    return value


def _require_list(value: Any, field: str, location: str) -> list[Any]:
    if not isinstance(value, list):
        _fail("必须为 JSON 数组", field, location)
    return value


def _require_text(value: Any, field: str, location: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("必须为非空字符串", field, location)
    return value


def _require_id(value: Any, field: str, location: str) -> str:
    text = _require_text(value, field, location)
    if not _ID_RE.fullmatch(text):
        _fail("标识只允许 1-64 位字母、数字、点、下划线或连字符", field, location)
    return text


def _require_nonnegative_int(value: Any, field: str, location: str) -> int:
    if not _is_int(value) or value < 0:
        _fail("必须为非负整数", field, location)
    return value


def _revision_cycle(edges: dict[tuple[str, int], int | None]) -> list[int] | None:
    visiting: set[tuple[str, int]] = set()
    visited: set[tuple[str, int]] = set()
    path: list[tuple[str, int]] = []

    def visit(node: tuple[str, int]) -> list[tuple[str, int]] | None:
        if node in visiting:
            start = path.index(node)
            return path[start:] + [node]
        if node in visited:
            return None
        visiting.add(node)
        path.append(node)
        parent = edges.get(node)
        if parent is not None:
            found = visit((node[0], parent))
            if found:
                return found
        path.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in sorted(edges):
        found = visit(node)
        if found:
            return [version for _, version in found]
    return None


def validate_spec(raw: Any) -> Spec:
    root = _require_mapping(raw, "spec", "$")
    format_version = root.get("format_version")
    if not _is_int(format_version) or format_version != FORMAT_VERSION:
        _fail(f"format_version 必须为整数 {FORMAT_VERSION}", "format_version", "$.format_version")

    seed = root.get("simulation_seed")
    if not _is_int(seed) or seed < 0:
        _fail("simulation_seed 必须为非负整数", "simulation_seed", "$.simulation_seed")

    clusters_raw = _require_list(root.get("clusters"), "clusters", "$.clusters")
    if not clusters_raw:
        _fail("集群集合不能为空", "clusters", "$.clusters")

    clusters: list[ClusterSpec] = []
    cluster_ids: set[str] = set()
    ports: dict[int, str] = {}
    for index, item in enumerate(clusters_raw):
        location = f"$.clusters[{index}]"
        record = _require_mapping(item, "clusters", location)
        cluster_id = _require_id(record.get("id"), "id", f"{location}.id")
        if cluster_id in cluster_ids:
            _fail("集群标识必须全局唯一", "id", f"{location}.id", duplicate=cluster_id)
        cluster_ids.add(cluster_id)
        host = record.get("host")
        if host != "127.0.0.1":
            _fail("host 必须为回环字面量 127.0.0.1", "host", f"{location}.host")
        port = record.get("port")
        if not _is_int(port) or not 1 <= port <= 65535:
            _fail("port 必须为 1-65535 的整数", "port", f"{location}.port")
        if port in ports:
            _fail("端口必须全局唯一", "port", f"{location}.port", duplicate=port)
        ports[port] = cluster_id
        nodes_raw = _require_list(record.get("nodes"), "nodes", f"{location}.nodes")
        if not nodes_raw:
            _fail("节点列表不能为空", "nodes", f"{location}.nodes")
        nodes: list[str] = []
        seen_nodes: set[str] = set()
        for node_index, node in enumerate(nodes_raw):
            node_id = _require_id(node, "nodes", f"{location}.nodes[{node_index}]")
            if node_id in seen_nodes:
                _fail("集群内节点标识必须唯一", "nodes", f"{location}.nodes[{node_index}]")
            seen_nodes.add(node_id)
            nodes.append(node_id)
        concurrency = _require_nonnegative_int(record.get("concurrency"), "concurrency", f"{location}.concurrency")
        clusters.append(
            ClusterSpec(
                id=cluster_id,
                host=host,
                port=port,
                nodes=tuple(sorted(nodes)),
                concurrency=concurrency,
            )
        )

    revisions_raw = _require_list(root.get("revisions"), "revisions", "$.revisions")
    revisions: list[Revision] = []
    revision_keys: set[tuple[str, int]] = set()
    revision_records: list[tuple[int, dict[str, Any], str, str, int, int | None]] = []
    for index, item in enumerate(revisions_raw):
        location = f"$.revisions[{index}]"
        record = _require_mapping(item, "revisions", location)
        cluster_id = _require_id(record.get("cluster_id"), "cluster_id", f"{location}.cluster_id")
        if cluster_id not in cluster_ids:
            _fail("修订引用的集群不存在", "cluster_id", f"{location}.cluster_id")
        version = record.get("version")
        if not _is_int(version) or version < 1:
            _fail("修订版本号必须为正整数", "version", f"{location}.version")
        key = (cluster_id, version)
        if key in revision_keys:
            _fail("同一集群的修订版本号必须唯一", "version", f"{location}.version")
        revision_keys.add(key)
        parent = record.get("parent")
        if parent is not None and (not _is_int(parent) or parent < 1):
            _fail("parent 必须为 null 或正整数版本号", "parent", f"{location}.parent")
        config = _require_mapping(record.get("config"), "config", f"{location}.config")
        if not config:
            _fail("配置键集合不能为空", "config", f"{location}.config")
        for key_name in sorted(config):
            if not isinstance(key_name, str) or not key_name:
                _fail("配置键必须为非空字符串", "config", f"{location}.config")
        revision_records.append((index, record, cluster_id, location, version, parent))

    edges: dict[tuple[str, int], int | None] = {
        (cluster_id, version): parent for _, _, cluster_id, _, version, parent in revision_records
    }
    cycle = _revision_cycle(edges)
    if cycle is not None:
        first_version = cycle[0]
        cycle_cluster = next(cluster_id for cluster_id, version in edges if version == first_version)
        location = next(loc for _, _, cid, loc, ver, _ in revision_records if cid == cycle_cluster and ver == first_version)
        _fail(
            "配置修订父引用成环",
            "parent",
            f"{location}.parent",
            code="E_CYCLE",
            cycle=cycle,
        )

    revisions_by_cluster: dict[str, list[tuple[int, dict[str, Any], str, int, int | None]]] = {}
    for index, record, cluster_id, location, version, parent in revision_records:
        revisions_by_cluster.setdefault(cluster_id, []).append((index, record, location, version, parent))
    for cluster_id, records in revisions_by_cluster.items():
        records.sort(key=lambda item: item[3])
        minimum_version = records[0][3]
        for index, record, location, version, parent in records:
            if version == minimum_version:
                if parent is not None:
                    _fail("集群首个修订的 parent 必须为 null", "parent", f"{location}.parent")
                continue
            if parent is None:
                _fail("非首个修订必须引用父修订", "parent", f"{location}.parent")
            if (cluster_id, parent) not in revision_keys:
                _fail("父修订引用不存在", "parent", f"{location}.parent", parent=parent)
            if parent >= version:
                _fail("修订版本号必须单调递增且父版本更小", "parent", f"{location}.parent")
        revisions.extend(
            Revision(
                cluster_id=cluster_id,
                version=version,
                parent=parent,
                config={key: record["config"][key] for key in sorted(record["config"])},
            )
            for _, record, _, version, parent in records
        )

    for cluster_id in sorted(cluster_ids):
        if cluster_id not in revisions_by_cluster:
            _fail("每个集群必须至少有一个配置修订", "revisions", "$.revisions", cluster_id=cluster_id)

    current_revisions = {
        cluster_id: max(records, key=lambda item: item[3])[1]
        for cluster_id, records in revisions_by_cluster.items()
    }

    budgets_raw = _require_mapping(root.get("budgets"), "budgets", "$.budgets")
    global_budget = _require_nonnegative_int(
        budgets_raw.get("global_resource_units"), "global_resource_units", "$.budgets.global_resource_units"
    )
    default_parallel = _require_nonnegative_int(
        budgets_raw.get("default_task_parallel"), "default_task_parallel", "$.budgets.default_task_parallel"
    )

    tasks_raw = _require_list(root.get("tasks"), "tasks", "$.tasks")
    tasks: list[TaskSpec] = []
    task_ids: set[str] = set()
    for index, item in enumerate(tasks_raw):
        location = f"$.tasks[{index}]"
        record = _require_mapping(item, "tasks", location)
        task_id = _require_id(record.get("id"), "id", f"{location}.id")
        if task_id in task_ids:
            _fail("任务标识必须全局唯一", "id", f"{location}.id", duplicate=task_id)
        task_ids.add(task_id)
        keys_raw = _require_list(record.get("config_keys"), "config_keys", f"{location}.config_keys")
        if not keys_raw:
            _fail("配置键集合不能为空", "config_keys", f"{location}.config_keys")
        config_keys: list[str] = []
        seen_keys: set[str] = set()
        for key_index, key_name in enumerate(keys_raw):
            key_text = _require_text(key_name, "config_keys", f"{location}.config_keys[{key_index}]")
            if key_text in seen_keys:
                _fail("任务配置键不能重复", "config_keys", f"{location}.config_keys[{key_index}]")
            seen_keys.add(key_text)
            config_keys.append(key_text)
        targets_raw = _require_list(record.get("target_clusters"), "target_clusters", f"{location}.target_clusters")
        if not targets_raw:
            _fail("目标集群列表不能为空", "target_clusters", f"{location}.target_clusters")
        targets: list[str] = []
        seen_targets: set[str] = set()
        for target_index, target in enumerate(targets_raw):
            target_id = _require_id(target, "target_clusters", f"{location}.target_clusters[{target_index}]")
            if target_id not in cluster_ids:
                _fail("目标集群不存在", "target_clusters", f"{location}.target_clusters[{target_index}]")
            if target_id in seen_targets:
                _fail("目标集群不能重复", "target_clusters", f"{location}.target_clusters[{target_index}]")
            seen_targets.add(target_id)
            targets.append(target_id)
        for target_id in sorted(targets):
            current_config = current_revisions[target_id]["config"]
            for key_name in sorted(config_keys):
                if key_name not in current_config:
                    _fail(
                        "任务配置键在当前集群修订中不存在",
                        "config_keys",
                        f"{location}.config_keys",
                        cluster_id=target_id,
                        config_key=key_name,
                    )
        shards = _require_nonnegative_int(record.get("shards"), "shards", f"{location}.shards")
        resource_units = _require_nonnegative_int(
            record.get("resource_units"), "resource_units", f"{location}.resource_units"
        )
        max_parallel_raw = record.get("max_parallel", default_parallel)
        max_parallel = _require_nonnegative_int(max_parallel_raw, "max_parallel", f"{location}.max_parallel")
        tasks.append(
            TaskSpec(
                id=task_id,
                config_keys=tuple(sorted(config_keys)),
                target_clusters=tuple(sorted(targets)),
                shards=shards,
                resource_units=resource_units,
                max_parallel=max_parallel,
            )
        )

    versions_raw = root.get("versions", {"clusters": 1, "revisions": 1, "budgets": 1})
    versions_record = _require_mapping(versions_raw, "versions", "$.versions")
    versions = Versions(
        clusters=_require_nonnegative_int(versions_record.get("clusters"), "clusters", "$.versions.clusters"),
        revisions=_require_nonnegative_int(versions_record.get("revisions"), "revisions", "$.versions.revisions"),
        budgets=_require_nonnegative_int(versions_record.get("budgets"), "budgets", "$.versions.budgets"),
    )

    return Spec(
        format_version=FORMAT_VERSION,
        simulation_seed=seed,
        clusters=tuple(sorted(clusters, key=lambda cluster: cluster.id)),
        revisions=tuple(sorted(revisions, key=lambda revision: (revision.cluster_id, revision.version))),
        tasks=tuple(sorted(tasks, key=lambda task: task.id)),
        budgets=Budgets(global_resource_units=global_budget, default_task_parallel=default_parallel),
        versions=versions,
    )


def load_spec(path: str | Path) -> Spec:
    try:
        raw = read_json(path)
    except FileNotFoundError:
        raise AppError("E_SPEC", "清单文件不存在", "spec", str(path)) from None
    except (OSError, ValueError) as exc:
        raise AppError("E_SPEC", f"清单文件无法解析: {exc}", "spec", str(path)) from None
    return validate_spec(raw)


def require_first_error(raw: Any) -> AppError:
    try:
        validate_spec(raw)
    except AppError as exc:
        return exc
    raise AssertionError("specification was valid")
