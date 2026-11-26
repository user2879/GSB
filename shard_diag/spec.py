from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import SpecError


SUPPORTED_SCALES = (1, 10, 100, 1000, 10000, 1024, 4096)
SUPPORTED_EVENT_TYPES = {
    "workspace_initialized",
    "spec_loaded",
    "shard_added",
    "shard_removed",
    "shard_updated",
    "sample_replaced",
    "quota_version_added",
    "scale_changed",
    "schedule_run",
    "node_killed",
    "recovery_replay",
}


def load_json_file(path: str | Path) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SpecError(f"声明文件不存在: {path}", entry=str(path), field="file") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"JSON 非法: {exc.msg}",
            entry=str(path),
            field=f"json:{exc.lineno}:{exc.colno}",
        ) from exc


def _err(message: str, entry: str, field: str) -> SpecError:
    return SpecError(message, entry=entry, field=field)


def _require_object(value: Any, entry: str = "$") -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _err("顶层必须是 JSON 对象", entry, "$")
    return value


def _require_int(
    value: Any,
    entry: str,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _err(f"{field} 必须是整数", entry, field)
    if minimum is not None and value < minimum:
        raise _err(f"{field} 必须 >= {minimum}", entry, field)
    if maximum is not None and value > maximum:
        raise _err(f"{field} 必须 <= {maximum}", entry, field)
    return value


def _require_nonempty_str(value: Any, entry: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _err(f"{field} 必须是非空字符串", entry, field)
    return value


def _validate_format_version(doc: dict[str, Any], entry: str) -> None:
    version = doc.get("format_version")
    _require_int(version, entry, "format_version", minimum=1, maximum=1)


def validate_nodes(doc: Any) -> list[dict[str, Any]]:
    root = _require_object(doc, "nodes.json")
    _validate_format_version(root, "nodes.json")
    nodes = root.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise _err("nodes 必须是非空数组", "nodes.json", "nodes")

    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(nodes):
        entry = f"nodes[{index}]"
        node = _require_object(raw, entry)
        node_id = _require_nonempty_str(node.get("node_id"), entry, "node_id")
        if node_id in seen_ids:
            raise _err("node_id 重复", entry, "node_id")
        seen_ids.add(node_id)

        host = node.get("host")
        if host != "127.0.0.1":
            raise _err("host 必须是 127.0.0.1 字面量", entry, "host")

        port = _require_int(node.get("port"), entry, "port", minimum=1024, maximum=65535)
        if port in seen_ports:
            raise _err("回环端口全局重复", entry, "port")
        seen_ports.add(port)

        seed = _require_int(node.get("seed"), entry, "seed", minimum=0, maximum=0xFFFFFFFF)
        fixed_scale = _require_int(node.get("fixed_scale"), entry, "fixed_scale")
        if fixed_scale not in SUPPORTED_SCALES:
            raise _err(
                f"fixed_scale 必须属于 {list(SUPPORTED_SCALES)}",
                entry,
                "fixed_scale",
            )

        weights = node.get("weights_q")
        if not isinstance(weights, list) or not (1 <= len(weights) <= 16):
            raise _err("weights_q 必须是长度 1..16 的整数数组", entry, "weights_q")
        for weight_index, weight in enumerate(weights):
            _require_int(weight, entry, f"weights_q[{weight_index}]")
            if not -(2**31) <= weight <= 2**31 - 1:
                raise _err("权重超出 int32 安全范围", entry, f"weights_q[{weight_index}]")

        bias_q = _require_int(node.get("bias_q", 0), entry, "bias_q")
        concurrency = _require_int(node.get("concurrency"), entry, "concurrency", minimum=0)
        lease_limit = _require_int(node.get("lease_limit"), entry, "lease_limit", minimum=0)
        startup_timeout_ms = node.get("startup_timeout_ms", 5000)
        _require_int(startup_timeout_ms, entry, "startup_timeout_ms", minimum=100, maximum=60000)
        socket_timeout_ms = node.get("socket_timeout_ms", 2000)
        _require_int(socket_timeout_ms, entry, "socket_timeout_ms", minimum=20, maximum=60000)

        normalized.append(
            {
                "node_id": node_id,
                "host": host,
                "port": port,
                "seed": seed,
                "fixed_scale": fixed_scale,
                "weights_q": [int(w) for w in weights],
                "bias_q": bias_q,
                "concurrency": concurrency,
                "lease_limit": lease_limit,
                "startup_timeout_ms": startup_timeout_ms,
                "socket_timeout_ms": socket_timeout_ms,
                "processing_delay_ms": _require_int(
                    node.get("processing_delay_ms", 0), entry, "processing_delay_ms", minimum=0
                ),
            }
        )
    return normalized


def validate_shards(doc: Any, nodes: Iterable[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    root = _require_object(doc, "shards.json")
    _validate_format_version(root, "shards.json")
    shards = root.get("shards")
    if not isinstance(shards, list):
        raise _err("shards 必须是数组", "shards.json", "shards")

    node_ids = {node["node_id"] for node in (nodes or [])}
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(shards):
        entry = f"shards[{index}]"
        shard = _require_object(raw, entry)
        shard_id = _require_nonempty_str(shard.get("shard_id"), entry, "shard_id")
        if shard_id in seen:
            raise _err("shard_id 重复", entry, "shard_id")
        seen.add(shard_id)

        node_id = _require_nonempty_str(shard.get("node_id"), entry, "node_id")
        if node_ids and node_id not in node_ids:
            raise _err("node_id 未在 nodes.json 声明", entry, "node_id")

        upstream = shard.get("upstream", [])
        if not isinstance(upstream, list):
            raise _err("upstream 必须是数组", entry, "upstream")
        upstream_ids: list[str] = []
        upstream_seen: set[str] = set()
        for up_index, upstream_id in enumerate(upstream):
            field = f"upstream[{up_index}]"
            upstream_id = _require_nonempty_str(upstream_id, entry, field)
            if upstream_id in upstream_seen:
                raise _err("upstream 引用重复", entry, field)
            if upstream_id == shard_id:
                raise _err("分片不能依赖自身", entry, field)
            upstream_seen.add(upstream_id)
            upstream_ids.append(upstream_id)

        sample_rows = _require_int(shard.get("sample_rows"), entry, "sample_rows", minimum=0, maximum=1_000_000)
        resource_units = _require_int(shard.get("resource_units"), entry, "resource_units", minimum=0, maximum=10_000_000)
        priority = _require_int(shard.get("priority", 0), entry, "priority", minimum=-10_000_000, maximum=10_000_000)
        batch_count_hint = shard.get("batch_count_hint", 4)
        _require_int(batch_count_hint, entry, "batch_count_hint", minimum=1, maximum=10000)

        normalized.append(
            {
                "shard_id": shard_id,
                "node_id": node_id,
                "upstream": upstream_ids,
                "sample_rows": sample_rows,
                "resource_units": resource_units,
                "priority": priority,
                "batch_count_hint": batch_count_hint,
            }
        )
    return normalized


def _normalize_quota_version(raw: Any, index: int, known_versions: dict[int, dict[str, Any]]) -> dict[str, Any]:
    entry = f"versions[{index}]"
    version_doc = _require_object(raw, entry)
    version = _require_int(version_doc.get("version"), entry, "version", minimum=1)
    parent_version = version_doc.get("parent_version")
    if parent_version is not None:
        parent_version = _require_int(parent_version, entry, "parent_version", minimum=1)
        if parent_version >= version:
            raise _err("parent_version 必须小于 version", entry, "parent_version")
        if parent_version not in known_versions:
            raise _err("parent_version 引用不存在", entry, "parent_version")

    session_scale = version_doc.get("session_scale")
    if session_scale is not None:
        session_scale = _require_int(session_scale, entry, "session_scale")
        if session_scale not in SUPPORTED_SCALES:
            raise _err("session_scale 属于不支持的整数集合", entry, "session_scale")

    global_budget = version_doc.get("global_budget_units")
    if global_budget is not None:
        _require_int(global_budget, entry, "global_budget_units", minimum=0)

    node_upserts = version_doc.get("upsert_node_quotas", {})
    if not isinstance(node_upserts, dict):
        raise _err("upsert_node_quotas 必须是对象", entry, "upsert_node_quotas")
    normalized_nodes: dict[str, dict[str, int]] = {}
    for node_id, quota in node_upserts.items():
        node_entry = f"{entry}.upsert_node_quotas.{node_id}"
        quota_obj = _require_object(quota, node_entry)
        normalized_nodes[node_id] = {
            "resource_units_cap": _require_int(
                quota_obj.get("resource_units_cap", 0), node_entry, "resource_units_cap", minimum=0
            ),
            "concurrency": _require_int(quota_obj.get("concurrency", 0), node_entry, "concurrency", minimum=0),
            "lease_limit": _require_int(quota_obj.get("lease_limit", 0), node_entry, "lease_limit", minimum=0),
        }

    deleted_nodes = version_doc.get("delete_node_quotas", [])
    if not isinstance(deleted_nodes, list) or not all(isinstance(x, str) and x for x in deleted_nodes):
        raise _err("delete_node_quotas 必须是非空字符串数组", entry, "delete_node_quotas")
    if len(deleted_nodes) != len(set(deleted_nodes)):
        raise _err("delete_node_quotas 存在重复", entry, "delete_node_quotas")

    batch_upserts = version_doc.get("upsert_batch_requirements", {})
    if not isinstance(batch_upserts, dict):
        raise _err("upsert_batch_requirements 必须是对象", entry, "upsert_batch_requirements")
    normalized_batches: dict[str, int] = {}
    for key, units in batch_upserts.items():
        normalized_batches[key] = _require_int(
            units, f"{entry}.upsert_batch_requirements.{key}", "value", minimum=0
        )

    deleted_batches = version_doc.get("delete_batch_requirements", [])
    if not isinstance(deleted_batches, list) or not all(isinstance(x, str) and x for x in deleted_batches):
        raise _err("delete_batch_requirements 必须是非空字符串数组", entry, "delete_batch_requirements")
    if len(deleted_batches) != len(set(deleted_batches)):
        raise _err("delete_batch_requirements 存在重复", entry, "delete_batch_requirements")

    return {
        "version": version,
        "parent_version": parent_version,
        "session_scale": session_scale,
        "global_budget_units": global_budget,
        "upsert_node_quotas": normalized_nodes,
        "delete_node_quotas": list(deleted_nodes),
        "upsert_batch_requirements": normalized_batches,
        "delete_batch_requirements": list(deleted_batches),
        "note": version_doc.get("note", ""),
    }


def validate_quota_chain(doc: Any) -> list[dict[str, Any]]:
    root = _require_object(doc, "quota_versions.json")
    _validate_format_version(root, "quota_versions.json")
    versions_raw = root.get("versions")
    if not isinstance(versions_raw, list) or not versions_raw:
        raise _err("versions 必须是非空数组", "quota_versions.json", "versions")

    by_version: dict[int, dict[str, Any]] = {}
    ordered_raw = sorted(enumerate(versions_raw), key=lambda item: item[1].get("version", -1) if isinstance(item[1], dict) else -1)
    for original_index, raw in ordered_raw:
        normalized = _normalize_quota_version(raw, original_index, by_version)
        version = normalized["version"]
        if version in by_version:
            raise _err("配额 version 重复", f"versions[{original_index}]", "version")
        if version == 1:
            if normalized["parent_version"] is not None:
                raise _err("v1 的 parent_version 必须为 null", f"versions[{original_index}]", "parent_version")
            if normalized["session_scale"] is None:
                raise _err("根配额版本必须声明 session_scale", f"versions[{original_index}]", "session_scale")
            if normalized["global_budget_units"] is None:
                raise _err("根配额版本必须声明 global_budget_units", f"versions[{original_index}]", "global_budget_units")
        by_version[version] = normalized

    versions = [by_version[v] for v in sorted(by_version)]
    root_seen = [v for v in versions if v["parent_version"] is None]
    if len(root_seen) != 1 or root_seen[0]["version"] != 1:
        raise _err("必须且只能有 version=1 的根版本", "quota_versions.json", "versions")

    seen: set[int] = set()
    for version_doc in versions:
        chain: set[int] = set()
        current = version_doc["version"]
        while current is not None:
            if current in chain:
                raise _err("配额父版本链存在环", f"versions[{version['version']}]", "parent_version")
            chain.add(current)
            if current in seen:
                break
            current_doc = by_version[current]
            parent = current_doc["parent_version"]
            if parent is None:
                if current != 1:
                    raise _err("只有 v1 可以没有父版本", f"versions[{current}]", "parent_version")
                break
            if parent not in by_version:
                raise _err("parent_version 引用不存在", f"versions[{current}]", "parent_version")
            if parent >= current:
                raise _err("配额版本必须沿父引用单调递增", f"versions[{current}]", "parent_version")
            current = parent
        seen.add(version_doc["version"])
    return versions


def validate_events(doc: Any) -> list[dict[str, Any]]:
    root = _require_object(doc, "events.json")
    _validate_format_version(root, "events.json")
    events = root.get("events")
    if not isinstance(events, list):
        raise _err("events 必须是数组", "events.json", "events")
    seen_seq: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(events):
        entry = f"events[{index}]"
        event = _require_object(raw, entry)
        seq = _require_int(event.get("seq"), entry, "seq", minimum=1)
        if seq in seen_seq:
            raise _err("事件 seq 重复", entry, "seq")
        seen_seq.add(seq)
        event_type = _require_nonempty_str(event.get("type"), entry, "type")
        if event_type not in SUPPORTED_EVENT_TYPES:
            raise _err(f"type 必须是 {sorted(SUPPORTED_EVENT_TYPES)} 之一", entry, "type")
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            raise _err("payload 必须是对象", entry, "payload")
        at = event.get("at", "")
        if not isinstance(at, str):
            raise _err("at 必须是 ISO 时间字符串", entry, "at")
        normalized.append({"seq": seq, "type": event_type, "at": at, "payload": payload})

    seqs = [event["seq"] for event in normalized]
    if seqs != sorted(seqs):
        raise _err("事件流必须按 seq 单调排列", "events.json", "events")
    return normalized


@dataclass(frozen=True)
class SpecBundle:
    nodes: list[dict[str, Any]]
    shards: list[dict[str, Any]]
    quota_versions: list[dict[str, Any]]
    events: list[dict[str, Any]]

    @property
    def node_by_id(self) -> dict[str, dict[str, Any]]:
        return {node["node_id"]: node for node in self.nodes}

    @property
    def shard_by_id(self) -> dict[str, dict[str, Any]]:
        return {shard["shard_id"]: shard for shard in self.shards}

    @property
    def quota_by_version(self) -> dict[int, dict[str, Any]]:
        return {item["version"]: item for item in self.quota_versions}


def load_spec_bundle(root: str | Path) -> SpecBundle:
    root = Path(root)
    nodes_doc = load_json_file(root / "declarations" / "nodes.json")
    nodes = validate_nodes(nodes_doc)
    shards_doc = load_json_file(root / "declarations" / "shards.json")
    shards = validate_shards(shards_doc, nodes)
    quota_doc = load_json_file(root / "declarations" / "quota_versions.json")
    quota_versions = validate_quota_chain(quota_doc)
    events_doc = load_json_file(root / "declarations" / "events.json")
    events = validate_events(events_doc)
    return SpecBundle(nodes=nodes, shards=shards, quota_versions=quota_versions, events=events)
