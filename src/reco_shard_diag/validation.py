from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .errors import BizError, make_error
from .models import Event, ProjectConfig


@dataclass
class ValidationReport:
    errors: list[BizError] = field(default_factory=list)
    warnings: list[BizError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, error: BizError) -> None:
        self.errors.append(error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


def _check_unique(
    values: Iterable[tuple[str, str]],
    *,
    code: str,
    object_type: str,
    field_name: str,
    report: ValidationReport,
) -> None:
    seen: dict[str, str] = {}
    for value, object_id in values:
        if value in seen:
            report.add(
                make_error(
                    code,
                    object_type,
                    object_id,
                    field_name,
                    f"{field_name} 与 {seen[value]} 重复",
                )
            )
        else:
            seen[value] = object_id


def _check_dependency_cycles(shards, report: ValidationReport) -> None:
    upstream = {shard.shard_id: list(shard.upstream) for shard in shards}
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for parent in upstream.get(node, []):
            if parent not in upstream:
                continue
            if state.get(parent, 0) == 0:
                visit(parent)
            elif state.get(parent) == 1:
                cycle_start = stack.index(parent)
                cycle = stack[cycle_start:] + [parent]
                report.add(
                    make_error(
                        "E_DEP_CYCLE",
                        "shard",
                        node,
                        "upstream",
                        "上游依赖成环: " + " -> ".join(cycle),
                    )
                )
        stack.pop()
        state[node] = 2

    for shard_id in sorted(upstream):
        if state.get(shard_id, 0) == 0:
            visit(shard_id)


def _check_quota_chain(quotas, report: ValidationReport) -> None:
    versions = {quota.version: quota for quota in quotas}
    previous: int | None = None
    for quota in sorted(quotas, key=lambda item: item.version):
        if previous is not None and quota.version <= previous:
            report.add(
                make_error(
                    "E_QUOTA_VERSION_NOT_MONOTONIC",
                    "quota_version",
                    str(quota.version),
                    "version",
                    f"版本号必须严格递增，前一个为 {previous}",
                )
            )
        previous = quota.version
        if quota.parent_version is not None:
            if quota.parent_version not in versions:
                report.add(
                    make_error(
                        "E_QUOTA_PARENT_MISSING",
                        "quota_version",
                        str(quota.version),
                        "parent_version",
                        f"父版本 {quota.parent_version} 不存在",
                    )
                )
            elif quota.parent_version >= quota.version:
                report.add(
                    make_error(
                        "E_QUOTA_PARENT_NOT_EARLIER",
                        "quota_version",
                        str(quota.version),
                        "parent_version",
                        "父版本号必须小于当前版本号",
                    )
                )
        seen = {quota.version}
        cursor = quota.parent_version
        while cursor is not None and cursor in versions:
            if cursor in seen:
                report.add(
                    make_error(
                        "E_QUOTA_CYCLE",
                        "quota_version",
                        str(quota.version),
                        "parent_version",
                        "父版本链存在环",
                    )
                )
                break
            seen.add(cursor)
            cursor = versions[cursor].parent_version


def _check_events(events: list[Event], config: ProjectConfig, report: ValidationReport) -> None:
    previous: int | None = None
    known_types = {
        "batch_append",
        "batch_replace",
        "batch_delete",
        "shard_replace",
        "shard_delete",
        "dependency_update",
        "quota_version_add",
        "quota_rollback",
        "node_adjust",
        "width_switch",
    }
    shard_ids = {shard.shard_id for shard in config.shards}
    node_ids = {node.node_id for node in config.nodes}
    quota_versions = {quota.version for quota in config.quota_versions}
    for event in events:
        object_id = str(event.seq)
        if previous is not None and event.seq <= previous:
            report.add(
                make_error(
                    "E_EVENT_SEQ_NOT_STRICT",
                    "event",
                    object_id,
                    "seq",
                    f"到达序号必须严格递增，前一个为 {previous}",
                )
            )
        previous = event.seq
        if event.event_type not in known_types:
            report.add(
                make_error("E_EVENT_TYPE_UNKNOWN", "event", object_id, "type", "未知事件类型")
            )
            continue
        payload = event.payload
        shard_ref_events = {
            "batch_append",
            "batch_replace",
            "batch_delete",
            "dependency_update",
            "shard_delete",
            "shard_replace",
        }
        if event.event_type in shard_ref_events:
            shard_id = payload.get("shard_id")
            if shard_id not in shard_ids:
                if event.event_type == "shard_replace":
                    replacement = payload.get("shard")
                    if isinstance(replacement, dict) and replacement.get("shard_id") == shard_id:
                        pass
                    else:
                        report.add(
                            make_error(
                                "E_EVENT_REF_MISSING",
                                "event",
                                object_id,
                                "payload.shard_id",
                                "分片不存在",
                            )
                        )
                else:
                    report.add(
                        make_error(
                            "E_EVENT_REF_MISSING",
                            "event",
                            object_id,
                            "payload.shard_id",
                            "分片不存在",
                        )
                    )
        if event.event_type == "dependency_update":
            for dependency in payload.get("upstream", []):
                if dependency not in shard_ids:
                    report.add(
                        make_error(
                            "E_EVENT_REF_MISSING",
                            "event",
                            object_id,
                            "payload.upstream",
                            f"上游 {dependency} 不存在",
                        )
                    )
        if event.event_type == "quota_version_add":
            quota = payload.get("quota")
            if not isinstance(quota, dict):
                report.add(
                    make_error(
                        "E_CONFIG_SCHEMA", "event", object_id, "payload.quota", "配额版本必须为对象"
                    )
                )
            else:
                version = quota.get("version")
                parent = quota.get("parent_version")
                if version in quota_versions:
                    report.add(
                        make_error(
                            "E_EVENT_REF_MISSING",
                            "event",
                            object_id,
                            "payload.quota.version",
                            "配额版本已存在",
                        )
                    )
                if parent is not None and parent not in quota_versions:
                    report.add(
                        make_error(
                            "E_QUOTA_PARENT_MISSING",
                            "quota_version",
                            str(version),
                            "parent_version",
                            f"父版本 {parent} 不存在",
                        )
                    )
        if event.event_type == "quota_rollback":
            target = payload.get("target_version")
            if target not in quota_versions:
                report.add(
                    make_error(
                        "E_EVENT_REF_MISSING",
                        "event",
                        object_id,
                        "payload.target_version",
                        "回滚目标版本不存在",
                    )
                )
        if event.event_type == "node_adjust":
            node_id = payload.get("node_id")
            if node_id not in node_ids:
                report.add(
                    make_error(
                        "E_EVENT_REF_MISSING", "event", object_id, "payload.node_id", "节点不存在"
                    )
                )
            for field_name in ("capacity_slots", "lease_limit"):
                if field_name in payload:
                    value = payload[field_name]
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        code = (
                            "E_NODE_CAPACITY_INVALID"
                            if field_name == "capacity_slots"
                            else "E_NODE_LEASE_LIMIT_INVALID"
                        )
                        report.add(
                            make_error(
                                code,
                                "event",
                                object_id,
                                f"payload.{field_name}",
                                f"{field_name} 必须为正整数",
                            )
                        )
        if event.event_type == "width_switch":
            width = payload.get("width")
            if isinstance(width, bool) or not isinstance(width, int) or not 2 <= width <= 62:
                report.add(
                    make_error(
                        "E_CONFIG_SCHEMA", "event", object_id, "payload.width", "位宽必须为 2..62"
                    )
                )


def validate_config(config: ProjectConfig) -> ValidationReport:
    report = ValidationReport()
    _check_unique(
        ((node.node_id, node.node_id) for node in config.nodes),
        code="E_NODE_ID_DUPLICATE",
        object_type="node",
        field_name="node_id",
        report=report,
    )
    _check_unique(
        ((str(node.port), node.node_id) for node in config.nodes),
        code="E_NODE_PORT_DUPLICATE",
        object_type="node",
        field_name="port",
        report=report,
    )
    _check_unique(
        ((shard.shard_id, shard.shard_id) for shard in config.shards),
        code="E_SHARD_ID_DUPLICATE",
        object_type="shard",
        field_name="shard_id",
        report=report,
    )
    node_ids = {node.node_id for node in config.nodes}
    quota_versions = {quota.version for quota in config.quota_versions}
    shard_ids = {shard.shard_id for shard in config.shards}
    if config.active_quota_version not in quota_versions:
        report.add(
            make_error(
                "E_QUOTA_ACTIVE_MISSING",
                "config",
                "root",
                "active_quota_version",
                "当前配额版本不存在",
            )
        )
    for shard in config.shards:
        if shard.node_id not in node_ids:
            report.add(
                make_error(
                    "E_SHARD_NODE_MISSING",
                    "shard",
                    shard.shard_id,
                    "node_id",
                    f"节点 {shard.node_id} 不存在",
                )
            )
        for dependency in shard.upstream:
            if dependency not in shard_ids:
                report.add(
                    make_error(
                        "E_DEP_MISSING",
                        "shard",
                        shard.shard_id,
                        "upstream",
                        f"上游 {dependency} 不存在",
                    )
                )
            elif dependency == shard.shard_id:
                report.add(
                    make_error(
                        "E_DEP_CYCLE",
                        "shard",
                        shard.shard_id,
                        "upstream",
                        "分片不得依赖自身",
                    )
                )
        for batch in shard.batches:
            for group in batch.group_keys:
                if group not in shard.groups:
                    report.add(
                        make_error(
                            "E_CONFIG_SCHEMA",
                            "batch",
                            f"{shard.shard_id}/{batch.batch_id}",
                            "group_keys",
                            f"分组 {group} 未在分片声明",
                        )
                    )
    _check_dependency_cycles(config.shards, report)
    _check_quota_chain(config.quota_versions, report)
    _check_events(config.events, config, report)
    return report
