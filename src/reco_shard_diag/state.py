from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import fingerprint
from .errors import BizError, make_error
from .models import BatchSpec, Event, NodeSpec, ProjectConfig, QuotaVersion, ShardSpec


def _clone_config(config: ProjectConfig) -> ProjectConfig:
    raw = config.to_dict()
    raw["events"] = []
    return ProjectConfig.from_raw(raw)


def _batch_from_payload(payload: dict[str, Any], shard_id: str) -> BatchSpec:
    raw = payload.get("batch", payload)
    batch, errors = BatchSpec.from_raw(raw, shard_id, 0)
    if batch is None or errors:
        raise ValueError(errors[0].message if errors else "batch invalid")
    return batch


@dataclass
class MaterializedState:
    base_config: ProjectConfig
    nodes: dict[str, NodeSpec]
    quotas: dict[int, QuotaVersion]
    shards: dict[str, ShardSpec]
    active_quota_version: int
    width: int
    shard_revisions: dict[str, int] = field(default_factory=dict)
    capacity_overrides: dict[str, int] = field(default_factory=dict)
    lease_overrides: dict[str, int] = field(default_factory=dict)
    applied_events: list[Event] = field(default_factory=list)
    errors: list[BizError] = field(default_factory=list)

    @property
    def active_quota(self) -> QuotaVersion:
        return self.quotas[self.active_quota_version]

    def effective_node(self, node_id: str) -> NodeSpec:
        base = self.nodes[node_id]
        quota = self.active_quota
        capacity = self.capacity_overrides.get(
            node_id, quota.capacity_slots.get(node_id, base.capacity_slots)
        )
        lease_limit = self.lease_overrides.get(
            node_id, quota.lease_limits.get(node_id, base.lease_limit)
        )
        return NodeSpec(
            node_id=base.node_id,
            host=base.host,
            port=base.port,
            capacity_slots=capacity,
            lease_limit=lease_limit,
        )

    def samples(self, shard_id: str) -> list[dict[str, Any]]:
        from .generator import generate_shard_samples

        return generate_shard_samples(self.base_config.seed, self.shards[shard_id])

    def sorted_shards(self) -> list[ShardSpec]:
        return [self.shards[key] for key in sorted(self.shards)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_version": self.base_config.global_version,
            "seed": self.base_config.seed,
            "resource_budget": self.base_config.resource_budget,
            "max_batch_points": self.base_config.max_batch_points,
            "reference_width": self.base_config.reference_width,
            "active_quota_version": self.active_quota_version,
            "width": self.width,
            "active_quota": self.active_quota.to_dict(),
            "nodes": {
                key: self.effective_node(key).to_dict() for key in sorted(self.nodes)
            },
            "quotas": {
                str(key): self.quotas[key].to_dict() for key in sorted(self.quotas)
            },
            "shards": {
                key: self.shards[key].to_dict() for key in sorted(self.shards)
            },
            "shard_revisions": {
                key: self.shard_revisions[key] for key in sorted(self.shard_revisions)
            },
            "capacity_overrides": dict(sorted(self.capacity_overrides.items())),
            "lease_overrides": dict(sorted(self.lease_overrides.items())),
            "events": [event.to_dict() for event in self.applied_events],
            "errors": [error.to_dict() for error in self.errors],
        }

    @property
    def state_fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    @property
    def base_config_hash(self) -> str:
        return self.base_config.base_config_hash


def _topology_errors(shards: dict[str, ShardSpec]) -> list[BizError]:
    errors: list[BizError] = []
    for shard in shards.values():
        for dependency in shard.upstream:
            if dependency not in shards:
                errors.append(
                    make_error(
                        "E_DEP_MISSING",
                        "shard",
                        shard.shard_id,
                        "upstream",
                        f"上游 {dependency} 不存在",
                    )
                )
    state: dict[str, int] = {}

    def visit(shard_id: str, stack: list[str]) -> None:
        state[shard_id] = 1
        stack.append(shard_id)
        for dependency in shards[shard_id].upstream:
            if dependency not in shards:
                continue
            if state.get(dependency, 0) == 0:
                visit(dependency, stack)
            elif state.get(dependency) == 1:
                start = stack.index(dependency)
                cycle = stack[start:] + [dependency]
                errors.append(
                    make_error(
                        "E_DEP_CYCLE",
                        "shard",
                        shard_id,
                        "upstream",
                        "上游依赖成环: " + " -> ".join(cycle),
                    )
                )
        stack.pop()
        state[shard_id] = 2

    for shard_id in sorted(shards):
        if state.get(shard_id, 0) == 0:
            visit(shard_id, [])
    return errors


def materialize(config: ProjectConfig) -> MaterializedState:
    cloned = _clone_config(config)
    state = MaterializedState(
        base_config=cloned,
        nodes={node.node_id: node for node in cloned.nodes},
        quotas={quota.version: quota for quota in cloned.quota_versions},
        shards={shard.shard_id: shard for shard in cloned.shards},
        active_quota_version=cloned.active_quota_version,
        width=cloned.quota_versions[0].width if cloned.quota_versions else 16,
        shard_revisions={shard.shard_id: 1 for shard in cloned.shards},
    )
    if state.active_quota_version in state.quotas:
        state.width = state.quotas[state.active_quota_version].width

    events = sorted(config.events, key=lambda event: event.seq)
    previous_seq: int | None = None
    for event in events:
        if previous_seq is not None and event.seq <= previous_seq:
            state.errors.append(
                make_error(
                    "E_EVENT_SEQ_NOT_STRICT",
                    "event",
                    str(event.seq),
                    "seq",
                    "事件序号必须严格递增",
                )
            )
            continue
        previous_seq = event.seq
        payload = event.payload
        try:
            if event.event_type in {"batch_append", "batch_replace"}:
                shard_id = str(payload.get("shard_id", ""))
                if shard_id not in state.shards:
                    raise ValueError("shard missing")
                batch = _batch_from_payload(payload, shard_id)
                shard = state.shards[shard_id]
                batches = [item for item in shard.batches if item.batch_id != batch.batch_id]
                batches.append(batch)
                shard.batches = sorted(batches, key=lambda item: item.batch_id)
                state.shard_revisions[shard_id] += 1
            elif event.event_type == "batch_delete":
                shard_id = str(payload.get("shard_id", ""))
                batch_id = str(payload.get("batch_id", ""))
                if shard_id not in state.shards:
                    raise ValueError("shard missing")
                shard = state.shards[shard_id]
                shard.batches = [item for item in shard.batches if item.batch_id != batch_id]
                state.shard_revisions[shard_id] += 1
            elif event.event_type == "shard_replace":
                shard_raw = payload.get("shard")
                if not isinstance(shard_raw, dict):
                    raise ValueError("shard missing")
                shard, errors = ShardSpec.from_raw(shard_raw, 0)
                if shard is None or errors:
                    raise ValueError(errors[0].message if errors else "shard invalid")
                old_id = str(payload.get("shard_id", shard.shard_id))
                if old_id != shard.shard_id:
                    state.shards.pop(old_id, None)
                    state.shard_revisions.pop(old_id, None)
                state.shards[shard.shard_id] = shard
                previous_revision = state.shard_revisions.get(shard.shard_id, 1)
                state.shard_revisions[shard.shard_id] = previous_revision + 1
            elif event.event_type == "shard_delete":
                shard_id = str(payload.get("shard_id", ""))
                state.shards.pop(shard_id, None)
                state.shard_revisions.pop(shard_id, None)
            elif event.event_type == "dependency_update":
                shard_id = str(payload.get("shard_id", ""))
                if shard_id not in state.shards:
                    raise ValueError("shard missing")
                upstream = payload.get("upstream")
                if not isinstance(upstream, list) or any(
                    not isinstance(item, str) for item in upstream
                ):
                    raise ValueError("upstream invalid")
                state.shards[shard_id].upstream = list(upstream)
                state.shard_revisions[shard_id] += 1
            elif event.event_type == "quota_version_add":
                quota_raw = payload.get("quota")
                if not isinstance(quota_raw, dict):
                    raise ValueError("quota invalid")
                quota, errors = QuotaVersion.from_raw(quota_raw, 0)
                if quota is None or errors:
                    raise ValueError(errors[0].message if errors else "quota invalid")
                state.quotas[quota.version] = quota
                if payload.get("activate", False):
                    state.active_quota_version = quota.version
                    state.width = quota.width
            elif event.event_type == "quota_rollback":
                target = payload.get("target_version")
                if not isinstance(target, int) or target not in state.quotas:
                    raise ValueError("rollback target missing")
                state.active_quota_version = target
                state.width = state.quotas[target].width
            elif event.event_type == "node_adjust":
                node_id = str(payload.get("node_id", ""))
                if node_id not in state.nodes:
                    raise ValueError("node missing")
                node = state.nodes[node_id]
                capacity = payload.get("capacity_slots", node.capacity_slots)
                lease_limit = payload.get("lease_limit", node.lease_limit)
                if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
                    raise ValueError("capacity invalid")
                if (
                    not isinstance(lease_limit, int)
                    or isinstance(lease_limit, bool)
                    or lease_limit <= 0
                ):
                    raise ValueError("lease limit invalid")
                state.nodes[node_id] = NodeSpec(
                    node_id=node_id,
                    host=node.host,
                    port=node.port,
                    capacity_slots=capacity,
                    lease_limit=lease_limit,
                )
                state.capacity_overrides[node_id] = capacity
                state.lease_overrides[node_id] = lease_limit
            elif event.event_type == "width_switch":
                width = payload.get("width")
                if not isinstance(width, int) or isinstance(width, bool) or not 2 <= width <= 62:
                    raise ValueError("width invalid")
                state.width = width
            else:
                state.errors.append(
                    make_error(
                        "E_EVENT_TYPE_UNKNOWN",
                        "event",
                        str(event.seq),
                        "type",
                        f"未知事件类型 {event.event_type}",
                    )
                )
                continue
            state.applied_events.append(event)
        except ValueError as exc:
            state.errors.append(
                make_error(
                    "E_EVENT_REF_MISSING",
                    "event",
                    str(event.seq),
                    "payload",
                    str(exc),
                )
            )

    if state.active_quota_version not in state.quotas:
        state.errors.append(
            make_error(
                "E_QUOTA_ACTIVE_MISSING",
                "config",
                "root",
                "active_quota_version",
                "当前配额版本不存在",
            )
        )
    else:
        quota = state.active_quota
        for shard in state.shards.values():
            if shard.node_id not in state.nodes:
                state.errors.append(
                    make_error(
                        "E_SHARD_NODE_MISSING",
                        "shard",
                        shard.shard_id,
                        "node_id",
                        f"节点 {shard.node_id} 不存在",
                    )
                )
        for node in state.nodes.values():
            capacity = state.capacity_overrides.get(
                node.node_id,
                quota.capacity_slots.get(node.node_id, node.capacity_slots),
            )
            limit = state.lease_overrides.get(
                node.node_id, quota.lease_limits.get(node.node_id, node.lease_limit)
            )
            if capacity <= 0:
                state.errors.append(
                    make_error(
                        "E_NODE_CAPACITY_INVALID",
                        "node",
                        node.node_id,
                        "capacity_slots",
                        "有效容量槽位必须为正整数",
                    )
                )
            if limit <= 0:
                state.errors.append(
                    make_error(
                        "E_NODE_LEASE_LIMIT_INVALID",
                        "node",
                        node.node_id,
                        "lease_limit",
                        "有效租约上限必须为正整数",
                    )
                )
    state.errors.extend(_topology_errors(state.shards))
    return state


def replay_events(config: ProjectConfig, events: list[Event]) -> MaterializedState:
    merged_raw = config.to_dict()
    merged_raw["events"] = [event.to_dict() for event in events]
    return materialize(ProjectConfig.from_raw(merged_raw))
