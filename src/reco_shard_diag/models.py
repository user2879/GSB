from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import fingerprint
from .errors import BizError, BizErrorException, make_error


def _int_field(
    raw: dict[str, Any],
    key: str,
    *,
    object_type: str,
    object_id: str,
    errors: list[BizError],
) -> int | None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(
            make_error("E_CONFIG_SCHEMA", object_type, object_id, key, f"{key} 必须为整数")
        )
        return None
    return value


def _str_field(
    raw: dict[str, Any],
    key: str,
    *,
    object_type: str,
    object_id: str,
    errors: list[BizError],
) -> str | None:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        errors.append(
            make_error("E_CONFIG_SCHEMA", object_type, object_id, key, f"{key} 必须为非空字符串")
        )
        return None
    return value


@dataclass
class NodeSpec:
    node_id: str
    host: str
    port: int
    capacity_slots: int
    lease_limit: int

    @classmethod
    def from_raw(
        cls, raw: dict[str, Any], index: int
    ) -> tuple["NodeSpec | None", list[BizError]]:
        errors: list[BizError] = []
        object_id = str(raw.get("node_id", f"nodes[{index}]"))
        node_id = _str_field(
            raw, "node_id", object_type="node", object_id=object_id, errors=errors
        )
        host = str(raw.get("host", ""))
        port = _int_field(raw, "port", object_type="node", object_id=object_id, errors=errors)
        capacity = _int_field(
            raw, "capacity_slots", object_type="node", object_id=object_id, errors=errors
        )
        lease_limit = _int_field(
            raw, "lease_limit", object_type="node", object_id=object_id, errors=errors
        )
        if host != "127.0.0.1":
            errors.append(
                make_error("E_NODE_PORT_INVALID", "node", object_id, "host", "节点必须绑定 127.0.0.1")
            )
        if port is not None and not (1 <= port <= 65535):
            errors.append(
                make_error("E_NODE_PORT_INVALID", "node", object_id, "port", "端口必须在 1..65535")
            )
        if capacity is not None and capacity <= 0:
            errors.append(
                make_error(
                    "E_NODE_CAPACITY_INVALID",
                    "node",
                    object_id,
                    "capacity_slots",
                    "容量槽位必须为正整数",
                )
            )
        if lease_limit is not None and lease_limit <= 0:
            errors.append(
                make_error(
                    "E_NODE_LEASE_LIMIT_INVALID",
                    "node",
                    object_id,
                    "lease_limit",
                    "每节点租约上限必须为正整数",
                )
            )
        if errors:
            return None, errors
        return (
            cls(
                node_id=node_id,
                host=host,
                port=port,
                capacity_slots=capacity,
                lease_limit=lease_limit,
            ),
            [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "capacity_slots": self.capacity_slots,
            "lease_limit": self.lease_limit,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass
class QuotaVersion:
    version: int
    parent_version: int | None
    capacity_slots: dict[str, int]
    lease_limits: dict[str, int]
    soft_watermark_bp: int
    hard_watermark_bp: int
    width: int
    points_per_resource_unit: int

    @classmethod
    def from_raw(
        cls, raw: dict[str, Any], index: int
    ) -> tuple["QuotaVersion | None", list[BizError]]:
        errors: list[BizError] = []
        object_id = str(raw.get("version", f"quota_versions[{index}]"))
        version = _int_field(
            raw, "version", object_type="quota_version", object_id=object_id, errors=errors
        )
        parent_raw = raw.get("parent_version")
        if parent_raw is not None and (
            isinstance(parent_raw, bool) or not isinstance(parent_raw, int)
        ):
            errors.append(
                make_error(
                    "E_CONFIG_SCHEMA",
                    "quota_version",
                    object_id,
                    "parent_version",
                    "parent_version 必须为整数或 null",
                )
            )
        capacity = raw.get("capacity_slots", {})
        limit = raw.get("lease_limits", {})
        if not isinstance(capacity, dict) or not isinstance(limit, dict):
            errors.append(
                make_error(
                    "E_CONFIG_SCHEMA",
                    "quota_version",
                    object_id,
                    "capacity_slots/lease_limits",
                    "容量与租约上限必须为对象",
                )
            )
            capacity, limit = {}, {}
        soft_bp = raw.get("soft_watermark_bp", 7000)
        hard_bp = raw.get("hard_watermark_bp", 10000)
        width = raw.get("width", 16)
        points_per_unit = raw.get("points_per_resource_unit", 10)
        for key, value in (
            ("soft_watermark_bp", soft_bp),
            ("hard_watermark_bp", hard_bp),
            ("width", width),
            ("points_per_resource_unit", points_per_unit),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(
                    make_error(
                        "E_CONFIG_SCHEMA", "quota_version", object_id, key, f"{key} 必须为整数"
                    )
                )
        if not (0 <= soft_bp <= 10000 and 0 <= hard_bp <= 10000):
            errors.append(
                make_error(
                    "E_QUOTA_WATERMARK_INVALID",
                    "quota_version",
                    object_id,
                    "watermark_bp",
                    "水印万分比必须在 0..10000",
                )
            )
        if soft_bp > hard_bp:
            errors.append(
                make_error(
                    "E_QUOTA_WATERMARK_INVALID",
                    "quota_version",
                    object_id,
                    "soft_watermark_bp",
                    "软水印不得高于硬水印",
                )
            )
        if isinstance(width, int) and not 2 <= width <= 62:
            errors.append(
                make_error(
                    "E_QUOTA_WATERMARK_INVALID",
                    "quota_version",
                    object_id,
                    "width",
                    "定点位宽必须在 2..62",
                )
            )
        if isinstance(points_per_unit, int) and points_per_unit <= 0:
            errors.append(
                make_error(
                    "E_QUOTA_WATERMARK_INVALID",
                    "quota_version",
                    object_id,
                    "points_per_resource_unit",
                    "每资源单位点数必须为正整数",
                )
            )
        for mapping_name, mapping in (("capacity_slots", capacity), ("lease_limits", limit)):
            for key, value in mapping.items():
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    errors.append(
                        make_error(
                            "E_NODE_CAPACITY_INVALID"
                            if mapping_name == "capacity_slots"
                            else "E_NODE_LEASE_LIMIT_INVALID",
                            "quota_version",
                            object_id,
                            f"{mapping_name}.{key}",
                            "声明值必须为正整数",
                        )
                    )
        if errors:
            return None, errors
        return (
            cls(
                version=version,
                parent_version=parent_raw,
                capacity_slots={str(key): int(value) for key, value in capacity.items()},
                lease_limits={str(key): int(value) for key, value in limit.items()},
                soft_watermark_bp=soft_bp,
                hard_watermark_bp=hard_bp,
                width=width,
                points_per_resource_unit=points_per_unit,
            ),
            [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "capacity_slots": dict(sorted(self.capacity_slots.items())),
            "lease_limits": dict(sorted(self.lease_limits.items())),
            "soft_watermark_bp": self.soft_watermark_bp,
            "hard_watermark_bp": self.hard_watermark_bp,
            "width": self.width,
            "points_per_resource_unit": self.points_per_resource_unit,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass
class BatchSpec:
    batch_id: str
    sample_count: int
    value_bound: int = 900
    group_keys: list[str] = field(default_factory=lambda: ["g0", "g1", "g2"])

    @classmethod
    def from_raw(
        cls, raw: dict[str, Any], shard_id: str, index: int
    ) -> tuple["BatchSpec | None", list[BizError]]:
        errors: list[BizError] = []
        object_id = f"{shard_id}/{raw.get('batch_id', f'batches[{index}]')}"
        batch_id = _str_field(
            raw, "batch_id", object_type="batch", object_id=object_id, errors=errors
        )
        sample_count = _int_field(
            raw, "sample_count", object_type="batch", object_id=object_id, errors=errors
        )
        value_bound = _int_field(
            raw, "value_bound", object_type="batch", object_id=object_id, errors=errors
        )
        group_keys = raw.get("group_keys", ["g0", "g1", "g2"])
        if sample_count is not None and sample_count <= 0:
            errors.append(
                make_error(
                    "E_CONFIG_SCHEMA", "batch", object_id, "sample_count", "样本数必须为正整数"
                )
            )
        if value_bound is not None and value_bound <= 0:
            errors.append(
                make_error("E_CONFIG_SCHEMA", "batch", object_id, "value_bound", "值域必须为正")
            )
        if not isinstance(group_keys, list) or not group_keys:
            errors.append(
                make_error(
                    "E_CONFIG_SCHEMA", "batch", object_id, "group_keys", "分组键必须为非空数组"
                )
            )
            group_keys = []
        if errors:
            return None, errors
        return (
            cls(
                batch_id=batch_id,
                sample_count=sample_count,
                value_bound=value_bound,
                group_keys=[str(key) for key in group_keys],
            ),
            [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "sample_count": self.sample_count,
            "value_bound": self.value_bound,
            "group_keys": list(self.group_keys),
        }


@dataclass
class ShardSpec:
    shard_id: str
    node_id: str
    upstream: list[str]
    dimensions: int
    groups: list[str]
    batches: list[BatchSpec]

    @classmethod
    def from_raw(
        cls, raw: dict[str, Any], index: int
    ) -> tuple["ShardSpec | None", list[BizError]]:
        errors: list[BizError] = []
        object_id = str(raw.get("shard_id", f"shards[{index}]"))
        shard_id = _str_field(
            raw, "shard_id", object_type="shard", object_id=object_id, errors=errors
        )
        node_id = _str_field(
            raw, "node_id", object_type="shard", object_id=object_id, errors=errors
        )
        dimensions = _int_field(
            raw, "dimensions", object_type="shard", object_id=object_id, errors=errors
        )
        upstream = raw.get("upstream", [])
        groups = raw.get("groups", ["g0", "g1", "g2"])
        if not isinstance(upstream, list) or any(
            not isinstance(item, str) for item in upstream
        ):
            errors.append(
                make_error(
                    "E_CONFIG_SCHEMA", "shard", object_id, "upstream", "上游必须为字符串数组"
                )
            )
            upstream = []
        if not isinstance(groups, list) or not groups:
            errors.append(
                make_error("E_CONFIG_SCHEMA", "shard", object_id, "groups", "分组必须为非空数组")
            )
            groups = []
        if dimensions is not None and dimensions <= 0:
            errors.append(
                make_error(
                    "E_CONFIG_SCHEMA", "shard", object_id, "dimensions", "维度必须为正整数"
                )
            )
        parsed_batches: list[BatchSpec] = []
        raw_batches = raw.get("batches", [])
        if not isinstance(raw_batches, list):
            errors.append(
                make_error("E_CONFIG_SCHEMA", "shard", object_id, "batches", "批次必须为数组")
            )
            raw_batches = []
        for batch_index, batch_raw in enumerate(raw_batches):
            if not isinstance(batch_raw, dict):
                errors.append(
                    make_error(
                        "E_CONFIG_SCHEMA",
                        "shard",
                        object_id,
                        f"batches[{batch_index}]",
                        "批次必须为对象",
                    )
                )
                continue
            batch, batch_errors = BatchSpec.from_raw(batch_raw, object_id, batch_index)
            errors.extend(batch_errors)
            if batch is not None:
                parsed_batches.append(batch)
        seen_batches: set[str] = set()
        for batch in parsed_batches:
            if batch.batch_id in seen_batches:
                errors.append(
                    make_error(
                        "E_BATCH_ID_DUPLICATE",
                        "batch",
                        f"{object_id}/{batch.batch_id}",
                        "batch_id",
                        "分片内批次标识必须唯一",
                    )
                )
            seen_batches.add(batch.batch_id)
        if errors:
            return None, errors
        return (
            cls(
                shard_id=shard_id,
                node_id=node_id,
                upstream=list(upstream),
                dimensions=dimensions,
                groups=[str(group) for group in groups],
                batches=parsed_batches,
            ),
            [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "upstream": list(self.upstream),
            "dimensions": self.dimensions,
            "groups": list(self.groups),
            "batches": [batch.to_dict() for batch in self.batches],
        }


@dataclass
class Event:
    seq: int
    event_type: str
    payload: dict[str, Any]

    @classmethod
    def from_raw(
        cls, raw: dict[str, Any], index: int
    ) -> tuple["Event | None", list[BizError]]:
        errors: list[BizError] = []
        object_id = str(raw.get("seq", f"events[{index}]"))
        seq = _int_field(
            raw, "seq", object_type="event", object_id=object_id, errors=errors
        )
        event_type = _str_field(
            raw, "type", object_type="event", object_id=object_id, errors=errors
        )
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            errors.append(
                make_error("E_CONFIG_SCHEMA", "event", object_id, "payload", "payload 必须为对象")
            )
        if errors:
            return None, errors
        return cls(seq=seq, event_type=event_type, payload=payload), []

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "type": self.event_type, "payload": self.payload}


@dataclass
class ProjectConfig:
    global_version: int
    seed: int
    resource_budget: int
    max_batch_points: int
    reference_width: int
    active_quota_version: int
    nodes: list[NodeSpec]
    quota_versions: list[QuotaVersion]
    shards: list[ShardSpec]
    events: list[Event] = field(default_factory=list)
    injections: list[dict[str, Any]] = field(default_factory=list)
    lease_ttl_ms: int = 3000

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "ProjectConfig":
        errors: list[BizError] = []

        def required_int(key: str, default: int | None = None) -> int:
            value = raw.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(
                    make_error("E_CONFIG_SCHEMA", "config", "root", key, f"{key} 必须为整数")
                )
                return 0
            return value

        global_version = required_int("global_version")
        seed = required_int("seed")
        resource_budget = required_int("resource_budget")
        max_batch_points = required_int("max_batch_points")
        reference_width = required_int("reference_width", 17)
        active_version = required_int("active_quota_version")
        lease_ttl_ms = required_int("lease_ttl_ms", 3000)
        if resource_budget <= 0:
            errors.append(
                make_error(
                    "E_CONFIG_SCHEMA", "config", "root", "resource_budget", "预算必须为正整数"
                )
            )
        if max_batch_points <= 0:
            errors.append(
                make_error(
                    "E_CONFIG_SCHEMA", "config", "root", "max_batch_points", "批次点数必须为正整数"
                )
            )
        if lease_ttl_ms <= 0:
            errors.append(
                make_error(
                    "E_CONFIG_SCHEMA",
                    "config",
                    "root",
                    "lease_ttl_ms",
                    "租约 TTL 必须为正整数",
                )
            )
        nodes: list[NodeSpec] = []
        for index, node_raw in enumerate(raw.get("nodes", [])):
            if not isinstance(node_raw, dict):
                errors.append(
                    make_error("E_CONFIG_SCHEMA", "node", str(index), "nodes", "节点必须为对象")
                )
                continue
            node, node_errors = NodeSpec.from_raw(node_raw, index)
            errors.extend(node_errors)
            if node is not None:
                nodes.append(node)
        quotas: list[QuotaVersion] = []
        for index, quota_raw in enumerate(raw.get("quota_versions", [])):
            if not isinstance(quota_raw, dict):
                errors.append(
                    make_error(
                        "E_CONFIG_SCHEMA",
                        "quota_version",
                        str(index),
                        "quota_versions",
                        "版本必须为对象",
                    )
                )
                continue
            quota, quota_errors = QuotaVersion.from_raw(quota_raw, index)
            errors.extend(quota_errors)
            if quota is not None:
                quotas.append(quota)
        shards: list[ShardSpec] = []
        for index, shard_raw in enumerate(raw.get("shards", [])):
            if not isinstance(shard_raw, dict):
                errors.append(
                    make_error("E_CONFIG_SCHEMA", "shard", str(index), "shards", "分片必须为对象")
                )
                continue
            shard, shard_errors = ShardSpec.from_raw(shard_raw, index)
            errors.extend(shard_errors)
            if shard is not None:
                shards.append(shard)
        events: list[Event] = []
        for index, event_raw in enumerate(raw.get("events", [])):
            if not isinstance(event_raw, dict):
                errors.append(
                    make_error("E_CONFIG_SCHEMA", "event", str(index), "events", "事件必须为对象")
                )
                continue
            event, event_errors = Event.from_raw(event_raw, index)
            errors.extend(event_errors)
            if event is not None:
                events.append(event)
        injections = raw.get("injections", [])
        if not isinstance(injections, list):
            errors.append(
                make_error("E_CONFIG_SCHEMA", "config", "root", "injections", "注入配置必须为数组")
            )
            injections = []
        if errors:
            raise BizErrorException(errors[0])
        return cls(
            global_version=global_version,
            seed=seed,
            resource_budget=resource_budget,
            max_batch_points=max_batch_points,
            reference_width=reference_width,
            active_quota_version=active_version,
            nodes=nodes,
            quota_versions=quotas,
            shards=shards,
            events=events,
            injections=injections,
            lease_ttl_ms=lease_ttl_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_version": self.global_version,
            "seed": self.seed,
            "resource_budget": self.resource_budget,
            "max_batch_points": self.max_batch_points,
            "reference_width": self.reference_width,
            "active_quota_version": self.active_quota_version,
            "nodes": [node.to_dict() for node in self.nodes],
            "quota_versions": [quota.to_dict() for quota in self.quota_versions],
            "shards": [shard.to_dict() for shard in self.shards],
            "events": [event.to_dict() for event in self.events],
            "injections": list(self.injections),
            "lease_ttl_ms": self.lease_ttl_ms,
        }

    @property
    def config_hash(self) -> str:
        return fingerprint(self.to_dict())

    @property
    def base_config_hash(self) -> str:
        raw = self.to_dict()
        raw["events"] = []
        return fingerprint(raw)
