"""三种分片算法的确定性路由实现。

物理映射统一规则（``P = db_count * tables_per_db`` 个槽位）::

    db_index    = slot % db_count
    table_index = slot // db_count

取模：``slot = stable_hash(key_text) % base``，要求 ``base > 0``，再经上式映射；
范围：边界把整数域切为单调区间，显式声明边界值归属（lower 闭/upper 开）；
一致性哈希：虚拟节点按 FNV 排序成环，顺时针首个节点并正确处理环回绕。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .determinism import (
    all_physical_targets,
    fnv1a_64,
    physical_target,
    stable_hash,
)
from .errors import ConfigError, RingError

MODULO = "modulo"
RANGE = "range"
CONSISTENT_HASH = "consistent_hash"
ALGORITHMS = frozenset({MODULO, RANGE, CONSISTENT_HASH})


@dataclass(frozen=True)
class RouteDecision:
    targets: tuple[str, ...]
    algorithm: str
    params: dict[str, Any]
    ring_version: str | None
    slots: tuple[int, ...]
    boundary_ownership: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": list(self.targets),
            "algorithm": self.algorithm,
            "params": self.params,
            "ring_version": self.ring_version,
            "slots": list(self.slots),
            "boundary_ownership": self.boundary_ownership,
            "reason": self.reason,
        }


def slot_to_target(slot: int, db_count: int, tables_per_db: int) -> str:
    total = db_count * tables_per_db
    normalized = slot % total
    db_index = normalized % db_count
    table_index = normalized // db_count
    return physical_target(db_index, table_index)


def integerize(value: Any, column: str) -> int:
    """把分片键值整数化；布尔与无法整数化的值给出明确错误。"""
    if isinstance(value, bool):
        raise ConfigError(f"分片键列 {column} 的布尔值不能用于整数路由", paths=[column])
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return stable_hash(value)
    raise ConfigError(f"分片键列 {column} 的值 {value!r} 无法整数化", paths=[column])


def composite_key_text(column_values: Sequence[tuple[str, Any]]) -> str:
    """组合分片键按声明列顺序拼接为稳定文本。"""
    return "|".join(f"{column}={_stable_literal(value)}" for column, value in column_values)


def _stable_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if value is None:
        return "NULL"
    return str(value)


def validate_algorithm(
    algorithm: str,
    params: dict[str, Any],
    *,
    path: str,
    shard_key_columns: tuple[str, ...],
) -> None:
    if algorithm not in ALGORITHMS:
        raise ConfigError(
            f"未知分片算法 {algorithm!r}",
            paths=[f"{path}.algorithm"],
        )
    if algorithm == MODULO:
        base = params.get("base")
        if not isinstance(base, int) or isinstance(base, bool) or base <= 0:
            raise ConfigError(
                "取模基数 base 必须为大于零的整数",
                paths=[f"{path}.params.base"],
            )
    elif algorithm == RANGE:
        if len(shard_key_columns) != 1:
            raise ConfigError(
                "范围分片仅支持单列分片键",
                paths=[f"{path}.shard_key_columns"],
            )
        boundaries = params.get("boundaries")
        if not isinstance(boundaries, list) or not boundaries:
            raise ConfigError(
                "范围边界 boundaries 必须为非空列表",
                paths=[f"{path}.params.boundaries"],
            )
        if any(not isinstance(item, int) or isinstance(item, bool) for item in boundaries):
            raise ConfigError(
                "范围边界必须全部为整数",
                paths=[f"{path}.params.boundaries"],
            )
        if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
            raise ConfigError(
                "范围边界必须严格递增",
                paths=[f"{path}.params.boundaries"],
            )
        ownership = params.get("boundary_ownership", "lower")
        if ownership not in ("lower", "upper"):
            raise ConfigError(
                "boundary_ownership 仅支持 lower 或 upper",
                paths=[f"{path}.params.boundary_ownership"],
            )
    elif algorithm == CONSISTENT_HASH:
        nodes = params.get("nodes")
        _validate_ring_nodes(nodes, path)


def _validate_ring_nodes(nodes: Any, path: str) -> None:
    if not isinstance(nodes, list) or not nodes:
        raise ConfigError(
            "哈希环节点 nodes 必须为非空列表",
            paths=[f"{path}.params.nodes"],
        )
    seen: set[str] = set()
    for index, node in enumerate(nodes):
        node_path = f"{path}.params.nodes[{index}]"
        if not isinstance(node, dict):
            raise ConfigError("哈希环节点必须是对象", paths=[node_path])
        node_id = node.get("id")
        weight = node.get("weight")
        if not isinstance(node_id, str) or not node_id:
            raise ConfigError("节点 id 必须为非空字符串", paths=[f"{node_path}.id"])
        if not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0:
            raise ConfigError(
                "节点权重 weight 必须为正整数",
                paths=[f"{node_path}.weight"],
            )
        identity = f"{node_id}@{weight}"
        if identity in seen:
            raise RingError(
                f"哈希环同权重节点冲突: 节点 {node_id} 权重 {weight} 重复定义",
                paths=[f"{node_path}"],
                details={"node_id": node_id, "weight": weight},
            )
        seen.add(identity)


@dataclass(frozen=True)
class RingNode:
    node_id: str
    weight: int


@dataclass(frozen=True)
class VirtualPoint:
    position: int
    node_id: str
    vnode_index: int


class HashRing:
    """确定性一致性哈希环。

    每个物理节点按权重生成等量虚拟节点，虚拟节点位置为
    ``FNV1a(node_id + "#" + vnode_index)``，按位置升序排列；
    查询时定位顺时针首个虚拟节点，越过末位则回绕到首位。
    """

    def __init__(self, nodes: Sequence[RingNode], ring_version: str) -> None:
        self.ring_version = ring_version
        self.nodes = tuple(sorted(nodes, key=lambda node: (node.node_id, node.weight)))
        points: list[VirtualPoint] = []
        for node in self.nodes:
            for vnode_index in range(node.weight):
                token = f"{node.node_id}#{vnode_index}"
                points.append(
                    VirtualPoint(
                        position=fnv1a_64(token.encode("utf-8")),
                        node_id=node.node_id,
                        vnode_index=vnode_index,
                    )
                )
        points.sort(key=lambda point: (point.position, point.node_id, point.vnode_index))
        self.points = tuple(points)
        self.node_ids = tuple(node.node_id for node in self.nodes)

    @classmethod
    def from_params(cls, params: dict[str, Any], ring_version: str) -> "HashRing":
        nodes = [
            RingNode(node_id=node["id"], weight=node["weight"])
            for node in params["nodes"]
        ]
        return cls(nodes, ring_version)

    def locate(self, key_text: str) -> tuple[str, int]:
        key_position = fnv1a_64(key_text.encode("utf-8"))
        return self.locate_position(key_position)

    def locate_position(self, key_position: int) -> tuple[str, int]:
        """按键的环上位置定位顺时针首个虚拟节点，越过末位则回绕到首位。"""
        for point in self.points:
            if point.position >= key_position:
                return point.node_id, point.position
        first = self.points[0]
        return first.node_id, first.position

    def node_slot(self, node_id: str) -> int:
        return self.node_ids.index(node_id)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "position": point.position,
                "node_id": point.node_id,
                "vnode_index": point.vnode_index,
            }
            for point in self.points
        ]


def route_modulo(
    key_text: str,
    params: dict[str, Any],
    db_count: int,
    tables_per_db: int,
) -> RouteDecision:
    base = params["base"]
    slot = stable_hash(key_text) % base
    target = slot_to_target(slot, db_count, tables_per_db)
    return RouteDecision(
        targets=(target,),
        algorithm=MODULO,
        params={"base": base},
        ring_version=None,
        slots=(slot,),
        boundary_ownership=None,
        reason=f"stable_hash({key_text!r}) % {base} = {slot} -> {target}",
    )


def route_range(
    integer_value: int,
    params: dict[str, Any],
    db_count: int,
    tables_per_db: int,
) -> RouteDecision:
    boundaries = list(params["boundaries"])
    ownership = params.get("boundary_ownership", "lower")
    slot = _range_slot(integer_value, boundaries, ownership)
    target = slot_to_target(slot, db_count, tables_per_db)
    ownership_text = "边界值归入下界区间 [boundary, next)" if ownership == "lower" \
        else "边界值归入上界区间 (prev, boundary]"
    return RouteDecision(
        targets=(target,),
        algorithm=RANGE,
        params={"boundaries": boundaries, "boundary_ownership": ownership},
        ring_version=None,
        slots=(slot,),
        boundary_ownership=ownership,
        reason=(
            f"值 {integer_value} 落入槽位 {slot}（边界 {boundaries}，{ownership_text}）-> {target}"
        ),
    )


def _range_slot(value: int, boundaries: list[int], ownership: str) -> int:
    if ownership == "lower":
        for index, boundary in enumerate(boundaries):
            if value < boundary:
                return index
        return len(boundaries)
    for index, boundary in enumerate(boundaries):
        if value <= boundary:
            return index
    return len(boundaries)


def route_consistent_hash(
    key_text: str,
    ring: HashRing,
    db_count: int,
    tables_per_db: int,
    params: dict[str, Any],
) -> RouteDecision:
    node_id, point_position = ring.locate(key_text)
    slot = ring.node_slot(node_id)
    target = slot_to_target(slot, db_count, tables_per_db)
    return RouteDecision(
        targets=(target,),
        algorithm=CONSISTENT_HASH,
        params={"nodes": params["nodes"]},
        ring_version=ring.ring_version,
        slots=(slot,),
        boundary_ownership=None,
        reason=(
            f"环 {ring.ring_version} 顺时针首个节点 {node_id}"
            f"（虚拟点 {point_position}，槽位 {slot}）-> {target}"
        ),
    )


def broadcast_decision(
    reason: str,
    algorithm: str,
    params: dict[str, Any],
    db_count: int,
    tables_per_db: int,
    ring_version: str | None,
) -> RouteDecision:
    targets = tuple(all_physical_targets(db_count, tables_per_db))
    return RouteDecision(
        targets=targets,
        algorithm=algorithm,
        params=dict(params),
        ring_version=ring_version,
        slots=tuple(range(db_count * tables_per_db)),
        boundary_ownership=None,
        reason=f"广播路由：{reason}",
    )
