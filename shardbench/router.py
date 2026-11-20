"""把逻辑谓词编译为确定性路由计划并定位物理库表。

关键约束：
* 谓词未覆盖完整分片键（单列或组合列任一缺失）→ 显式广播，禁止静默降级；
* 取模/一致性哈希遇到非等值（范围）条件无法定点 → 显式广播并说明原因；
* 范围算法支持等值/IN 与可枚举区间，边界归属来自配置；
* 派生单元的目标集合 = 自身谓词目标 ∪ 全部上游目标，路由理由记录派生依据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .algorithms import (
    MODULO,
    RANGE,
    RouteDecision,
    broadcast_decision,
    composite_key_text,
    integerize,
    route_consistent_hash,
    route_modulo,
    route_range,
    slot_to_target,
)
from .config import RoutingUnit, ShardConfig, TableConfig
from .determinism import all_physical_targets, stable_targets
from .errors import ConfigError
from .predicate import Condition, Predicate, parse_predicate


@dataclass
class RoutePlan:
    unit_id: str
    table: str
    normalized_predicate: str
    route_key: str
    is_broadcast: bool
    broadcast_reason: str | None
    targets: list[str]
    decisions: list[dict[str, Any]]
    derivation: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """稳定的单字符串结果摘要，用于逐字节比对。"""
        return f"{self.route_key}|{int(self.is_broadcast)}|{','.join(self.targets)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "table": self.table,
            "normalized_predicate": self.normalized_predicate,
            "route_key": self.route_key,
            "is_broadcast": self.is_broadcast,
            "broadcast_reason": self.broadcast_reason,
            "targets": list(self.targets),
            "decisions": list(self.decisions),
            "derivation": list(self.derivation),
            "summary": self.summary(),
        }


class Router:
    def __init__(self, config: ShardConfig) -> None:
        self.config = config

    def compile(self, unit_id: str) -> RoutePlan:
        unit = self.config.units[unit_id]
        table = self.config.table_of(unit_id)
        if unit.predicate_text is None:
            return self._derived_only_plan(unit, table)
        predicate = parse_predicate(unit.predicate_text)
        normalized = predicate.normalized_text()
        route_key = predicate.route_key()
        if not predicate.covers_key(table.shard_key_columns):
            reason = predicate.broadcast_reason(table.shard_key_columns)
            decision = broadcast_decision(
                reason,
                table.algorithm,
                table.params,
                table.db_count,
                table.tables_per_db,
                self._ring_version(table),
            )
            plan = RoutePlan(
                unit_id=unit_id,
                table=table.name,
                normalized_predicate=normalized,
                route_key=route_key,
                is_broadcast=True,
                broadcast_reason=reason,
                targets=list(decision.targets),
                decisions=[decision.to_dict()],
            )
        else:
            decisions = self._route_covered(predicate, table)
            is_broadcast = any(
                set(decision.targets)
                == set(all_physical_targets(table.db_count, table.tables_per_db))
                for decision in decisions
            )
            targets: set[str] = set()
            reasons: list[str] = []
            for decision in decisions:
                targets.update(decision.targets)
                reasons.append(decision.reason)
            broadcast_reason = None
            if is_broadcast:
                broadcast_reason = "；".join(
                    decision.reason.replace("广播路由：", "")
                    for decision in decisions
                    if set(decision.targets)
                    == set(
                        all_physical_targets(table.db_count, table.tables_per_db)
                    )
                )
            plan = RoutePlan(
                unit_id=unit_id,
                table=table.name,
                normalized_predicate=normalized,
                route_key=route_key,
                is_broadcast=is_broadcast,
                broadcast_reason=broadcast_reason,
                targets=stable_targets(targets),
                decisions=[decision.to_dict() for decision in decisions],
            )
        return plan

    def _ring_version(self, table: TableConfig) -> str | None:
        ring = self.config.rings.get(table.name)
        return ring.ring_version if ring else None

    def _derived_only_plan(self, unit: RoutingUnit, table: TableConfig) -> RoutePlan:
        normalized = "∅(derived-only)"
        route_key = "rk_derived_" + unit.unit_id
        return RoutePlan(
            unit_id=unit.unit_id,
            table=table.name,
            normalized_predicate=normalized,
            route_key=route_key,
            is_broadcast=False,
            broadcast_reason=None,
            targets=[],
            decisions=[],
            derivation=[],
        )

    def _route_covered(
        self, predicate: Predicate, table: TableConfig
    ) -> list[RouteDecision]:
        equality = predicate.equality_map()
        range_conditions = predicate.range_conditions()
        range_by_column: dict[str, list[Condition]] = {}
        for condition in range_conditions:
            range_by_column.setdefault(condition.column, []).append(condition)

        if table.algorithm == RANGE:
            return self._route_range_algorithm(
                table, equality, range_by_column
            )

        if any(
            column in range_by_column for column in table.shard_key_columns
        ):
            reason = (
                f"{table.algorithm} 分片对分片键上的范围条件无法定点，"
                "需扫描全部物理库表"
            )
            return [
                broadcast_decision(
                    reason,
                    table.algorithm,
                    table.params,
                    table.db_count,
                    table.tables_per_db,
                    self._ring_version(table),
                )
            ]

        value_sets: list[list[Any]] = []
        for column in table.shard_key_columns:
            if column not in equality:
                reason = f"分片键列 {column} 缺少等值/IN 条件"
                return [
                    broadcast_decision(
                        reason,
                        table.algorithm,
                        table.params,
                        table.db_count,
                        table.tables_per_db,
                        self._ring_version(table),
                    )
                ]
            value = equality[column]
            value_sets.append(value if isinstance(value, list) else [value])

        combinations = _cartesian_product(table.shard_key_columns, value_sets)
        decisions: list[RouteDecision] = []
        for combination in combinations:
            if len(table.shard_key_columns) == 1:
                column = table.shard_key_columns[0]
                raw_value = combination[0]
                if table.algorithm == MODULO:
                    integer_value = integerize(raw_value, column)
                    key_text = f"{column}={integer_value}"
                    decisions.append(
                        route_modulo(
                            key_text,
                            table.params,
                            table.db_count,
                            table.tables_per_db,
                        )
                    )
                else:
                    key_text = f"{column}={_stable_literal(raw_value)}"
                    decisions.append(
                        route_consistent_hash(
                            key_text,
                            self.config.rings[table.name],
                            table.db_count,
                            table.tables_per_db,
                            table.params,
                        )
                    )
            else:
                pairs = list(zip(table.shard_key_columns, combination))
                key_text = composite_key_text(
                    [
                        (column, _stable_literal(value))
                        for column, value in pairs
                    ]
                )
                if table.algorithm == MODULO:
                    decisions.append(
                        route_modulo(
                            key_text,
                            table.params,
                            table.db_count,
                            table.tables_per_db,
                        )
                    )
                else:
                    decisions.append(
                        route_consistent_hash(
                            key_text,
                            self.config.rings[table.name],
                            table.db_count,
                            table.tables_per_db,
                            table.params,
                        )
                    )
        return decisions

    def _route_range_algorithm(
        self,
        table: TableConfig,
        equality: dict[str, Any],
        range_by_column: dict[str, list[Condition]],
    ) -> list[RouteDecision]:
        column = table.shard_key_columns[0]
        boundaries = list(table.params["boundaries"])
        ownership = table.params.get("boundary_ownership", "lower")
        slot_count = len(boundaries) + 1

        if column in equality:
            raw = equality[column]
            values = raw if isinstance(raw, list) else [raw]
            decisions = []
            for value in sorted(values, key=lambda item: (type(item).__name__, str(item))):
                integer_value = integerize(value, column)
                decisions.append(
                    route_range(
                        integer_value,
                        table.params,
                        table.db_count,
                        table.tables_per_db,
                    )
                )
            return decisions

        conditions = range_by_column.get(column)
        if not conditions:
            reason = f"范围分片键列 {column} 既无等值也无范围条件"
            return [
                broadcast_decision(
                    reason,
                    table.algorithm,
                    table.params,
                    table.db_count,
                    table.tables_per_db,
                    self._ring_version(table),
                )
            ]
        lower_slot, upper_slot = _interval_slots(
            conditions, boundaries, ownership, slot_count
        )
        decisions = []
        for slot in range(lower_slot, upper_slot + 1):
            target = slot_to_target(slot, table.db_count, table.tables_per_db)
            decisions.append(
                RouteDecision(
                    targets=(target,),
                    algorithm=RANGE,
                    params={
                        "boundaries": boundaries,
                        "boundary_ownership": ownership,
                    },
                    ring_version=None,
                    slots=(slot,),
                    boundary_ownership=ownership,
                    reason=(
                        f"范围条件覆盖槽位 {slot}（归属 {ownership}）-> {target}"
                    ),
                )
            )
        return decisions


def _stable_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if value is None:
        return "NULL"
    return str(value)


def _cartesian_product(
    columns: tuple[str, ...], value_sets: list[list[Any]]
) -> list[tuple[Any, ...]]:
    ordered_sets = [
        sorted(values, key=lambda item: (type(item).__name__, str(item)))
        for values in value_sets
    ]
    result: list[tuple[Any, ...]] = [()]
    for values in ordered_sets:
        result = [base + (value,) for base in result for value in values]
    return result


def _interval_slots(
    conditions: list[Condition],
    boundaries: list[int],
    ownership: str,
    slot_count: int,
) -> tuple[int, int]:
    """把一组范围约束折算为连续槽位区间 [lower_slot, upper_slot]。

    lower 归属（边界 b_i 属于槽 i，槽 i = [b_i, b_{i+1})）：
        >= v 下界槽 = bisect_right(boundaries, v)（边界 b_i==v 时 v 在槽 i）
        >  v 下界槽 = bisect_right(boundaries, v)
        <= v 上界槽 = bisect_right(boundaries, v) - 1
        <  v 上界槽 = bisect_left(boundaries, v)
    upper 归属（边界 b_i 属于槽 i-1，槽 i = (b_{i-1}, b_i]）：
        >= v 下界槽 = bisect_left(boundaries, v)
        >  v 下界槽 = bisect_right(boundaries, v)
        <= v 上界槽 = bisect_right(boundaries, v) - 1
        <  v 上界槽 = bisect_left(boundaries, v) - 1
    """
    import bisect

    for condition in conditions:
        value = condition.values[0]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(
                f"范围条件值 {value!r} 必须为整数",
                paths=["predicate"],
            )
    left = bisect.bisect_left
    right = bisect.bisect_right
    lower_slot = 0
    upper_slot = slot_count - 1
    for condition in conditions:
        value = condition.values[0]
        if ownership == "lower":
            if condition.operator in (">=", ">"):
                candidate = right(boundaries, value)
            elif condition.operator == "<=":
                candidate = right(boundaries, value) - 1
            else:  # "<"
                candidate = left(boundaries, value)
        else:
            if condition.operator == ">=":
                candidate = left(boundaries, value)
            elif condition.operator == ">":
                candidate = right(boundaries, value)
            elif condition.operator == "<=":
                candidate = right(boundaries, value) - 1
            else:  # "<"
                candidate = left(boundaries, value) - 1
        if condition.operator in (">=", ">"):
            lower_slot = max(lower_slot, candidate)
        else:
            upper_slot = min(upper_slot, candidate)
    lower_slot = max(0, min(lower_slot, slot_count - 1))
    upper_slot = max(0, min(upper_slot, slot_count - 1))
    if lower_slot > upper_slot:
        return lower_slot, upper_slot
    return lower_slot, upper_slot


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
