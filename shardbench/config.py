"""分片配置、路由单元与依赖图的加载和强校验。

输入由本地 JSON 组成：分片配置（逻辑表、分片键、算法、物理库/表数量、
预算参数）与路由单元（逻辑谓词样本、派生依赖）。

任何非法输入都以 ConfigError (E_CONFIG) 整体拒绝；依赖成环以
CycleError (E_CYCLE) 报告环上单元。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .algorithms import CONSISTENT_HASH, HashRing, RingNode, validate_algorithm
from .determinism import content_digest
from .errors import ConfigError, CycleError


@dataclass(frozen=True)
class TableConfig:
    name: str
    columns: tuple[str, ...]
    shard_key_columns: tuple[str, ...]
    algorithm: str
    params: dict[str, Any]
    db_count: int
    tables_per_db: int


@dataclass(frozen=True)
class RoutingUnit:
    unit_id: str
    table: str
    predicate_text: str | None
    depends_on: tuple[str, ...]


@dataclass(frozen=True)
class BudgetConfig:
    global_concurrency: int
    per_db_quota: int
    wait_timeout_ms: int


@dataclass
class ShardConfig:
    tables: dict[str, TableConfig]
    units: dict[str, RoutingUnit]
    budget: BudgetConfig
    rings: dict[str, HashRing]
    topo_order: list[str]
    config_digest: str
    ring_digest: str
    dependency_digest: str
    budget_digest: str
    raw: dict[str, Any] = field(default_factory=dict)

    def table_of(self, unit_id: str) -> TableConfig:
        return self.tables[self.units[unit_id].table]

    def downstream(self, unit_id: str) -> list[str]:
        """直接下游（依赖于该单元的单元），按标识稳定排序。"""
        return sorted(
            other.unit_id
            for other in self.units.values()
            if unit_id in other.depends_on
        )


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise ConfigError(f"输入文件不存在: {path}", paths=[path])
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"JSON 解析失败: {exc.msg} (行 {exc.lineno} 列 {exc.colno})",
            paths=[path],
        ) from exc


def _parse_tables(payload: Any) -> dict[str, TableConfig]:
    if not isinstance(payload, dict) or not isinstance(payload.get("tables"), list):
        raise ConfigError("配置根对象必须包含 tables 列表", paths=["tables"])
    tables: dict[str, TableConfig] = {}
    for index, item in enumerate(payload["tables"]):
        path = f"tables[{index}]"
        if not isinstance(item, dict):
            raise ConfigError("表配置必须是对象", paths=[path])
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError("表名必须为非空字符串", paths=[f"{path}.name"])
        if name in tables:
            raise ConfigError(f"表名重复: {name}", paths=[f"{path}.name"])
        columns = item.get("columns")
        if not isinstance(columns, list) or not columns or any(
            not isinstance(column, str) or not column for column in columns
        ):
            raise ConfigError(
                "columns 必须为非空字符串列表", paths=[f"{path}.columns"]
            )
        if len(set(columns)) != len(columns):
            raise ConfigError("表内列名重复", paths=[f"{path}.columns"])
        shard_key = item.get("shard_key_columns")
        if not isinstance(shard_key, list) or not shard_key:
            raise ConfigError(
                "shard_key_columns 必须为非空列表",
                paths=[f"{path}.shard_key_columns"],
            )
        if any(not isinstance(column, str) for column in shard_key):
            raise ConfigError(
                "分片键列必须为字符串", paths=[f"{path}.shard_key_columns"]
            )
        column_set = set(columns)
        for column in shard_key:
            if column not in column_set:
                raise ConfigError(
                    f"分片键列 {column} 不存在于表 {name} 的结构中",
                    paths=[f"{path}.shard_key_columns"],
                )
        if len(set(shard_key)) != len(shard_key):
            raise ConfigError(
                "分片键列重复", paths=[f"{path}.shard_key_columns"]
            )
        if not _is_positive_int(item.get("db_count")):
            raise ConfigError(
                "物理库数量 db_count 必须为正整数", paths=[f"{path}.db_count"]
            )
        if not _is_positive_int(item.get("tables_per_db")):
            raise ConfigError(
                "每库表数量 tables_per_db 必须为正整数",
                paths=[f"{path}.tables_per_db"],
            )
        algorithm = item.get("algorithm")
        params = item.get("params", {})
        if not isinstance(params, dict):
            raise ConfigError("算法参数 params 必须为对象", paths=[f"{path}.params"])
        validate_algorithm(
            algorithm,
            params,
            path=path,
            shard_key_columns=tuple(shard_key),
        )
        tables[name] = TableConfig(
            name=name,
            columns=tuple(columns),
            shard_key_columns=tuple(shard_key),
            algorithm=algorithm,
            params=dict(params),
            db_count=item["db_count"],
            tables_per_db=item["tables_per_db"],
        )
    return tables


def _parse_units(
    payload: Any, tables: dict[str, TableConfig]
) -> dict[str, RoutingUnit]:
    raw_units = payload.get("routing_units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ConfigError(
            "routing_units 必须为非空列表", paths=["routing_units"]
        )
    units: dict[str, RoutingUnit] = {}
    for index, item in enumerate(raw_units):
        path = f"routing_units[{index}]"
        if not isinstance(item, dict):
            raise ConfigError("路由单元必须是对象", paths=[path])
        unit_id = item.get("id")
        if not isinstance(unit_id, str) or not unit_id:
            raise ConfigError(
                "路由单元 id 必须为非空字符串", paths=[f"{path}.id"]
            )
        if unit_id in units:
            raise ConfigError(
                f"路由单元 id 重复: {unit_id}", paths=[f"{path}.id"]
            )
        table_name = item.get("table")
        if not isinstance(table_name, str) or table_name not in tables:
            raise ConfigError(
                f"路由单元 {unit_id} 引用了不存在的逻辑表 {table_name!r}",
                paths=[f"{path}.table"],
            )
        predicate_text = item.get("predicate")
        if predicate_text is not None and not isinstance(predicate_text, str):
            raise ConfigError(
                "predicate 必须为字符串或 null", paths=[f"{path}.predicate"]
            )
        depends_on = item.get("depends_on", [])
        if not isinstance(depends_on, list) or any(
            not isinstance(dep, str) for dep in depends_on
        ):
            raise ConfigError(
                "depends_on 必须为字符串列表", paths=[f"{path}.depends_on"]
            )
        if predicate_text is None and not depends_on:
            raise ConfigError(
                f"路由单元 {unit_id} 必须提供谓词或至少一个依赖",
                paths=[f"{path}.predicate"],
            )
        units[unit_id] = RoutingUnit(
            unit_id=unit_id,
            table=table_name,
            predicate_text=predicate_text,
            depends_on=tuple(depends_on),
        )
    for unit in units.values():
        for dep in unit.depends_on:
            if dep not in units:
                raise ConfigError(
                    f"路由单元 {unit.unit_id} 依赖了不存在的单元 {dep}",
                    paths=[f"routing_units.{unit.unit_id}.depends_on"],
                )
            if dep == unit.unit_id:
                raise CycleError([unit.unit_id])
    return units


def topological_order(units: dict[str, RoutingUnit]) -> list[str]:
    """Kahn 算法；就绪候选按标识排序，保证输出稳定。检测成环则 E_CYCLE。"""
    indegree = {unit_id: 0 for unit_id in units}
    for unit in units.values():
        for dep in unit.depends_on:
            indegree[unit.unit_id] += 1
    ready = sorted(unit_id for unit_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        unit_id = ready.pop(0)
        order.append(unit_id)
        for child in sorted(
            other.unit_id
            for other in units.values()
            if unit_id in other.depends_on
        ):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(order) != len(units):
        cycle = _find_cycle(units)
        raise CycleError(cycle)
    return order


def _find_cycle(units: dict[str, RoutingUnit]) -> list[str]:
    """DFS 返回首个环上的单元标识序列。"""
    state: dict[str, int] = {unit_id: 0 for unit_id in units}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for dep in sorted(units[node].depends_on):
            if state[dep] == 0:
                found = visit(dep)
                if found is not None:
                    return found
            elif state[dep] == 1:
                start = stack.index(dep)
                return stack[start:] + [dep]
        stack.pop()
        state[node] = 2
        return None

    for unit_id in sorted(units):
        if state[unit_id] == 0:
            found = visit(unit_id)
            if found is not None:
                return found
    return []


def _parse_budget(payload: Any) -> BudgetConfig:
    raw = payload.get("budget", {})
    if not isinstance(raw, dict):
        raise ConfigError("budget 必须为对象", paths=["budget"])
    global_concurrency = raw.get("global_concurrency", 2)
    per_db_quota = raw.get("per_db_quota", 1)
    wait_timeout_ms = raw.get("wait_timeout_ms", 5000)
    if not _is_positive_int(global_concurrency):
        raise ConfigError(
            "global_concurrency 必须为正整数",
            paths=["budget.global_concurrency"],
        )
    if not _is_positive_int(per_db_quota):
        raise ConfigError(
            "per_db_quota 必须为正整数", paths=["budget.per_db_quota"]
        )
    if not _is_positive_int(wait_timeout_ms):
        raise ConfigError(
            "wait_timeout_ms 必须为正整数", paths=["budget.wait_timeout_ms"]
        )
    return BudgetConfig(
        global_concurrency=global_concurrency,
        per_db_quota=per_db_quota,
        wait_timeout_ms=wait_timeout_ms,
    )


def _build_rings(tables: dict[str, TableConfig]) -> dict[str, HashRing]:
    rings: dict[str, HashRing] = {}
    for table_name in sorted(tables):
        table = tables[table_name]
        if table.algorithm != CONSISTENT_HASH:
            continue
        ring_version = "ring-" + content_digest(
            {"table": table.name, "params": table.params}
        )[:16]
        nodes = [
            RingNode(node_id=node["id"], weight=node["weight"])
            for node in table.params["nodes"]
        ]
        rings[table.name] = HashRing(nodes, ring_version)
    return rings


def load_config(config_path: str, units_path: str | None = None) -> ShardConfig:
    """加载并强校验分片配置与路由单元，任何非法输入整体拒绝。"""
    payload = _load_json(config_path)
    if not isinstance(payload, dict):
        raise ConfigError("配置根对象必须是 JSON 对象", paths=[config_path])
    tables = _parse_tables(payload)

    if units_path is not None:
        units_payload = _load_json(units_path)
        if not isinstance(units_payload, dict):
            raise ConfigError(
                "路由单元文件根对象必须是 JSON 对象", paths=[units_path]
            )
        merged = dict(payload)
        merged["routing_units"] = units_payload.get("routing_units", [])
    else:
        merged = payload
    units = _parse_units(merged, tables)
    order = topological_order(units)
    budget = _parse_budget(payload)
    rings = _build_rings(tables)

    config_digest = content_digest(
        [
            {
                "name": table.name,
                "columns": list(table.columns),
                "shard_key_columns": list(table.shard_key_columns),
                "algorithm": table.algorithm,
                "params": table.params,
                "db_count": table.db_count,
                "tables_per_db": table.tables_per_db,
            }
            for table_name, table in sorted(tables.items())
        ]
    )
    ring_digest = content_digest(
        {
            table_name: {
                "version": rings[table_name].ring_version,
                "points": rings[table_name].describe(),
            }
            for table_name in sorted(rings)
        }
    )
    dependency_digest = content_digest(
        sorted(
            (unit.unit_id, sorted(unit.depends_on)) for unit in units.values()
        )
    )
    budget_digest = content_digest(
        {
            "global_concurrency": budget.global_concurrency,
            "per_db_quota": budget.per_db_quota,
            "wait_timeout_ms": budget.wait_timeout_ms,
        }
    )
    return ShardConfig(
        tables=tables,
        units=units,
        budget=budget,
        rings=rings,
        topo_order=order,
        config_digest=config_digest,
        ring_digest=ring_digest,
        dependency_digest=dependency_digest,
        budget_digest=budget_digest,
        raw=payload,
    )


def load_config_from_dict(
    payload: dict[str, Any], units_payload: dict[str, Any] | None = None
) -> ShardConfig:
    """供测试与异常注入使用的内存入口，校验逻辑与文件加载完全一致。"""
    tables = _parse_tables(payload)
    merged = dict(payload)
    if units_payload is not None:
        merged["routing_units"] = units_payload.get("routing_units", [])
    units = _parse_units(merged, tables)
    order = topological_order(units)
    budget = _parse_budget(payload)
    rings = _build_rings(tables)
    config_digest = content_digest(
        [
            {
                "name": table.name,
                "columns": list(table.columns),
                "shard_key_columns": list(table.shard_key_columns),
                "algorithm": table.algorithm,
                "params": table.params,
                "db_count": table.db_count,
                "tables_per_db": table.tables_per_db,
            }
            for table_name, table in sorted(tables.items())
        ]
    )
    ring_digest = content_digest(
        {
            table_name: {
                "version": rings[table_name].ring_version,
                "points": rings[table_name].describe(),
            }
            for table_name in sorted(rings)
        }
    )
    dependency_digest = content_digest(
        sorted(
            (unit.unit_id, sorted(unit.depends_on)) for unit in units.values()
        )
    )
    budget_digest = content_digest(
        {
            "global_concurrency": budget.global_concurrency,
            "per_db_quota": budget.per_db_quota,
            "wait_timeout_ms": budget.wait_timeout_ms,
        }
    )
    return ShardConfig(
        tables=tables,
        units=units,
        budget=budget,
        rings=rings,
        topo_order=order,
        config_digest=config_digest,
        ring_digest=ring_digest,
        dependency_digest=dependency_digest,
        budget_digest=budget_digest,
        raw=payload,
    )
