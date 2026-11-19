from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import GSBError, spec_error
from .util import sha256_json


TYPE_ALIASES = {
    "int": "int",
    "integer": "int",
    "int64": "int",
    "long": "int",
    "float": "float",
    "double": "float",
    "float64": "float",
    "real": "float",
}
BOUNDARY_RULES = {
    "left_closed_right_open": "left_closed_right_open",
    "left_open_right_closed": "left_open_right_closed",
    "lcr_o": "left_closed_right_open",
    "lcr_o_": "left_closed_right_open",
    "lor_c": "left_open_right_closed",
}
PREDICATE_OPS = {"eq", "ne", "lt", "le", "gt", "ge", "between", "in", "not_in", "is_null", "not_null"}
VALUE_OPS = {"eq", "ne", "lt", "le", "gt", "ge"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _normalize_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return TYPE_ALIASES.get(value.strip().lower())


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class Predicate:
    column: str
    op: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"column": self.column, "op": self.op}
        if self.op not in {"is_null", "not_null"}:
            result["value"] = _json_value(self.value)
        return result

    def sort_key(self) -> tuple[str, str, str]:
        return (self.column, self.op, json.dumps(_json_value(self.value), sort_keys=True, ensure_ascii=False))


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str
    null_fraction: float = 0.0
    unique: bool = False
    domain: int | float | None = None
    step: float | None = None

    @property
    def dtype(self) -> np.dtype:
        return np.dtype("<i8") if self.kind == "int" else np.dtype("<f8")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "type": self.kind,
            "null_fraction": self.null_fraction,
            "unique": self.unique,
        }
        if self.domain is not None:
            payload["domain"] = self.domain
        if self.step is not None:
            payload["step"] = self.step
        return payload


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: tuple[ColumnSpec, ...]
    declared_rows: int
    seed: int
    skewed: bool = False
    skew_hot_fraction: float = 0.82
    skew_hot_domain_fraction: float = 0.03

    def column(self, name: str) -> ColumnSpec | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "declared_rows": self.declared_rows,
            "seed": self.seed,
            "skewed": self.skewed,
            "skew": {
                "hot_fraction": self.skew_hot_fraction,
                "hot_domain_fraction": self.skew_hot_domain_fraction,
            },
            "columns": [column.to_dict() for column in self.columns],
        }


@dataclass(frozen=True)
class JoinSpec:
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    left_predicates: tuple[Predicate, ...]
    right_predicates: tuple[Predicate, ...]
    buckets: int
    sample_size: int
    sample_seed: int
    parameter_version: int
    boundary_rules: Mapping[str, str]
    top_k: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_table": self.left_table,
            "right_table": self.right_table,
            "left_key": self.left_key,
            "right_key": self.right_key,
            "left_predicates": [predicate.to_dict() for predicate in self.left_predicates],
            "right_predicates": [predicate.to_dict() for predicate in self.right_predicates],
            "buckets": self.buckets,
            "sample_size": self.sample_size,
            "sample_seed": self.sample_seed,
            "parameter_version": self.parameter_version,
            "boundary_rules": dict(self.boundary_rules),
            "top_k": self.top_k,
        }

    @property
    def version_parameters(self) -> dict[str, Any]:
        return {
            "buckets": self.buckets,
            "sample_size": self.sample_size,
            "sample_seed": self.sample_seed,
            "boundary_rules": dict(self.boundary_rules),
        }


@dataclass(frozen=True)
class ScenarioSpec:
    tables: Mapping[str, TableSpec]
    join: JoinSpec
    schema_version: int = 1
    name: str = "local-join-lab"

    def table(self, name: str) -> TableSpec:
        return self.tables[name]

    @property
    def left(self) -> TableSpec:
        return self.tables[self.join.left_table]

    @property
    def right(self) -> TableSpec:
        return self.tables[self.join.right_table]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "tables": {name: self.tables[name].to_dict() for name in sorted(self.tables)},
            "join": self.join.to_dict(),
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


def _parse_predicate(raw: Any, table: TableSpec, side: str, errors: list[dict[str, Any]]) -> Predicate | None:
    if not isinstance(raw, Mapping):
        errors.append({"field": f"join.{side}_predicates", "error": "谓词必须为 JSON 对象"})
        return None
    column = raw.get("column")
    op = raw.get("op")
    if not isinstance(column, str) or not column:
        errors.append({"field": f"join.{side}_predicates.column", "error": "谓词列名缺失"})
        return None
    if table.column(column) is None:
        errors.append(
            {
                "field": f"join.{side}_predicates.column",
                "error": "谓词引用不存在的列",
                "table": table.name,
                "column": column,
            }
        )
        return None
    if not isinstance(op, str) or op not in PREDICATE_OPS:
        errors.append(
            {
                "field": f"join.{side}_predicates.op",
                "error": "不支持的谓词运算符",
                "table": table.name,
                "column": column,
                "op": op,
            }
        )
        return None
    if op in {"is_null", "not_null"}:
        return Predicate(column=column, op=op)
    if op in VALUE_OPS:
        value = raw.get("value")
        if not _is_number(value):
            errors.append(
                {
                    "field": f"join.{side}_predicates.value",
                    "error": "比较谓词的值必须为有限数值",
                    "column": column,
                }
            )
            return None
        return Predicate(column=column, op=op, value=value)
    if op in {"in", "not_in"}:
        value = raw.get("value")
        if not isinstance(value, list) or not value or not all(_is_number(item) for item in value):
            errors.append(
                {
                    "field": f"join.{side}_predicates.value",
                    "error": "IN 谓词值必须为非空数值数组",
                    "column": column,
                }
            )
            return None
        unique_values = sorted({float(item) if isinstance(item, float) else int(item) for item in value})
        return Predicate(column=column, op=op, value=tuple(unique_values))
    value = raw.get("value")
    if not isinstance(value, list) or len(value) != 2 or not all(_is_number(item) for item in value):
        errors.append(
            {
                "field": f"join.{side}_predicates.value",
                "error": "BETWEEN 谓词值必须为两个有限数值",
                "column": column,
            }
        )
        return None
    low, high = value
    if float(low) > float(high):
        errors.append(
            {
                "field": f"join.{side}_predicates.value",
                "error": "BETWEEN 下界不能大于上界",
                "column": column,
            }
        )
        return None
    return Predicate(column=column, op=op, value=(low, high))


def _parse_table(raw: Any, fallback_name: str | None, errors: list[dict[str, Any]]) -> TableSpec | None:
    if not isinstance(raw, Mapping):
        errors.append({"field": "tables", "error": "表声明必须为 JSON 对象"})
        return None
    name = raw.get("name", fallback_name)
    if not isinstance(name, str) or not name.strip():
        errors.append({"field": "tables.name", "error": "表名必须为非空字符串"})
        return None
    name = name.strip()
    declared_rows = raw.get("declared_rows")
    if not _is_int(declared_rows) or declared_rows <= 0:
        errors.append({"field": f"tables.{name}.declared_rows", "error": "声明行数必须为正整数"})
    seed = raw.get("seed")
    if not _is_int(seed):
        errors.append({"field": f"tables.{name}.seed", "error": "随机种子必须为整数"})
    skewed = raw.get("skewed", False)
    if not isinstance(skewed, bool):
        errors.append({"field": f"tables.{name}.skewed", "error": "倾斜标记必须为布尔值"})
        skewed = False
    skew = raw.get("skew", {})
    if skew is None:
        skew = {}
    if not isinstance(skew, Mapping):
        errors.append({"field": f"tables.{name}.skew", "error": "倾斜配置必须为对象"})
        skew = {}
    hot_fraction = skew.get("hot_fraction", 0.82)
    hot_domain_fraction = skew.get("hot_domain_fraction", 0.03)
    if not _is_number(hot_fraction) or not (0.0 <= float(hot_fraction) <= 1.0):
        errors.append({"field": f"tables.{name}.skew.hot_fraction", "error": "hot_fraction 必须在 [0,1]"})
        hot_fraction = 0.82
    if not _is_number(hot_domain_fraction) or not (0.0 < float(hot_domain_fraction) <= 1.0):
        errors.append(
            {"field": f"tables.{name}.skew.hot_domain_fraction", "error": "hot_domain_fraction 必须在 (0,1]"}
        )
        hot_domain_fraction = 0.03
    raw_columns = raw.get("columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        errors.append({"field": f"tables.{name}.columns", "error": "列定义必须为非空数组"})
        return None if not _is_int(declared_rows) or not _is_int(seed) else None
    columns: list[ColumnSpec] = []
    seen_columns: set[str] = set()
    for index, raw_column in enumerate(raw_columns):
        if not isinstance(raw_column, Mapping):
            errors.append({"field": f"tables.{name}.columns[{index}]", "error": "列声明必须为对象"})
            continue
        column_name = raw_column.get("name")
        if not isinstance(column_name, str) or not column_name.strip():
            errors.append({"field": f"tables.{name}.columns[{index}].name", "error": "列名必须为非空字符串"})
            continue
        column_name = column_name.strip()
        if column_name in seen_columns:
            errors.append(
                {"field": f"tables.{name}.columns", "error": "同一表内列名必须唯一", "column": column_name}
            )
            continue
        seen_columns.add(column_name)
        kind = _normalize_type(raw_column.get("type"))
        if kind is None:
            errors.append(
                {
                    "field": f"tables.{name}.columns.{column_name}.type",
                    "error": "列类型必须为 int 或 float",
                    "type": raw_column.get("type"),
                }
            )
            continue
        null_fraction = raw_column.get("null_fraction", 0.0)
        if not _is_number(null_fraction) or not (0.0 <= float(null_fraction) < 1.0):
            errors.append(
                {
                    "field": f"tables.{name}.columns.{column_name}.null_fraction",
                    "error": "null_fraction 必须在 [0,1)",
                }
            )
            null_fraction = 0.0
        unique = raw_column.get("unique", False)
        if not isinstance(unique, bool):
            errors.append(
                {"field": f"tables.{name}.columns.{column_name}.unique", "error": "unique 必须为布尔值"}
            )
            unique = False
        domain = raw_column.get("domain")
        if domain is not None:
            if kind == "int" and (not _is_int(domain) or domain <= 0):
                errors.append(
                    {"field": f"tables.{name}.columns.{column_name}.domain", "error": "整数列 domain 必须为正整数"}
                )
                domain = None
            elif kind == "float" and (not _is_number(domain) or float(domain) <= 0):
                errors.append(
                    {"field": f"tables.{name}.columns.{column_name}.domain", "error": "浮点列 domain 必须为正数"}
                )
                domain = None
        step = raw_column.get("step")
        if step is not None:
            if kind != "float" or not _is_number(step) or float(step) <= 0:
                errors.append(
                    {
                        "field": f"tables.{name}.columns.{column_name}.step",
                        "error": "step 仅适用于浮点列且必须为正数",
                    }
                )
                step = None
            else:
                step = float(step)
        if unique and domain is not None and _is_int(declared_rows) and int(domain) < declared_rows:
            errors.append(
                {
                    "field": f"tables.{name}.columns.{column_name}.domain",
                    "error": "unique 列的 domain 不能小于声明行数",
                }
            )
        columns.append(
            ColumnSpec(
                name=column_name,
                kind=kind,
                null_fraction=float(null_fraction),
                unique=unique,
                domain=domain,
                step=step,
            )
        )
    if not _is_int(declared_rows) or declared_rows <= 0 or not _is_int(seed):
        return None
    return TableSpec(
        name=name,
        columns=tuple(columns),
        declared_rows=int(declared_rows),
        seed=int(seed),
        skewed=skewed,
        skew_hot_fraction=float(hot_fraction),
        skew_hot_domain_fraction=float(hot_domain_fraction),
    )


def parse_scenario(payload: Any, source: str | None = None) -> ScenarioSpec:
    source_context = {"source": source} if source else {}
    if not isinstance(payload, Mapping):
        raise spec_error("场景声明根节点必须为 JSON 对象", [{"field": "root", **source_context}])
    schema_version = payload.get("schema_version", 1)
    if not _is_int(schema_version) or schema_version != 1:
        raise spec_error(
            "不支持的 schema_version",
            [{"field": "schema_version", "error": "当前仅支持 1", **source_context}],
        )
    name = payload.get("name", "local-join-lab")
    if not isinstance(name, str) or not name.strip():
        raise spec_error("场景名必须为非空字符串", [{"field": "name", "error": "必须为非空字符串", **source_context}])
    raw_join = payload.get("join")
    if not isinstance(raw_join, Mapping):
        raise spec_error("缺少 join 声明", [{"field": "join", "error": "必须为 JSON 对象", **source_context}])
    if "parameter_version" not in raw_join:
        raise GSBError(
            "E_VERSION",
            "参数版本缺失",
            {"field": "join.parameter_version", "hint": "显式提供正整数 parameter_version", **source_context},
        )
    raw_buckets = raw_join.get("buckets")
    if _is_int(raw_buckets) and raw_buckets <= 0:
        raise GSBError(
            "E_BUCKET",
            "分桶数必须为正整数",
            {"field": "join.buckets", "buckets": raw_buckets, **source_context},
        )
    if any(key in raw_join for key in ("left", "right")):
        left_raw = raw_join.get("left", {})
        right_raw = raw_join.get("right", {})
        if isinstance(left_raw, Mapping):
            raw_join = dict(raw_join)
            raw_join.setdefault("left_table", left_raw.get("table"))
            raw_join.setdefault("left_key", left_raw.get("key"))
            raw_join.setdefault("left_predicates", left_raw.get("predicates", []))
        if isinstance(right_raw, Mapping):
            raw_join = dict(raw_join)
            raw_join.setdefault("right_table", right_raw.get("table"))
            raw_join.setdefault("right_key", right_raw.get("key"))
            raw_join.setdefault("right_predicates", right_raw.get("predicates", []))
    errors: list[dict[str, Any]] = []
    raw_tables = payload.get("tables")
    table_pairs: list[tuple[str | None, Any]] = []
    if isinstance(raw_tables, list):
        table_pairs = [(None, item) for item in raw_tables]
    elif isinstance(raw_tables, Mapping):
        table_pairs = [(str(key), value) for key, value in raw_tables.items()]
    else:
        raise spec_error("tables 必须为两张表的对象或数组", [{"field": "tables", **source_context}])
    if len(table_pairs) != 2:
        errors.append({"field": "tables", "error": "必须且只能声明两张基表", "count": len(table_pairs)})
    tables: dict[str, TableSpec] = {}
    for fallback_name, raw_table in table_pairs:
        table = _parse_table(raw_table, fallback_name, errors)
        if table is None:
            continue
        if table.name in tables:
            errors.append({"field": "tables.name", "error": "表名必须唯一", "table": table.name})
            continue
        tables[table.name] = table
    if len(tables) != 2 and not errors:
        errors.append({"field": "tables", "error": "必须得到两张有效表", "count": len(tables)})
    left_table = raw_join.get("left_table")
    right_table = raw_join.get("right_table")
    if left_table not in tables:
        errors.append(
            {"field": "join.left_table", "error": "左表不存在", "value": left_table, "tables": sorted(tables)}
        )
        left_table = None
    if right_table not in tables:
        errors.append(
            {"field": "join.right_table", "error": "右表不存在", "value": right_table, "tables": sorted(tables)}
        )
        right_table = None
    if left_table is not None and right_table is not None and left_table == right_table:
        errors.append({"field": "join", "error": "联接两侧必须引用不同基表"})
    left_key = raw_join.get("left_key")
    right_key = raw_join.get("right_key")
    if left_table in tables:
        if not isinstance(left_key, str) or tables[left_table].column(left_key) is None:
            errors.append(
                {
                    "field": "join.left_key",
                    "error": "左联接键列不存在",
                    "table": left_table,
                    "key": left_key,
                }
            )
    if right_table in tables:
        if not isinstance(right_key, str) or tables[right_table].column(right_key) is None:
            errors.append(
                {
                    "field": "join.right_key",
                    "error": "右联接键列不存在",
                    "table": right_table,
                    "key": right_key,
                }
            )
    if (
        left_table in tables
        and right_table in tables
        and isinstance(left_key, str)
        and isinstance(right_key, str)
        and tables[left_table].column(left_key) is not None
        and tables[right_table].column(right_key) is not None
    ):
        left_column = tables[left_table].column(left_key)
        right_column = tables[right_table].column(right_key)
        if left_column is not None and right_column is not None:
            if left_column.kind not in {"int", "float"} or right_column.kind not in {"int", "float"}:
                errors.append({"field": "join", "error": "两侧联接键类型无法比较"})
    left_predicates = []
    right_predicates = []
    for raw_predicate in raw_join.get("left_predicates", raw_join.get("left_filters", [])) or []:
        if left_table in tables:
            parsed = _parse_predicate(raw_predicate, tables[left_table], "left", errors)
            if parsed:
                left_predicates.append(parsed)
    for raw_predicate in raw_join.get("right_predicates", raw_join.get("right_filters", [])) or []:
        if right_table in tables:
            parsed = _parse_predicate(raw_predicate, tables[right_table], "right", errors)
            if parsed:
                right_predicates.append(parsed)
    buckets = raw_join.get("buckets")
    if not _is_int(buckets) or buckets <= 0:
        errors.append({"field": "join.buckets", "error": "分桶数必须为正整数", "buckets": buckets})
    sample_size = raw_join.get("sample_size")
    if not _is_int(sample_size) or sample_size <= 0:
        errors.append({"field": "join.sample_size", "error": "抽样规模必须为正整数"})
    sample_seed = raw_join.get("sample_seed")
    if not _is_int(sample_seed):
        errors.append({"field": "join.sample_seed", "error": "抽样种子必须为整数"})
    parameter_version = raw_join.get("parameter_version")
    if not _is_int(parameter_version) or parameter_version <= 0:
        errors.append({"field": "join.parameter_version", "error": "参数版本必须为正整数"})
    top_k = raw_join.get("top_k", 5)
    if not _is_int(top_k) or top_k <= 0:
        errors.append({"field": "join.top_k", "error": "top_k 必须为正整数"})
        top_k = 5
    boundary_payload = raw_join.get("boundary_rules", {})
    if boundary_payload is None:
        boundary_payload = {}
    if not isinstance(boundary_payload, Mapping):
        errors.append({"field": "join.boundary_rules", "error": "边界规则必须为对象"})
        boundary_payload = {}
    boundary_rules = {
        "int": "left_closed_right_open",
        "float": "left_open_right_closed",
    }
    for kind in ("int", "float"):
        if kind in boundary_payload:
            alias = boundary_payload[kind]
            normalized = BOUNDARY_RULES.get(alias) if isinstance(alias, str) else None
            if normalized is None:
                errors.append(
                    {
                        "field": f"join.boundary_rules.{kind}",
                        "error": "未知边界归属规则",
                        "value": alias,
                        "allowed": sorted(set(BOUNDARY_RULES.values())),
                    }
                )
            else:
                boundary_rules[kind] = normalized
    if errors:
        raise spec_error("场景声明校验失败", [{**item, **source_context} for item in errors])
    assert left_table is not None and right_table is not None
    assert left_key is not None and right_key is not None
    join = JoinSpec(
        left_table=left_table,
        right_table=right_table,
        left_key=left_key,
        right_key=right_key,
        left_predicates=tuple(sorted(left_predicates, key=lambda item: item.sort_key())),
        right_predicates=tuple(sorted(right_predicates, key=lambda item: item.sort_key())),
        buckets=int(buckets),
        sample_size=int(sample_size),
        sample_seed=int(sample_seed),
        parameter_version=int(parameter_version),
        boundary_rules=boundary_rules,
        top_k=int(top_k),
    )
    return ScenarioSpec(
        tables={name: tables[name] for name in sorted(tables)},
        join=join,
        schema_version=1,
        name=name.strip(),
    )


def load_scenario(path: str | Path) -> ScenarioSpec:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise GSBError("E_SPEC", "无法读取场景声明", {"path": str(source), "reason": str(exc)}) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise spec_error(
            "场景声明不是合法 JSON",
            [{"field": "root", "error": "JSON 解析失败", "path": str(source), "line": exc.lineno, "column": exc.colno}],
        ) from exc
    return parse_scenario(payload, source=str(source))
