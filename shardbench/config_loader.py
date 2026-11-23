"""Load and strictly validate sharding config + routing-unit dependency graph.

Any violation rejects the whole document with E_CONFIG naming the first
offending field (document order); no routing probe may run afterwards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .canon import digest
from .errors import BizError, E_CONFIG

ALGO_MOD = "mod"
ALGO_RANGE = "mod_range"
ALGO_RING = "consistent_hash"
ALGO_TYPES = (ALGO_MOD, ALGO_RANGE, ALGO_RING)

BOUNDARY_POLICIES = ("left_closed", "right_closed")


def _fail(message, field_path):
    raise BizError(E_CONFIG, message, field=field_path)


def _is_pos_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


@dataclass(frozen=True)
class LogicalTable:
    name: str
    columns: tuple


@dataclass(frozen=True)
class RoutingUnit:
    unit_id: str
    table: str
    shard_keys: tuple
    algo_type: str
    algo_params: tuple  # tuple of (key, value) sorted by key, JSON-safe values
    physical_dbs: int
    tables_per_db: int
    depends_on: tuple

    @property
    def total_slots(self) -> int:
        return self.physical_dbs * self.tables_per_db

    def param(self, key, default=None):
        return dict(self.algo_params).get(key, default)

    def rule_view(self):
        """Canonical view of everything that defines this unit's routing rule."""
        return {
            "unit_id": self.unit_id,
            "table": self.table,
            "shard_keys": list(self.shard_keys),
            "algorithm": {"type": self.algo_type, "params": dict(self.algo_params)},
            "physical_dbs": self.physical_dbs,
            "tables_per_db": self.tables_per_db,
            "depends_on": sorted(self.depends_on),
        }


@dataclass
class Config:
    config_version: int
    ring_version: int
    dep_graph_version: int
    tables: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)

    def dep_graph_digest(self) -> str:
        edges = sorted(
            [u.unit_id, dep] for u in self.units.values() for dep in u.depends_on
        )
        return digest({"edges": edges, "version": self.dep_graph_version})


def _validate_tables(raw_tables):
    if not isinstance(raw_tables, list) or not raw_tables:
        _fail("logical_tables must be a non-empty array", "logical_tables")
    tables = {}
    for i, raw in enumerate(raw_tables):
        base = f"logical_tables[{i}]"
        if not isinstance(raw, dict):
            _fail("table entry must be an object", base)
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            _fail("table name must be a non-empty string", f"{base}.name")
        name = name.strip().lower()
        if name in tables:
            _fail(f"duplicate table name '{name}'", f"{base}.name")
        columns = raw.get("columns")
        if not isinstance(columns, list) or not columns:
            _fail("columns must be a non-empty array", f"{base}.columns")
        norm_cols = []
        for j, col in enumerate(columns):
            if not isinstance(col, str) or not col.strip():
                _fail("column name must be a non-empty string", f"{base}.columns[{j}]")
            norm_cols.append(col.strip().lower())
        if len(set(norm_cols)) != len(norm_cols):
            _fail("duplicate column names", f"{base}.columns")
        tables[name] = LogicalTable(name=name, columns=tuple(norm_cols))
    return tables


def _validate_algorithm(algo, total_slots, base):
    if not isinstance(algo, dict):
        _fail("algorithm must be an object", f"{base}.algorithm")
    algo_type = algo.get("type")
    if algo_type not in ALGO_TYPES:
        _fail(
            f"algorithm.type must be one of {list(ALGO_TYPES)}",
            f"{base}.algorithm.type",
        )
    params = {}
    if algo_type == ALGO_MOD:
        modulus = algo.get("modulus", total_slots)
        if not _is_pos_int(modulus):
            _fail("modulus must be a positive integer", f"{base}.algorithm.modulus")
        if modulus != total_slots:
            _fail(
                f"modulus {modulus} inconsistent with physical slots {total_slots}",
                f"{base}.algorithm.modulus",
            )
        salt = algo.get("salt", "")
        if not isinstance(salt, str):
            _fail("salt must be a string", f"{base}.algorithm.salt")
        params = {"modulus": modulus, "salt": salt}
    elif algo_type == ALGO_RANGE:
        boundaries = algo.get("boundaries")
        if not isinstance(boundaries, list) or not boundaries:
            _fail("boundaries must be a non-empty array", f"{base}.algorithm.boundaries")
        for j, b in enumerate(boundaries):
            if not _is_number(b):
                _fail("boundary must be a number", f"{base}.algorithm.boundaries[{j}]")
            if j > 0 and not boundaries[j - 1] < b:
                _fail(
                    "boundaries must be strictly increasing",
                    f"{base}.algorithm.boundaries[{j}]",
                )
        if len(boundaries) != total_slots - 1:
            _fail(
                f"need {total_slots - 1} boundaries for {total_slots} slots, "
                f"got {len(boundaries)}",
                f"{base}.algorithm.boundaries",
            )
        policy = algo.get("boundary_policy", "left_closed")
        if policy not in BOUNDARY_POLICIES:
            _fail(
                f"boundary_policy must be one of {list(BOUNDARY_POLICIES)}",
                f"{base}.algorithm.boundary_policy",
            )
        params = {"boundaries": list(boundaries), "boundary_policy": policy}
    else:  # consistent_hash
        vnodes = algo.get("virtual_nodes")
        if not _is_pos_int(vnodes):
            _fail(
                "virtual_nodes must be a positive integer",
                f"{base}.algorithm.virtual_nodes",
            )
        params = {"virtual_nodes": vnodes}
    return algo_type, tuple(sorted(params.items()))


def _validate_units(raw_units, tables):
    if not isinstance(raw_units, list) or not raw_units:
        _fail("routing_units must be a non-empty array", "routing_units")
    units = {}
    for i, raw in enumerate(raw_units):
        base = f"routing_units[{i}]"
        if not isinstance(raw, dict):
            _fail("routing unit must be an object", base)
        unit_id = raw.get("id")
        if not isinstance(unit_id, str) or not unit_id.strip():
            _fail("unit id must be a non-empty string", f"{base}.id")
        unit_id = unit_id.strip()
        if unit_id in units:
            _fail(f"duplicate routing unit id '{unit_id}'", f"{base}.id")
        table = raw.get("table")
        if not isinstance(table, str) or table.strip().lower() not in tables:
            _fail(f"unknown logical table '{table}'", f"{base}.table")
        table = table.strip().lower()
        shard_keys = raw.get("shard_keys")
        if not isinstance(shard_keys, list) or not shard_keys:
            _fail("shard_keys must be a non-empty array", f"{base}.shard_keys")
        norm_keys = []
        for j, key in enumerate(shard_keys):
            if not isinstance(key, str) or not key.strip():
                _fail("shard key must be a non-empty string", f"{base}.shard_keys[{j}]")
            key = key.strip().lower()
            if key not in tables[table].columns:
                _fail(
                    f"shard key '{key}' not in columns of table '{table}'",
                    f"{base}.shard_keys[{j}]",
                )
            norm_keys.append(key)
        if len(set(norm_keys)) != len(norm_keys):
            _fail("duplicate shard key columns", f"{base}.shard_keys")
        physical_dbs = raw.get("physical_dbs")
        if not _is_pos_int(physical_dbs):
            _fail("physical_dbs must be a positive integer", f"{base}.physical_dbs")
        tables_per_db = raw.get("tables_per_db")
        if not _is_pos_int(tables_per_db):
            _fail("tables_per_db must be a positive integer", f"{base}.tables_per_db")
        total_slots = physical_dbs * tables_per_db
        algo_type, algo_params = _validate_algorithm(
            raw.get("algorithm"), total_slots, base
        )
        depends_on = raw.get("depends_on", [])
        if not isinstance(depends_on, list):
            _fail("depends_on must be an array", f"{base}.depends_on")
        norm_deps = []
        for j, dep in enumerate(depends_on):
            if not isinstance(dep, str) or not dep.strip():
                _fail(
                    "dependency reference must be a non-empty string",
                    f"{base}.depends_on[{j}]",
                )
            dep = dep.strip()
            if dep == unit_id:
                _fail("unit cannot depend on itself", f"{base}.depends_on[{j}]")
            norm_deps.append(dep)
        if len(set(norm_deps)) != len(norm_deps):
            _fail("duplicate dependency reference", f"{base}.depends_on")
        units[unit_id] = RoutingUnit(
            unit_id=unit_id,
            table=table,
            shard_keys=tuple(norm_keys),
            algo_type=algo_type,
            algo_params=algo_params,
            physical_dbs=physical_dbs,
            tables_per_db=tables_per_db,
            depends_on=tuple(sorted(norm_deps)),
        )
    # dependency references must exist
    for unit in units.values():
        for dep in unit.depends_on:
            if dep not in units:
                _fail(
                    f"unknown dependency '{dep}'",
                    f"routing_units.{unit.unit_id}.depends_on",
                )
    _check_acyclic(units)
    return units


def _check_acyclic(units):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {u: WHITE for u in units}

    def visit(node, stack):
        color[node] = GRAY
        for dep in units[node].depends_on:
            if color[dep] == GRAY:
                cycle = " -> ".join(stack + [node, dep])
                _fail(f"dependency cycle detected: {cycle}", f"routing_units.{node}.depends_on")
            if color[dep] == WHITE:
                visit(dep, stack + [node])
        color[node] = BLACK

    for unit_id in sorted(units):
        if color[unit_id] == WHITE:
            visit(unit_id, [])


def load_config(path: str) -> Config:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw_text = fh.read()
    except OSError as exc:
        raise BizError(E_CONFIG, f"cannot read config file: {exc}", field="$file")
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise BizError(
            E_CONFIG, f"invalid JSON: {exc.msg} at line {exc.lineno}", field="$json"
        )
    if not isinstance(raw, dict):
        _fail("config root must be an object", "$")
    config_version = raw.get("config_version")
    if not _is_pos_int(config_version):
        _fail("config_version must be a positive integer", "config_version")
    ring_version = raw.get("ring_version", 1)
    if not _is_pos_int(ring_version):
        _fail("ring_version must be a positive integer", "ring_version")
    dep_graph_version = raw.get("dep_graph_version", 1)
    if not _is_pos_int(dep_graph_version):
        _fail("dep_graph_version must be a positive integer", "dep_graph_version")
    tables = _validate_tables(raw.get("logical_tables"))
    units = _validate_units(raw.get("routing_units"), tables)
    return Config(
        config_version=config_version,
        ring_version=ring_version,
        dep_graph_version=dep_graph_version,
        tables=tables,
        units=units,
    )
