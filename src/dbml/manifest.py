from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from .database import (
    NUMERIC_TYPES,
    SUPPORTED_TYPES,
    column_types,
    connect,
    normalize_sql_type,
    table_exists,
    table_info,
    validate_supported_type,
)
from .errors import MigrationError, fail
from .expression import ParsedExpression, parse_expression
from .fixedpoint import QuantizationSpec


SUPPORTED_OPERATIONS = (
    "create_column",
    "widen_column",
    "rename_column",
    "backfill",
    "add_constraint",
    "create_index",
)
ALLOWED_WIDENINGS = {
    ("INTEGER", "REAL"),
    ("INTEGER", "DECIMAL"),
    ("REAL", "DECIMAL"),
}
COMMON_STEP_FIELDS = {"step_id", "version", "depends_on", "operation", "table"}
OPERATION_FIELDS = {
    "create_column": {"column", "type", "nullable", "default", "quantization"},
    "widen_column": {"column", "from_type", "to_type", "quantization"},
    "rename_column": {"column", "new_name"},
    "backfill": {"column", "expression", "quantization"},
    "add_constraint": {"constraint"},
    "create_index": {"index"},
}


@dataclass(frozen=True)
class MigrationStep:
    index: int
    step_id: str
    version: int
    depends_on: tuple[str, ...]
    operation: str
    table: str
    column: str | None = None
    new_name: str | None = None
    create_type: str | None = None
    from_type: str | None = None
    to_type: str | None = None
    nullable: bool = True
    default_sql: str | None = None
    expression_text: str | None = None
    expression: ParsedExpression | None = None
    quantization: QuantizationSpec | None = None
    constraint_name: str | None = None
    constraint_kind: str | None = None
    constraint_expression: str | None = None
    index_name: str | None = None
    index_columns: tuple[str, ...] = ()
    index_unique: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        index: int,
        default_quantization: QuantizationSpec | None,
    ) -> "MigrationStep":
        path = f"steps[{index}]"
        step_id = payload.get("step_id")
        if not isinstance(step_id, str) or not step_id.strip():
            fail(
                "E_MIGRATE",
                "step_id must be a non-empty string",
                step_id=str(step_id) if step_id is not None else None,
                field_path=f"{path}.step_id",
            )
        version = payload.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            fail(
                "E_MIGRATE",
                "version must be an integer",
                step_id=step_id,
                field_path=f"{path}.version",
            )
        depends_on_raw = payload.get("depends_on", [])
        if not isinstance(depends_on_raw, list) or not all(
            isinstance(item, str) and item for item in depends_on_raw
        ):
            fail(
                "E_MIGRATE",
                "depends_on must be a list of step identifiers",
                step_id=step_id,
                field_path=f"{path}.depends_on",
            )
        if len(set(depends_on_raw)) != len(depends_on_raw):
            fail(
                "E_MIGRATE",
                "depends_on contains duplicate step identifiers",
                step_id=step_id,
                field_path=f"{path}.depends_on",
            )
        operation = payload.get("operation")
        if operation not in SUPPORTED_OPERATIONS:
            fail(
                "E_MIGRATE",
                f"unsupported migration operation {operation!r}",
                step_id=step_id,
                field_path=f"{path}.operation",
                supported=list(SUPPORTED_OPERATIONS),
            )
        table = payload.get("table")
        if not isinstance(table, str) or not table:
            fail(
                "E_MIGRATE",
                "table must be a non-empty string",
                step_id=step_id,
                field_path=f"{path}.table",
            )
        allowed = COMMON_STEP_FIELDS | OPERATION_FIELDS[operation]
        extra = sorted(set(payload) - allowed)
        if extra:
            fail(
                "E_MIGRATE",
                "unknown field in migration step",
                step_id=step_id,
                field_path=f"{path}.{extra[0]}",
            )
        required = {
            "create_column": {"column", "type"},
            "widen_column": {"column", "to_type"},
            "rename_column": {"column", "new_name"},
            "backfill": {"column", "expression"},
            "add_constraint": {"constraint"},
            "create_index": {"index"},
        }[operation]
        missing = sorted(required - set(payload))
        if missing:
            fail(
                "E_MIGRATE",
                "required migration field is missing",
                step_id=step_id,
                field_path=f"{path}.{missing[0]}",
            )
        column = payload.get("column")
        if "column" in required or "column" in payload:
            if not isinstance(column, str) or not column:
                fail(
                    "E_MIGRATE",
                    "column must be a non-empty string",
                    step_id=step_id,
                    field_path=f"{path}.column",
                )
        quantization = None
        if "quantization" in payload:
            quantization = QuantizationSpec.from_dict(
                payload.get("quantization"),
                step_id=step_id,
                field_path=f"{path}.quantization",
                default=default_quantization,
            )
        elif operation in {"create_column", "widen_column", "backfill"} and default_quantization is not None:
            quantization = default_quantization

        new_name = payload.get("new_name")
        if operation == "rename_column" and (not isinstance(new_name, str) or not new_name):
            fail(
                "E_MIGRATE",
                "new_name must be a non-empty string",
                step_id=step_id,
                field_path=f"{path}.new_name",
            )
        create_type = None
        if operation == "create_column":
            create_type = validate_supported_type(
                payload.get("type"), step_id=step_id, field_path=f"{path}.type"
            )
            if create_type in NUMERIC_TYPES and quantization is None:
                fail(
                    "E_MIGRATE",
                    "numeric create_column requires quantization",
                    step_id=step_id,
                    field_path=f"{path}.quantization",
                )
            if create_type not in NUMERIC_TYPES and quantization is not None:
                fail(
                    "E_MIGRATE",
                    "quantization is only valid for numeric columns",
                    step_id=step_id,
                    field_path=f"{path}.quantization",
                )
        from_type = None
        to_type = None
        if operation == "widen_column":
            to_type = validate_supported_type(
                payload.get("to_type"), step_id=step_id, field_path=f"{path}.to_type"
            )
            if "from_type" in payload:
                from_type = validate_supported_type(
                    payload.get("from_type"),
                    step_id=step_id,
                    field_path=f"{path}.from_type",
                )
            if to_type not in NUMERIC_TYPES or quantization is None:
                fail(
                    "E_MIGRATE",
                    "numeric widening requires a numeric target and quantization",
                    step_id=step_id,
                    field_path=f"{path}.quantization",
                )
        nullable = True
        if operation == "create_column":
            nullable = payload.get("nullable", True)
            if not isinstance(nullable, bool):
                fail(
                    "E_MIGRATE",
                    "nullable must be a boolean",
                    step_id=step_id,
                    field_path=f"{path}.nullable",
                )
        default_sql = payload.get("default")
        if default_sql is not None and not isinstance(default_sql, (str, int, float)):
            fail(
                "E_MIGRATE",
                "default must be a string or number",
                step_id=step_id,
                field_path=f"{path}.default",
            )
        expression = None
        expression_text = None
        if operation == "backfill":
            expression_text = payload.get("expression")
            expression = parse_expression(
                expression_text,
                step_id=step_id,
                field_path=f"{path}.expression",
            )
            if quantization is None:
                fail(
                    "E_MIGRATE",
                    "backfill requires a target quantization declaration",
                    step_id=step_id,
                    field_path=f"{path}.quantization",
                )
        constraint_name = None
        constraint_kind = None
        constraint_expression = None
        if operation == "add_constraint":
            constraint = payload.get("constraint")
            if not isinstance(constraint, dict):
                fail(
                    "E_MIGRATE",
                    "constraint must be an object",
                    step_id=step_id,
                    field_path=f"{path}.constraint",
                )
            constraint_name = constraint.get("name")
            constraint_kind = constraint.get("kind")
            if not isinstance(constraint_name, str) or not constraint_name:
                fail(
                    "E_MIGRATE",
                    "constraint.name must be a non-empty string",
                    step_id=step_id,
                    field_path=f"{path}.constraint.name",
                )
            if constraint_kind not in {"check", "unique"}:
                fail(
                    "E_MIGRATE",
                    "constraint.kind must be check or unique",
                    step_id=step_id,
                    field_path=f"{path}.constraint.kind",
                )
            if constraint_kind == "check":
                constraint_expression = constraint.get("expression")
                if not isinstance(constraint_expression, str) or not constraint_expression.strip():
                    fail(
                        "E_MIGRATE",
                        "check constraint requires an expression",
                        step_id=step_id,
                        field_path=f"{path}.constraint.expression",
                    )
            else:
                columns = constraint.get("columns")
                if not isinstance(columns, list) or not columns or not all(
                    isinstance(item, str) and item for item in columns
                ):
                    fail(
                        "E_MIGRATE",
                        "unique constraint requires a non-empty columns list",
                        step_id=step_id,
                        field_path=f"{path}.constraint.columns",
                    )
        index_name = None
        index_columns: tuple[str, ...] = ()
        index_unique = False
        if operation == "create_index":
            index_payload = payload.get("index")
            if not isinstance(index_payload, dict):
                fail(
                    "E_MIGRATE",
                    "index must be an object",
                    step_id=step_id,
                    field_path=f"{path}.index",
                )
            index_name = index_payload.get("name")
            index_columns_raw = index_payload.get("columns")
            index_unique = index_payload.get("unique", False)
            if not isinstance(index_name, str) or not index_name:
                fail(
                    "E_MIGRATE",
                    "index.name must be a non-empty string",
                    step_id=step_id,
                    field_path=f"{path}.index.name",
                )
            if not isinstance(index_columns_raw, list) or not index_columns_raw or not all(
                isinstance(item, str) and item for item in index_columns_raw
            ):
                fail(
                    "E_MIGRATE",
                    "index.columns must be a non-empty list of column names",
                    step_id=step_id,
                    field_path=f"{path}.index.columns",
                )
            if not isinstance(index_unique, bool):
                fail(
                    "E_MIGRATE",
                    "index.unique must be a boolean",
                    step_id=step_id,
                    field_path=f"{path}.index.unique",
                )
            index_columns = tuple(index_columns_raw)
        return cls(
            index=index,
            step_id=step_id,
            version=version,
            depends_on=tuple(depends_on_raw),
            operation=operation,
            table=table,
            column=column,
            new_name=new_name,
            create_type=create_type,
            from_type=from_type,
            to_type=to_type,
            nullable=nullable,
            default_sql=default_sql,
            expression_text=expression_text,
            expression=expression,
            quantization=quantization,
            constraint_name=constraint_name,
            constraint_kind=constraint_kind,
            constraint_expression=constraint_expression,
            index_name=index_name,
            index_columns=index_columns,
            index_unique=index_unique,
            raw=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "step_id": self.step_id,
            "version": self.version,
            "depends_on": list(self.depends_on),
            "operation": self.operation,
            "table": self.table,
            "column": self.column,
            "new_name": self.new_name,
            "create_type": self.create_type,
            "from_type": self.from_type,
            "to_type": self.to_type,
            "nullable": self.nullable,
            "default_sql": self.default_sql,
            "expression": self.expression_text,
            "quantization": self.quantization.to_dict() if self.quantization else None,
            "constraint_name": self.constraint_name,
            "constraint_kind": self.constraint_kind,
            "constraint_expression": self.constraint_expression,
            "index_name": self.index_name,
            "index_columns": list(self.index_columns),
            "index_unique": self.index_unique,
        }


@dataclass
class Manifest:
    manifest_id: str
    steps: tuple[MigrationStep, ...]
    execution_order: tuple[str, ...]
    initial_calibrations: dict[tuple[str, str], QuantizationSpec]
    table_row_keys: dict[str, tuple[str, ...]]
    default_quantization: QuantizationSpec | None
    raw: dict[str, Any]
    sha256: str
    derivation_edges: tuple[tuple[str, str], ...] = ()
    derivation_order: tuple[str, ...] = ()

    @property
    def step_map(self) -> dict[str, MigrationStep]:
        return {step.step_id: step for step in self.steps}

    def ordered_steps(self) -> list[MigrationStep]:
        mapping = self.step_map
        return [mapping[step_id] for step_id in self.execution_order]

    def step(self, step_id: str) -> MigrationStep:
        return self.step_map[step_id]

    def calibration_for(self, table: str, column: str) -> QuantizationSpec | None:
        return self.initial_calibrations.get((table, column))

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "sha256": self.sha256,
            "execution_order": list(self.execution_order),
            "derivation_edges": [list(edge) for edge in self.derivation_edges],
            "derivation_order": list(self.derivation_order),
            "table_row_keys": {
                table: list(columns)
                for table, columns in sorted(self.table_row_keys.items())
            },
            "initial_calibrations": {
                table: {
                    column: spec.to_dict()
                    for (cal_table, column), spec in sorted(self.initial_calibrations.items())
                    if cal_table == table
                }
                for table in sorted({table for table, _ in self.initial_calibrations})
            },
        }


def load_manifest(path: str | Path, db_path: str | Path | None = None) -> Manifest:
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(
            "E_MIGRATE",
            f"cannot load migration manifest: {exc}",
            field_path="$",
        ) from exc
    digest = _payload_digest(payload)
    if db_path is not None:
        connection = connect(db_path, readonly=True)
        try:
            has_run_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_mig_run'"
            ).fetchone()
            if has_run_table is not None:
                run = connection.execute(
                    "SELECT manifest_sha256, plan_json FROM _mig_run WHERE run_id = 1"
                ).fetchone()
                if run is not None and str(run["manifest_sha256"]) == digest:
                    return manifest_from_plan(
                        payload, json.loads(str(run["plan_json"])), digest
                    )
        except sqlite3.Error:
            pass
        finally:
            connection.close()
    return validate_manifest(payload, db_path=db_path)


def _payload_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_from_plan(
    raw: dict[str, Any], plan: dict[str, Any], digest: str | None = None
) -> Manifest:
    structural = validate_manifest(raw, db_path=None)
    initial: dict[tuple[str, str], QuantizationSpec] = {}
    for table, columns in plan.get("initial_calibrations", {}).items():
        for column, spec_payload in columns.items():
            initial[(table, column)] = QuantizationSpec.from_dict(
                spec_payload,
                field_path=f"plan.initial_calibrations.{table}.{column}",
            )
    row_keys = {
        table: tuple(columns)
        for table, columns in plan.get("table_row_keys", {}).items()
    }
    if not row_keys:
        row_keys = structural.table_row_keys
    return Manifest(
        manifest_id=structural.manifest_id,
        steps=structural.steps,
        execution_order=tuple(plan.get("execution_order", structural.execution_order)),
        initial_calibrations=initial or structural.initial_calibrations,
        table_row_keys=row_keys,
        default_quantization=structural.default_quantization,
        raw=raw,
        sha256=digest or _payload_digest(raw),
        derivation_edges=tuple(
            tuple(edge) for edge in plan.get("derivation_edges", [])
        ),
        derivation_order=tuple(plan.get("derivation_order", [])),
    )


def validate_manifest(
    payload: dict[str, Any], *, db_path: str | Path | None = None
) -> Manifest:
    if not isinstance(payload, dict):
        fail("E_MIGRATE", "manifest root must be a JSON object", field_path="$")
    allowed_top = {
        "manifest_version",
        "manifest_id",
        "description",
        "default_quantization",
        "initial_calibrations",
        "table_row_keys",
        "steps",
    }
    extra = sorted(set(payload) - allowed_top)
    if extra:
        fail(
            "E_MIGRATE",
            "unknown top-level manifest field",
            field_path=f"$.{extra[0]}",
        )
    manifest_version = payload.get("manifest_version", 1)
    if manifest_version != 1:
        fail(
            "E_MIGRATE",
            "unsupported manifest_version",
            field_path="$.manifest_version",
            supported=[1],
        )
    manifest_id = payload.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id:
        fail("E_MIGRATE", "manifest_id must be a non-empty string", field_path="$.manifest_id")
    default_quantization = None
    if "default_quantization" in payload:
        default_quantization = QuantizationSpec.from_dict(
            payload.get("default_quantization"),
            field_path="$.default_quantization",
        )
    steps_payload = payload.get("steps")
    if not isinstance(steps_payload, list) or not steps_payload:
        fail("E_MIGRATE", "steps must be a non-empty list", field_path="$.steps")
    steps: list[MigrationStep] = []
    seen_ids: set[str] = set()
    previous_version: int | None = None
    for index, step_payload in enumerate(steps_payload):
        if not isinstance(step_payload, dict):
            fail(
                "E_MIGRATE",
                "migration step must be a JSON object",
                field_path=f"steps[{index}]",
            )
        step = MigrationStep.from_payload(
            step_payload, index=index, default_quantization=default_quantization
        )
        if step.step_id in seen_ids:
            fail(
                "E_MIGRATE",
                "step_id must be unique",
                step_id=step.step_id,
                field_path=f"steps[{index}].step_id",
            )
        seen_ids.add(step.step_id)
        if previous_version is not None and step.version <= previous_version:
            fail(
                "E_MIGRATE",
                "step versions must be strictly increasing in manifest order",
                step_id=step.step_id,
                field_path=f"steps[{index}].version",
                previous_version=previous_version,
            )
        previous_version = step.version
        steps.append(step)
    step_map = {step.step_id: step for step in steps}
    for step in steps:
        for dependency in step.depends_on:
            if dependency == step.step_id:
                fail(
                    "E_MIGRATE",
                    "a step cannot depend on itself",
                    step_id=step.step_id,
                    field_path=f"steps[{step.index}].depends_on",
                )
            if dependency not in step_map:
                fail(
                    "E_MIGRATE",
                    "predecessor step does not exist",
                    step_id=step.step_id,
                    field_path=f"steps[{step.index}].depends_on",
                    missing=dependency,
                )
    execution_order = _stable_topological_order(steps, step_map)
    ancestors = _ancestor_map(steps, step_map)

    initial_calibrations: dict[tuple[str, str], QuantizationSpec] = {}
    table_row_keys: dict[str, tuple[str, ...]] = {}
    derivation_edges: set[tuple[str, str]] = set()
    derivation_order: tuple[str, ...] = ()
    if db_path is not None:
        connection = connect(db_path, readonly=True)
        try:
            initial_calibrations, table_row_keys, derivation_edges, derivation_order = (
                _validate_against_database(
                    connection,
                    payload,
                    steps,
                    execution_order,
                    ancestors,
                    default_quantization,
                )
            )
        finally:
            connection.close()
    else:
        initial_calibrations = _parse_initial_calibrations(payload)
        table_row_keys = _parse_row_keys(
            payload, None, {table for table, _ in initial_calibrations}
        )
        _validate_derivation_edges_graph(steps, derivation_edges)

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Manifest(
        manifest_id=manifest_id,
        steps=tuple(steps),
        execution_order=execution_order,
        initial_calibrations=initial_calibrations,
        table_row_keys=table_row_keys,
        default_quantization=default_quantization,
        raw=payload,
        sha256=digest,
        derivation_edges=tuple(sorted(derivation_edges)),
        derivation_order=derivation_order,
    )


def _stable_topological_order(
    steps: list[MigrationStep], step_map: dict[str, MigrationStep]
) -> tuple[str, ...]:
    indegree = {step.step_id: 0 for step in steps}
    outgoing: dict[str, list[str]] = {step.step_id: [] for step in steps}
    for step in steps:
        for dependency in step.depends_on:
            indegree[step.step_id] += 1
            outgoing[dependency].append(step.step_id)
    available = sorted(
        [step for step in steps if indegree[step.step_id] == 0],
        key=lambda step: (step.version, step.index, step.step_id),
    )
    result: list[str] = []
    while available:
        step = available.pop(0)
        result.append(step.step_id)
        for successor_id in sorted(outgoing[step.step_id]):
            indegree[successor_id] -= 1
            if indegree[successor_id] == 0:
                available.append(step_map[successor_id])
        available.sort(key=lambda item: (item.version, item.index, item.step_id))
    if len(result) != len(steps):
        blocked = sorted(step_id for step_id, degree in indegree.items() if degree > 0)
        first = (
            min((step_map[step_id] for step_id in blocked), key=lambda step: step.index)
            if blocked
            else steps[0]
        )
        fail(
            "E_MIGRATE",
            "migration dependency graph contains a cycle",
            step_id=first.step_id,
            field_path=f"steps[{first.index}].depends_on",
            cycle_steps=blocked,
        )
    return tuple(result)


def _ancestor_map(
    steps: list[MigrationStep], step_map: dict[str, MigrationStep]
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}

    def visit(step_id: str, stack: set[str]) -> set[str]:
        if step_id in result:
            return result[step_id]
        if step_id in stack:
            return set()
        found: set[str] = set()
        for dependency in step_map[step_id].depends_on:
            found.add(dependency)
            found.update(visit(dependency, stack | {step_id}))
        result[step_id] = found
        return found

    for step in steps:
        visit(step.step_id, set())
    return result


def _parse_initial_calibrations(
    payload: dict[str, Any]
) -> dict[tuple[str, str], QuantizationSpec]:
    result: dict[tuple[str, str], QuantizationSpec] = {}
    raw = payload.get("initial_calibrations", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        fail(
            "E_MIGRATE",
            "initial_calibrations must be an object",
            field_path="$.initial_calibrations",
        )
    for table in sorted(raw):
        table_payload = raw[table]
        if not isinstance(table_payload, dict):
            fail(
                "E_MIGRATE",
                "initial_calibrations table entry must be an object",
                field_path=f"$.initial_calibrations.{table}",
            )
        for column in sorted(table_payload):
            result[(table, column)] = QuantizationSpec.from_dict(
                table_payload[column],
                field_path=f"$.initial_calibrations.{table}.{column}",
            )
    return result


def _parse_row_keys(
    payload: dict[str, Any],
    connection: sqlite3.Connection | None,
    known_tables: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    raw = payload.get("table_row_keys", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        fail(
            "E_MIGRATE",
            "table_row_keys must be an object",
            field_path="$.table_row_keys",
        )
    result: dict[str, tuple[str, ...]] = {}
    for table in sorted(raw):
        columns = raw[table]
        if not isinstance(columns, list) or not columns or not all(
            isinstance(item, str) and item for item in columns
        ):
            fail(
                "E_MIGRATE",
                "table row key must be a non-empty list of column names",
                field_path=f"$.table_row_keys.{table}",
            )
        if connection is not None:
            if not table_exists(connection, table):
                fail(
                    "E_MIGRATE",
                    "row key references a table that does not exist",
                    field_path=f"$.table_row_keys.{table}",
                )
            types = column_types(connection, table)
            for column in columns:
                if column not in types:
                    fail(
                        "E_MIGRATE",
                        "row key references a column that does not exist",
                        field_path=f"$.table_row_keys.{table}.{column}",
                    )
        result[table] = tuple(columns)
    return result


def _validate_against_database(
    connection: sqlite3.Connection,
    payload: dict[str, Any],
    steps: list[MigrationStep],
    execution_order: tuple[str, ...],
    ancestors: dict[str, set[str]],
    default_quantization: QuantizationSpec | None,
) -> tuple[
    dict[tuple[str, str], QuantizationSpec],
    dict[str, tuple[str, ...]],
    set[tuple[str, str]],
    tuple[str, ...],
]:
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    for table in sorted(
        str(row[0])
        for row in connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_mig_%'
            ORDER BY name
            """
        ).fetchall()
    ):
        tables[table] = {}
        for row in table_info(connection, table):
            if row["hidden"]:
                continue
            tables[table][str(row["name"])] = {
                "type": normalize_sql_type(row["type"]),
                "producer": None,
                "derived_sources": (),
            }
    initial = _parse_initial_calibrations(payload)
    for (table, column), spec in sorted(initial.items()):
        if table not in tables or column not in tables[table]:
            fail(
                "E_MIGRATE",
                "initial calibration references an unknown column",
                field_path=f"$.initial_calibrations.{table}.{column}",
            )
        sql_type = tables[table][column]["type"]
        if sql_type not in NUMERIC_TYPES:
            fail(
                "E_MIGRATE",
                "initial calibration must target a numeric column",
                field_path=f"$.initial_calibrations.{table}.{column}",
            )
    for table in sorted(tables):
        for column in sorted(tables[table]):
            sql_type = tables[table][column]["type"]
            if sql_type not in NUMERIC_TYPES or (table, column) in initial:
                continue
            if default_quantization is not None:
                initial[(table, column)] = default_quantization
            else:
                fail(
                    "E_MIGRATE",
                    "every initial numeric column requires an explicit calibration",
                    field_path=f"$.initial_calibrations.{table}.{column}",
                )
    row_keys = _parse_row_keys(payload, connection, tables)
    step_map = {step.step_id: step for step in steps}
    index_names: set[str] = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    constraint_names: set[str] = set()
    derivation_edges: set[tuple[str, str]] = set()
    for step_id in execution_order:
        step = step_map[step_id]
        path = f"steps[{step.index}]"
        if step.table not in tables:
            fail(
                "E_MIGRATE",
                "step references a table that does not exist",
                step_id=step.step_id,
                field_path=f"{path}.table",
            )
        table_columns = tables[step.table]
        column = step.column
        if step.operation == "create_column":
            assert column is not None and step.create_type is not None
            if column in table_columns:
                fail(
                    "E_MIGRATE",
                    "create_column target already exists",
                    step_id=step.step_id,
                    field_path=f"{path}.column",
                )
            if step.create_type in NUMERIC_TYPES and step.quantization is None:
                fail(
                    "E_MIGRATE",
                    "numeric column creation requires quantization",
                    step_id=step.step_id,
                    field_path=f"{path}.quantization",
                )
            if step.create_type not in NUMERIC_TYPES and step.quantization is not None:
                fail(
                    "E_MIGRATE",
                    "quantization is only valid for numeric columns",
                    step_id=step.step_id,
                    field_path=f"{path}.quantization",
                )
            table_columns[column] = {
                "type": step.create_type,
                "producer": None,
                "derived_sources": (),
            }
            continue
        if step.operation == "widen_column":
            assert column is not None and step.to_type is not None
            if column not in table_columns:
                fail(
                    "E_MIGRATE",
                    "widen_column target does not exist",
                    step_id=step.step_id,
                    field_path=f"{path}.column",
                )
            current_type = table_columns[column]["type"]
            if current_type not in NUMERIC_TYPES:
                fail(
                    "E_MIGRATE",
                    "only numeric columns can be widened",
                    step_id=step.step_id,
                    field_path=f"{path}.column",
                )
            if step.from_type is not None and step.from_type != current_type:
                fail(
                    "E_MIGRATE",
                    "from_type does not match the column type at execution time",
                    step_id=step.step_id,
                    field_path=f"{path}.from_type",
                    actual=current_type,
                )
            if (current_type, step.to_type) not in ALLOWED_WIDENINGS:
                fail(
                    "E_MIGRATE",
                    "unsupported widening transition",
                    step_id=step.step_id,
                    field_path=f"{path}.to_type",
                    source=current_type,
                    target=step.to_type,
                    supported=[list(item) for item in sorted(ALLOWED_WIDENINGS)],
                )
            table_columns[column]["type"] = step.to_type
            continue
        if step.operation == "rename_column":
            assert column is not None and step.new_name is not None
            if column not in table_columns:
                fail(
                    "E_MIGRATE",
                    "rename_column source does not exist",
                    step_id=step.step_id,
                    field_path=f"{path}.column",
                )
            if step.new_name in table_columns:
                fail(
                    "E_MIGRATE",
                    "rename_column target already exists",
                    step_id=step.step_id,
                    field_path=f"{path}.new_name",
                )
            table_columns[step.new_name] = table_columns.pop(column)
            continue
        if step.operation == "backfill":
            assert column is not None and step.expression is not None
            if column not in table_columns:
                fail(
                    "E_MIGRATE",
                    "backfill target does not exist",
                    step_id=step.step_id,
                    field_path=f"{path}.column",
                )
            if table_columns[column]["type"] not in NUMERIC_TYPES:
                fail(
                    "E_MIGRATE",
                    "backfill target must be numeric",
                    step_id=step.step_id,
                    field_path=f"{path}.column",
                )
            if column in step.expression.references:
                fail(
                    "E_MIGRATE",
                    "backfill expression cannot reference its own target column",
                    step_id=step.step_id,
                    field_path=f"{path}.expression",
                )
            derived_sources: list[str] = []
            for reference in step.expression.references:
                if reference not in table_columns:
                    fail(
                        "E_MIGRATE",
                        "backfill expression references a column that does not exist at execution time",
                        step_id=step.step_id,
                        field_path=f"{path}.expression",
                        column=reference,
                    )
                if table_columns[reference]["type"] not in NUMERIC_TYPES:
                    fail(
                        "E_MIGRATE",
                        "backfill expression references a non-numeric column",
                        step_id=step.step_id,
                        field_path=f"{path}.expression",
                        column=reference,
                    )
                producer = table_columns[reference].get("producer")
                if producer is not None:
                    derived_sources.append(reference)
                    if producer not in ancestors[step.step_id]:
                        fail(
                            "E_MIGRATE",
                            "derived column reference is not ordered by a declared predecessor",
                            step_id=step.step_id,
                            field_path=f"{path}.expression",
                            column=reference,
                            producer_step=producer,
                        )
                    step_by_id = step_map[producer]
                    producer_column = _producer_target_after_renames(
                        step_by_id, step_map, execution_order, reference
                    )
                    derivation_edges.add(
                        (f"{step.table}.{reference}", f"{step.table}.{column}")
                    )
            table_columns[column]["producer"] = step.step_id
            table_columns[column]["derived_sources"] = tuple(sorted(derived_sources))
            continue
        if step.operation == "add_constraint":
            assert step.constraint_name is not None and step.constraint_kind is not None
            if step.constraint_name in constraint_names or step.constraint_name.startswith("sqlite_"):
                fail(
                    "E_MIGRATE",
                    "constraint name must be unique and must not use the sqlite_ prefix",
                    step_id=step.step_id,
                    field_path=f"{path}.constraint.name",
                )
            constraint_names.add(step.constraint_name)
            raw_constraint = step.raw["constraint"]
            if step.constraint_kind == "check":
                expression = step.constraint_expression or ""
                try:
                    _validate_check_against_simulated_columns(
                        step.table, table_columns, expression
                    )
                except sqlite3.Error as exc:
                    fail(
                        "E_MIGRATE",
                        f"invalid check constraint expression: {exc}",
                        step_id=step.step_id,
                        field_path=f"{path}.constraint.expression",
                    )
                for reference in _sql_identifier_candidates(expression):
                    if reference in table_columns and reference.upper() not in {
                        "AND",
                        "OR",
                        "NOT",
                        "NULL",
                        "IS",
                        "IN",
                        "LIKE",
                        "BETWEEN",
                    }:
                        continue
            else:
                columns = raw_constraint.get("columns")
                for constraint_column in columns:
                    if constraint_column not in table_columns:
                        fail(
                            "E_MIGRATE",
                            "unique constraint references a column that does not exist",
                            step_id=step.step_id,
                            field_path=f"{path}.constraint.columns",
                            column=constraint_column,
                        )
            continue
        if step.operation == "create_index":
            assert step.index_name is not None
            if step.index_name in index_names or step.index_name.startswith("sqlite_"):
                fail(
                    "E_MIGRATE",
                    "index name must be unique and must not use the sqlite_ prefix",
                    step_id=step.step_id,
                    field_path=f"{path}.index.name",
                )
            index_names.add(step.index_name)
            for index_column in step.index_columns:
                if index_column not in table_columns:
                    fail(
                        "E_MIGRATE",
                        "index references a column that does not exist",
                        step_id=step.step_id,
                        field_path=f"{path}.index.columns",
                        column=index_column,
                    )
            continue
        raise AssertionError(f"unhandled operation {step.operation}")
    _validate_derivation_edges_graph(steps, derivation_edges)
    derivation_order = _derivation_topological_order(derivation_edges)
    return initial, row_keys, derivation_edges, derivation_order


def _producer_target_after_renames(
    producer_step: MigrationStep,
    step_map: dict[str, MigrationStep],
    execution_order: tuple[str, ...],
    reference: str,
) -> str:
    return reference


def _validate_derivation_edges_graph(
    steps: list[MigrationStep], edges: set[tuple[str, str]]
) -> None:
    nodes: set[str] = set()
    outgoing: dict[str, list[str]] = {}
    indegree: dict[str, int] = {}
    for source, target in edges:
        nodes.add(source)
        nodes.add(target)
        outgoing.setdefault(source, []).append(target)
        indegree[target] = indegree.get(target, 0) + 1
        indegree.setdefault(source, indegree.get(source, 0))
    available = sorted(node for node in nodes if indegree.get(node, 0) == 0)
    visited = 0
    while available:
        node = available.pop(0)
        visited += 1
        for successor in sorted(outgoing.get(node, [])):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                available.append(successor)
                available.sort()
    if visited != len(nodes):
        blocked = sorted(node for node, degree in indegree.items() if degree > 0)
        fail(
            "E_MIGRATE",
            "derived column dependency graph contains a cycle",
            step_id=None,
            field_path="steps.expression",
            cycle_columns=blocked,
        )


def _derivation_topological_order(edges: set[tuple[str, str]]) -> tuple[str, ...]:
    nodes: set[str] = set()
    outgoing: dict[str, set[str]] = {}
    indegree: dict[str, int] = {}
    for source, target in edges:
        nodes.update({source, target})
        outgoing.setdefault(source, set()).add(target)
        indegree[target] = indegree.get(target, 0) + 1
        indegree.setdefault(source, indegree.get(source, 0))
    available = sorted(node for node in nodes if indegree[node] == 0)
    result: list[str] = []
    while available:
        node = available.pop(0)
        result.append(node)
        for successor in sorted(outgoing.get(node, set())):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                available.append(successor)
        available.sort()
    return tuple(result)


def _sql_identifier_candidates(expression: str) -> set[str]:
    import re

    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _validate_check_against_simulated_columns(
    table: str,
    columns: dict[str, dict[str, Any]],
    expression: str,
) -> None:
    memory = sqlite3.connect(":memory:")
    try:
        definitions = ", ".join(
            f"{_quote(column)} {details['type'] or 'NUMERIC'}"
            for column, details in sorted(columns.items())
        )
        memory.execute(f"CREATE TABLE {_quote(table)} ({definitions})")
        memory.execute(f"SELECT ({expression}) FROM {_quote(table)} WHERE 0").fetchall()
    finally:
        memory.close()
