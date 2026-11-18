from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable

from .analysis import (
    build_step_result,
    derivation_graph_payload,
    derivation_graph_version,
    execute_backfill,
)
from .database import (
    NUMERIC_TYPES,
    canonical_json,
    check_constraint_violations,
    check_expression_valid,
    column_types,
    connect,
    database_fingerprint,
    ensure_internal_tables,
    normalize_sql_type,
    ordered_rows,
    quote_identifier,
    read_column_meta,
    rebuild_table,
    row_key_columns,
    row_key_for_row,
    schema_digest,
    table_content_digest,
    table_exists,
    table_row_count,
    units_to_db_value,
    upsert_column_meta,
    value_to_units,
)
from .errors import MigrationError, fail
from .fixedpoint import QuantizationSpec, parse_decimal, pow10, quantize_decimal, rescale_units
from .ledger import (
    calibration_snapshot,
    completed_step_ids,
    current_revision,
    export_ledger,
    ledger_view,
    record_recalibration,
    record_step,
    seed_initial_calibrations,
    verify_ledger_chain,
)
from .manifest import Manifest, MigrationStep, manifest_from_plan, validate_manifest


@dataclass
class MigrationRunResult:
    manifest: Manifest
    complete: bool
    status: str
    committed_steps: list[str]
    unknown_steps: list[str]
    ledger: dict[str, Any]
    interrupted_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest.manifest_id,
            "manifest_sha256": self.manifest.sha256,
            "complete": self.complete,
            "status": self.status,
            "committed_steps": self.committed_steps,
            "unknown_steps": self.unknown_steps,
            "interrupted_by": self.interrupted_by,
            "ledger": self.ledger,
        }


@dataclass
class RecalibrationResult:
    event_id: str
    table: str
    column: str
    old_scale: int
    new_scale: int
    affected_columns: list[str]
    affected_steps: list[str]
    report: dict[str, Any]
    ledger: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "table": self.table,
            "column": self.column,
            "old_scale": self.old_scale,
            "new_scale": self.new_scale,
            "affected_columns": self.affected_columns,
            "affected_steps": self.affected_steps,
            "report": self.report,
            "ledger": self.ledger,
        }


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(
            "E_MIGRATE", f"cannot load manifest: {exc}", field_path="$"
        ) from exc
    if not isinstance(payload, dict):
        fail("E_MIGRATE", "manifest root must be a JSON object", field_path="$")
    return payload


def _manifest_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_manifest_from_plan = manifest_from_plan


class MigrationRunner:
    def __init__(
        self,
        db_path: str | Path,
        manifest: Manifest | str | Path,
        *,
        ledger_dir: str | Path | None = None,
        batch_size: int = 256,
    ):
        self.db_path = Path(db_path)
        self.manifest_input = manifest
        self.ledger_dir = Path(ledger_dir) if ledger_dir is not None else None
        self.batch_size = max(1, int(batch_size))
        self.manifest: Manifest | None = None
        self.raw_manifest: dict[str, Any] | None = None
        self.manifest_sha256: str | None = None

    def _load_raw(self) -> tuple[dict[str, Any], str]:
        if isinstance(self.manifest_input, Manifest):
            raw = self.manifest_input.raw
        else:
            raw = _read_json(self.manifest_input)
        return raw, _manifest_digest(raw)

    def _open(self) -> sqlite3.Connection:
        return connect(self.db_path)

    def _existing_run(self, connection: sqlite3.Connection) -> sqlite3.Row | None:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_mig_run'"
        ).fetchone()
        if row is None:
            return None
        return connection.execute("SELECT * FROM _mig_run WHERE run_id = 1").fetchone()

    def prepare(self) -> sqlite3.Connection:
        raw, digest = self._load_raw()
        self.raw_manifest = raw
        self.manifest_sha256 = digest
        connection = self._open()
        run = self._existing_run(connection)
        if run is not None:
            if str(run["manifest_sha256"]) != digest:
                connection.close()
                fail(
                    "E_MIGRATE",
                    "the existing ledger belongs to a different manifest",
                    step_id=None,
                    field_path="$.manifest_id",
                    existing_manifest_id=str(run["manifest_id"]),
                )
            plan = json.loads(str(run["plan_json"]))
            self.manifest = _manifest_from_plan(raw, plan, digest)
            return connection
        full_manifest = validate_manifest(raw, db_path=self.db_path)
        ensure_internal_tables(connection)
        revision = 0
        connection.execute("BEGIN IMMEDIATE")
        try:
            seed_initial_calibrations(connection, full_manifest, revision=revision)
            base_digest = schema_digest(connection)
            plan_json = canonical_json(full_manifest.to_dict())
            connection.execute(
                """
                INSERT INTO _mig_run(
                    run_id, manifest_id, manifest_sha256, plan_json,
                    base_schema_digest, status, current_step, revision
                ) VALUES (1, ?, ?, ?, ?, 'ready', NULL, 0)
                """,
                (full_manifest.manifest_id, digest, plan_json, base_digest),
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            connection.close()
            raise
        self.manifest = full_manifest
        return connection

    def _assert_prefix(self, connection: sqlite3.Connection) -> None:
        assert self.manifest is not None
        plan = list(self.manifest.execution_order)
        ledger = ledger_view(connection)["steps"]
        actual = [str(item["step_id"]) for item in ledger]
        expected = plan[: len(actual)]
        if actual != expected:
            fail(
                "E_MIGRATE",
                "committed ledger steps are not a prefix of the execution plan",
                step_id=actual[0] if actual else None,
                field_path="$.steps",
                committed=actual,
                expected_prefix=expected,
            )

    def status(self) -> dict[str, Any]:
        connection = self.prepare()
        try:
            assert self.manifest is not None
            self._assert_prefix(connection)
            ledger = ledger_view(connection)
            committed = [str(item["step_id"]) for item in ledger["steps"]]
            unknown = [
                step_id
                for step_id in self.manifest.execution_order
                if step_id not in set(committed)
            ]
            run = ledger["run"] or {}
            complete = len(unknown) == 0
            return {
                "complete": complete,
                "status": "complete" if complete else "incomplete",
                "committed_steps": committed,
                "unknown_steps": unknown,
                "requires_rerun": bool(unknown),
                "current_step": run.get("current_step"),
                "revision": ledger["revision"],
            }
        finally:
            connection.close()

    def run(
        self,
        *,
        batch_size: int | None = None,
        crash_after_step: Callable[[str], None] | None = None,
    ) -> MigrationRunResult:
        connection = self.prepare()
        assert self.manifest is not None
        effective_batch = max(1, int(batch_size or self.batch_size))
        try:
            self._assert_prefix(connection)
            committed = completed_step_ids(connection)
            plan = self.manifest.ordered_steps()
            for position, step in enumerate(plan):
                if step.step_id in committed:
                    continue
                start_revision = current_revision(connection)
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE _mig_run SET status='running', current_step=?, revision=? WHERE run_id=1",
                    (step.step_id, start_revision),
                )
                connection.execute("COMMIT")
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    before_columns = calibration_snapshot(connection)
                    row_count, operation_report = self._apply_operation(
                        connection,
                        step=step,
                        revision=start_revision + 1,
                        batch_size=effective_batch,
                    )
                    after_columns = calibration_snapshot(connection)
                    content_digest = table_content_digest(connection, step.table)
                    result = build_step_result(
                        connection,
                        step=step,
                        manifest=self.manifest,
                        operation_report=operation_report,
                    )
                    revision = current_revision(connection) + 1
                    record_step(
                        connection,
                        step=step,
                        before_columns=before_columns,
                        after_columns=after_columns,
                        row_count=row_count,
                        result=result,
                        content_digest=content_digest,
                        revision=revision,
                    )
                    connection.execute(
                        "UPDATE _mig_run SET status='running', current_step=NULL, revision=? WHERE run_id=1",
                        (revision,),
                    )
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    try:
                        connection.execute("BEGIN IMMEDIATE")
                        connection.execute(
                            "UPDATE _mig_run SET status='interrupted' WHERE run_id=1"
                        )
                        connection.execute("COMMIT")
                    except sqlite3.Error:
                        pass
                    raise
                committed.add(step.step_id)
                if self.ledger_dir is not None:
                    export_ledger(connection, ledger_dir=self.ledger_dir)
                if crash_after_step is not None:
                    crash_after_step(step.step_id)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE _mig_run SET status='complete', current_step=NULL, revision=? WHERE run_id=1",
                (current_revision(connection),),
            )
            connection.execute("COMMIT")
            ledger = ledger_view(connection)
            chain_ok, bad_step = verify_ledger_chain(connection)
            if not chain_ok:
                fail(
                    "E_MIGRATE",
                    "ledger hash chain verification failed",
                    step_id=bad_step,
                    field_path="$.ledger",
                )
            committed_list = [str(item["step_id"]) for item in ledger["steps"]]
            return MigrationRunResult(
                manifest=self.manifest,
                complete=True,
                status="complete",
                committed_steps=committed_list,
                unknown_steps=[],
                ledger=ledger,
            )
        finally:
            connection.close()

    def _apply_operation(
        self,
        connection: sqlite3.Connection,
        *,
        step: MigrationStep,
        revision: int,
        batch_size: int,
    ) -> tuple[int, dict[str, Any]]:
        assert self.manifest is not None
        try:
            if step.operation == "create_column":
                return self._create_column(connection, step, revision)
            if step.operation == "widen_column":
                return self._widen_column(connection, step, revision)
            if step.operation == "rename_column":
                return self._rename_column(connection, step, revision)
            if step.operation == "backfill":
                count, report = execute_backfill(
                    connection,
                    step=step,
                    manifest=self.manifest,
                    revision=revision,
                    batch_size=batch_size,
                )
                return count, report
            if step.operation == "add_constraint":
                return self._add_constraint(connection, step)
            if step.operation == "create_index":
                return self._create_index(connection, step)
        except MigrationError:
            raise
        except sqlite3.Error as exc:
            fail(
                "E_MIGRATE",
                f"SQLite operation failed: {exc}",
                step_id=step.step_id,
                field_path="operation",
            )
        raise AssertionError(f"unsupported operation {step.operation}")

    def _create_column(
        self, connection: sqlite3.Connection, step: MigrationStep, revision: int
    ) -> tuple[int, dict[str, Any]]:
        assert step.column is not None and step.create_type is not None
        nullability = "" if step.nullable else " NOT NULL"
        default = "" if step.default_sql is None else f" DEFAULT {step.default_sql}"
        connection.execute(
            f"ALTER TABLE {quote_identifier(step.table)} ADD COLUMN "
            f"{quote_identifier(step.column)} {step.create_type}{nullability}{default}"
        )
        upsert_column_meta(
            connection,
            table=step.table,
            column=step.column,
            sql_type=step.create_type,
            spec=step.quantization,
            step_id=step.step_id,
            revision=revision,
        )
        row_count = table_row_count(connection, step.table)
        return row_count, {
            "column": step.column,
            "type": step.create_type,
            "nullable": step.nullable,
            "default": step.default_sql,
        }

    def _widen_column(
        self, connection: sqlite3.Connection, step: MigrationStep, revision: int
    ) -> tuple[int, dict[str, Any]]:
        assert step.column is not None and step.to_type is not None and step.quantization is not None
        meta_map = read_column_meta(connection, table=step.table)
        old_meta = meta_map.get((step.table, step.column))
        if old_meta is None or old_meta.spec is None:
            fail(
                "E_MIGRATE",
                "widen target calibration is missing",
                step_id=step.step_id,
                field_path="column",
            )
        old_type = column_types(connection, step.table)[step.column]
        target_spec = step.quantization

        def transform(column: str, value: Any, table: str) -> Any:
            if column != step.column or value is None:
                return value
            if old_meta.sql_type == "REAL":
                new_units = quantize_decimal(value, target_spec, step_id=step.step_id).units
            else:
                try:
                    old_units = int(value)
                except (TypeError, ValueError):
                    old_units = quantize_decimal(
                        value, old_meta.spec, step_id=step.step_id
                    ).units
                new_units = rescale_units(
                    old_units,
                    old_meta.spec.scale,
                    target_spec.scale,
                    target_spec.truncation,
                )
            if step.to_type == "REAL":
                return float(Decimal(new_units) / Decimal(pow10(target_spec.scale)))
            return int(new_units)

        row_count = rebuild_table(
            connection,
            step.table,
            type_overrides={step.column: step.to_type},
            value_transform=transform,
        )
        updated = upsert_column_meta(
            connection,
            table=step.table,
            column=step.column,
            sql_type=step.to_type,
            spec=target_spec,
            step_id=step.step_id,
            revision=revision,
        )
        return row_count, {
            "column": step.column,
            "from_type": old_type,
            "to_type": step.to_type,
            "previous_calibration": old_meta.to_dict(),
            "updated_calibration": updated.to_dict(),
            "rows_copied": row_count,
        }

    def _rename_column(
        self, connection: sqlite3.Connection, step: MigrationStep, revision: int
    ) -> tuple[int, dict[str, Any]]:
        assert step.column is not None and step.new_name is not None
        old_name = step.column
        new_name = step.new_name
        meta_map = read_column_meta(connection, table=step.table)
        old_meta = meta_map.get((step.table, old_name))
        connection.execute(
            f"ALTER TABLE {quote_identifier(step.table)} "
            f"RENAME COLUMN {quote_identifier(old_name)} TO {quote_identifier(new_name)}"
        )
        if old_meta is not None:
            upsert_column_meta(
                connection,
                table=step.table,
                column=new_name,
                sql_type=old_meta.sql_type,
                spec=old_meta.spec,
                step_id=step.step_id,
                revision=revision,
            )
            connection.execute(
                "DELETE FROM _mig_column_meta WHERE table_name = ? AND column_name = ?",
                (step.table, old_name),
            )
        connection.execute(
            "UPDATE _mig_row_errors SET column_name = ? WHERE column_name = ?",
            (new_name, old_name),
        )
        pattern = re.compile(rf"(?<![A-Za-z0-9_\"]){re.escape(old_name)}(?![A-Za-z0-9_\"])")
        constraint_rows = connection.execute(
            "SELECT constraint_name, definition, columns_json FROM _mig_constraints WHERE table_name = ?",
            (step.table,),
        ).fetchall()
        for row in constraint_rows:
            definition = pattern.sub(new_name, str(row["definition"]))
            columns_payload = json.loads(str(row["columns_json"])) if row["columns_json"] else None
            if columns_payload:
                columns_payload = [new_name if item == old_name else item for item in columns_payload]
            connection.execute(
                "UPDATE _mig_constraints SET definition = ?, columns_json = ? WHERE constraint_name = ?",
                (
                    definition,
                    canonical_json(columns_payload) if columns_payload is not None else None,
                    str(row["constraint_name"]),
                ),
            )
        row_count = table_row_count(connection, step.table)
        return row_count, {
            "old_name": old_name,
            "new_name": new_name,
            "calibration": old_meta.to_dict() if old_meta else None,
        }

    def _add_constraint(
        self, connection: sqlite3.Connection, step: MigrationStep
    ) -> tuple[int, dict[str, Any]]:
        assert step.constraint_name is not None and step.constraint_kind is not None
        if step.constraint_kind == "check":
            expression = step.constraint_expression or ""
            check_expression_valid(connection, step.table, expression)
            violations = check_constraint_violations(connection, step.table, expression)
            if violations:
                fail(
                    "E_MIGRATE",
                    "check constraint is violated by existing data",
                    step_id=step.step_id,
                    field_path="constraint.expression",
                    violations=violations,
                )
            row_count = rebuild_table(
                connection,
                step.table,
                extra_checks=[expression],
            )
            connection.execute(
                """
                INSERT INTO _mig_constraints(
                    constraint_name, table_name, kind, definition, columns_json, created_step
                ) VALUES (?, ?, 'check', ?, NULL, ?)
                """,
                (step.constraint_name, step.table, expression, step.step_id),
            )
            return row_count, {
                "constraint": step.constraint_name,
                "kind": "check",
                "expression": expression,
                "rows_checked": row_count,
            }
        columns = step.raw["constraint"].get("columns", [])
        row_count = table_row_count(connection, step.table)
        try:
            connection.execute(
                f"CREATE UNIQUE INDEX {quote_identifier(step.constraint_name)} "
                f"ON {quote_identifier(step.table)} "
                f"({', '.join(quote_identifier(column) for column in columns)})"
            )
        except sqlite3.IntegrityError as exc:
            fail(
                "E_MIGRATE",
                f"unique constraint is violated by existing data: {exc}",
                step_id=step.step_id,
                field_path="constraint.columns",
            )
        connection.execute(
            """
            INSERT INTO _mig_constraints(
                constraint_name, table_name, kind, definition, columns_json, created_step
            ) VALUES (?, ?, 'unique', ?, ?, ?)
            """,
            (
                step.constraint_name,
                step.table,
                "UNIQUE (" + ", ".join(columns) + ")",
                canonical_json(columns),
                step.step_id,
            ),
        )
        return row_count, {
            "constraint": step.constraint_name,
            "kind": "unique",
            "columns": columns,
        }

    def _create_index(
        self, connection: sqlite3.Connection, step: MigrationStep
    ) -> tuple[int, dict[str, Any]]:
        assert step.index_name is not None
        unique_sql = "UNIQUE " if step.index_unique else ""
        connection.execute(
            f"CREATE {unique_sql}INDEX {quote_identifier(step.index_name)} "
            f"ON {quote_identifier(step.table)} "
            f"({', '.join(quote_identifier(column) for column in step.index_columns)})"
        )
        row_count = table_row_count(connection, step.table)
        return row_count, {
            "index": step.index_name,
            "columns": list(step.index_columns),
            "unique": step.index_unique,
            "rows_indexed": row_count,
        }


def _affected_columns(
    manifest: Manifest, start_column: str
) -> set[str]:
    reverse: dict[str, set[str]] = {}
    for target, source in manifest.derivation_edges:
        reverse.setdefault(source, set()).add(target)
    for step in manifest.steps:
        if step.operation != "backfill" or step.expression is None or step.column is None:
            continue
        target = f"{step.table}.{step.column}"
        for reference in step.expression.references:
            reverse.setdefault(f"{step.table}.{reference}", set()).add(target)
    affected = {start_column}
    queue = [start_column]
    while queue:
        source = queue.pop(0)
        for target in sorted(reverse.get(source, set())):
            if target not in affected:
                affected.add(target)
                queue.append(target)
    return affected


def perturb_scale(
    db_path: str | Path,
    manifest: Manifest,
    *,
    table: str,
    column: str,
    new_scale: int,
    new_truncation: str | None = None,
    batch_size: int = 256,
) -> RecalibrationResult:
    if isinstance(new_scale, bool) or not isinstance(new_scale, int) or not 0 <= new_scale <= 9:
        fail(
            "E_MIGRATE",
            "new_scale must be an integer from 0 to 9",
            step_id=None,
            field_path="new_scale",
        )
    connection = connect(db_path)
    try:
        meta_map = read_column_meta(connection, table=table)
        old_meta = meta_map.get((table, column))
        if old_meta is None or old_meta.spec is None:
            fail(
                "E_MIGRATE",
                "scale perturbation target has no numeric calibration",
                step_id=None,
                field_path=f"tables.{table}.columns.{column}",
            )
        new_spec = QuantizationSpec(
            scale=new_scale,
            truncation=new_truncation or old_meta.spec.truncation,
            saturation=old_meta.spec.saturation,
            min_units=old_meta.spec.min_units,
            max_units=old_meta.spec.max_units,
        )
        new_spec.validate(field_path="new_quantization")
        before_fingerprint = database_fingerprint(connection)
        before_digest = before_fingerprint["data_version"]
        revision = current_revision(connection) + 1
        event_id = f"recal:{table}.{column}:{old_meta.spec.scale}->{new_scale}:r{revision}"
        affected_column_keys = _affected_columns(manifest, f"{table}.{column}")
        affected_steps = []
        target_steps = {
            f"{step.table}.{step.column}": step
            for step in manifest.steps
            if step.operation == "backfill"
        }
        for step_id in manifest.execution_order:
            step = manifest.step(step_id)
            if (
                step.operation == "backfill"
                and f"{step.table}.{step.column}" in affected_column_keys
            ):
                affected_steps.append(step.step_id)
        connection.execute("BEGIN IMMEDIATE")
        try:
            if old_meta.sql_type in {"DECIMAL", "INTEGER"}:
                key_columns = row_key_columns(connection, table)
                for row in ordered_rows(
                    connection,
                    table,
                    columns=sorted({column, *key_columns}),
                    batch_size=max(1, batch_size),
                ):
                    key = row_key_for_row(row, key_columns)
                    try:
                        old_units = int(row[column])
                    except (TypeError, ValueError):
                        old_units = quantize_decimal(
                            row[column], old_meta.spec
                        ).units
                    new_units = rescale_units(
                        old_units, old_meta.spec.scale, new_scale, new_spec.truncation
                    )
                    predicate_parts: list[str] = []
                    predicate_values: list[Any] = []
                    for key_column in key_columns:
                        if key_column == "_rowid_":
                            predicate_parts.append("_rowid_ = ?")
                            predicate_values.append(row["_rowid_"])
                        else:
                            predicate_parts.append(f"{quote_identifier(key_column)} = ?")
                            predicate_values.append(row[key_column])
                    connection.execute(
                        f"UPDATE {quote_identifier(table)} SET {quote_identifier(column)} = ? "
                        f"WHERE {' AND '.join(predicate_parts)}",
                        [new_units, *predicate_values],
                    )
            updated_meta = upsert_column_meta(
                connection,
                table=table,
                column=column,
                sql_type=old_meta.sql_type,
                spec=new_spec,
                step_id=event_id,
                revision=revision,
            )
            backfill_reports: list[dict[str, Any]] = []
            for step_id in affected_steps:
                step = manifest.step(step_id)
                spec_override = new_spec if f"{step.table}.{step.column}" == f"{table}.{column}" else None
                count, report = execute_backfill(
                    connection,
                    step=step,
                    manifest=manifest,
                    revision=revision,
                    batch_size=batch_size,
                    quantization=spec_override,
                )
                report["rows_evaluated"] = count
                backfill_reports.append(report)
            after_digest = database_fingerprint(connection)["data_version"]
            report = {
                "event_id": event_id,
                "table": table,
                "column": column,
                "old_calibration": old_meta.to_dict(),
                "new_calibration": updated_meta.to_dict(),
                "affected_columns": sorted(affected_column_keys),
                "affected_steps": affected_steps,
                "backfills": backfill_reports,
                "before_digest": before_digest,
                "after_digest": after_digest,
            }
            record_recalibration(
                connection,
                event_id=event_id,
                table=table,
                column=column,
                old_scale=old_meta.spec.scale,
                new_scale=new_scale,
                old_truncation=old_meta.spec.truncation,
                new_truncation=new_spec.truncation,
                before_digest=before_digest,
                after_digest=after_digest,
                affected_steps=affected_steps,
                report=report,
                revision=revision,
            )
            connection.execute(
                "UPDATE _mig_run SET revision = ? WHERE run_id = 1", (revision,)
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        return RecalibrationResult(
            event_id=event_id,
            table=table,
            column=column,
            old_scale=old_meta.spec.scale,
            new_scale=new_scale,
            affected_columns=sorted(affected_column_keys),
            affected_steps=affected_steps,
            report=report,
            ledger=ledger_view(connection),
        )
    finally:
        connection.close()
