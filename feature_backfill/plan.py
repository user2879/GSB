"""Plan materialisation, edit application and rollback semantics."""

from __future__ import annotations

from typing import Any, Iterable

from .config_loader import plan_fingerprint
from .data_loader import parse_row_record, validate_entity_sequences
from .errors import E_CONFIG_INVALID, E_PLAN_INVALID, E_ROW_INVALID, make_failure
from .models import AppConfig, FeatureDef, PlanBatch, Row


def apply_operations(
    rows: list[Row],
    features: dict[str, FeatureDef],
    operations: Iterable[dict[str, Any]],
    *,
    config: AppConfig,
    raw_fields: Iterable[str],
) -> tuple[list[Row], dict[str, FeatureDef], list[dict[str, Any]], set[str]]:
    """Apply one batch and return affected entities.

    The operation list is atomic at the caller boundary: if any operation fails,
    the caller discards the returned state and keeps its previous snapshot.
    """

    new_rows = list(rows)
    new_features = dict(features)
    failures: list[dict[str, Any]] = []
    affected: set[str] = set()
    raw_field_set = tuple(raw_fields)
    for op_index, operation in enumerate(operations):
        kind = operation.get("op")
        if kind == "append":
            for item_index, record in enumerate(operation.get("rows", [])):
                row, row_failures = parse_row_record(
                    record,
                    line=-(op_index * 100_000 + item_index + 1),
                    config=config,
                    raw_fields=raw_field_set,
                )
                if row_failures:
                    for failure in row_failures:
                        failure["details"] = {
                            **failure.get("details", {}),
                            "operation": "append",
                            "batch_operation_index": op_index,
                        }
                    failures.extend(row_failures)
                    continue
                assert row is not None
                if any(existing.row_id == row.row_id for existing in new_rows):
                    failures.append(
                        make_failure(
                            E_ROW_INVALID,
                            f"append duplicates row_id {row.row_id}",
                            stage="load_validation",
                            entity_id=row.entity_id,
                            row_id=row.row_id,
                            details={"operation": "append", "batch_operation_index": op_index},
                        )
                    )
                    continue
                new_rows.append(row)
                affected.add(row.entity_id)
        elif kind == "correct":
            row_id = operation["row_id"]
            changes = dict(operation.get("changes", operation.get("set", {})))
            match_index = next(
                (index for index, row in enumerate(new_rows) if row.row_id == row_id),
                None,
            )
            if match_index is None:
                failures.append(
                    make_failure(
                        E_ROW_INVALID,
                        f"correction references unknown row_id {row_id}",
                        stage="load_validation",
                        row_id=row_id,
                        details={"operation": "correct", "batch_operation_index": op_index},
                    )
                )
                continue
            old_row = new_rows[match_index]
            metadata_changes = {
                key: value
                for key, value in changes.items()
                if key in {"row_id", "entity_id", "timestamp_ms"}
            }
            value_changes = {
                key: value for key, value in changes.items() if key not in metadata_changes
            }
            allowed_metadata = {"row_id", "entity_id", "timestamp_ms"}
            unknown_metadata = set(metadata_changes) - allowed_metadata
            if unknown_metadata:
                failures.append(
                    make_failure(
                        E_ROW_INVALID,
                        "correction contains unsupported metadata fields",
                        stage="load_validation",
                        row_id=row_id,
                        details={"fields": sorted(unknown_metadata)},
                    )
                )
                continue
            new_row_id = metadata_changes.get("row_id", old_row.row_id)
            new_entity_id = metadata_changes.get("entity_id", old_row.entity_id)
            new_timestamp = metadata_changes.get("timestamp_ms", old_row.timestamp_ms)
            replacement_record = old_row.to_dict()
            replacement_record.update(
                {
                    "row_id": new_row_id,
                    "entity_id": new_entity_id,
                    "timestamp_ms": new_timestamp,
                }
            )
            replacement_record.update(value_changes)
            replacement, row_failures = parse_row_record(
                replacement_record,
                line=-(op_index * 100_000 + 50_000),
                config=config,
                raw_fields=raw_field_set,
            )
            if row_failures:
                for failure in row_failures:
                    failure["details"] = {
                        **failure.get("details", {}),
                        "operation": "correct",
                        "batch_operation_index": op_index,
                    }
                failures.extend(row_failures)
                continue
            assert replacement is not None
            if new_row_id != old_row.row_id and any(
                row.row_id == new_row_id for row in new_rows
            ):
                failures.append(
                    make_failure(
                        E_ROW_INVALID,
                        f"correction duplicates row_id {new_row_id}",
                        stage="load_validation",
                        row_id=new_row_id,
                        details={"operation": "correct", "batch_operation_index": op_index},
                    )
                )
                continue
            new_rows[match_index] = replacement
            affected.add(old_row.entity_id)
            affected.add(replacement.entity_id)
        elif kind == "window_adjust":
            feature_name = operation["feature"]
            if feature_name not in new_features:
                failures.append(
                    make_failure(
                        E_FEATURE_DAG_INVALID,
                        f"window adjustment references unknown feature {feature_name}",
                        stage="feature_dag",
                        feature=feature_name,
                        details={"operation": "window_adjust"},
                    )
                )
                continue
            window = operation["window_ms"]
            if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
                failures.append(
                    make_failure(
                        E_CONFIG_INVALID,
                        "adjusted window_ms must be a positive integer",
                        stage="config",
                        feature=feature_name,
                        details={"window_ms": window},
                    )
                )
                continue
            new_features[feature_name] = new_features[feature_name].with_window(window)
            affected.update(row.entity_id for row in new_rows)
        elif kind == "rollback":
            failures.append(
                make_failure(
                    E_PLAN_INVALID,
                    "rollback must be handled by the plan materializer, not apply_operations",
                    stage="plan",
                    details={"operation_index": op_index},
                )
            )
        else:
            failures.append(
                make_failure(
                    E_PLAN_INVALID,
                    f"unknown plan operation {kind!r}",
                    stage="plan",
                    details={"operation_index": op_index},
                )
            )
    failures.extend(validate_entity_sequences(new_rows))
    return new_rows, new_features, failures, affected


def _batch_rows_for_cursor(
    base_rows: list[Row],
    base_features: dict[str, FeatureDef],
    batches: list[PlanBatch],
    cursor: int,
    *,
    config: AppConfig,
    raw_fields: Iterable[str],
) -> tuple[list[Row], dict[str, FeatureDef], list[dict[str, Any]], set[str]]:
    rows = list(base_rows)
    features = dict(base_features)
    failures: list[dict[str, Any]] = []
    affected: set[str] = set()
    for index in range(cursor + 1):
        batch = batches[index]
        rollback = next(
            (operation for operation in batch.operations if operation.get("op") == "rollback"),
            None,
        )
        if rollback is not None:
            if len(batch.operations) != 1:
                failures.append(
                    make_failure(
                        E_PLAN_INVALID,
                        "rollback batch must contain exactly one rollback operation",
                        stage="plan",
                        details={"batch_id": batch.batch_id},
                    )
                )
                continue
            target = rollback.get("to_batch")
            if target is not None:
                target_indices = [
                    idx for idx, candidate in enumerate(batches[:index]) if candidate.batch_id == target
                ]
                if not target_indices:
                    failures.append(
                        make_failure(
                            E_PLAN_INVALID,
                            f"rollback target batch {target} is not an earlier batch",
                            stage="plan",
                            details={"batch_id": batch.batch_id, "to_batch": target},
                        )
                    )
                    continue
                target_index = target_indices[-1]
            else:
                back = rollback.get("batches_back")
                if not isinstance(back, int) or isinstance(back, bool) or back <= 0 or back > index:
                    failures.append(
                        make_failure(
                            E_PLAN_INVALID,
                            "batches_back must point to an earlier batch",
                            stage="plan",
                            details={"batch_id": batch.batch_id, "batches_back": back},
                        )
                    )
                    continue
                target_index = index - back - 1
                if target_index < 0:
                    rows, features = list(base_rows), dict(base_features)
                else:
                    rows, features, sub_failures, _ = _batch_rows_for_cursor(
                        base_rows,
                        base_features,
                        batches,
                        target_index,
                        config=config,
                        raw_fields=raw_fields,
                    )
                    failures.extend(sub_failures)
                affected.update(row.entity_id for row in rows)
                affected.update(row.entity_id for row in base_rows)
                continue
            rows, features, sub_failures, _ = _batch_rows_for_cursor(
                base_rows,
                base_features,
                batches,
                target_index,
                config=config,
                raw_fields=raw_fields,
            )
            failures.extend(sub_failures)
            affected.update(row.entity_id for row in rows)
            affected.update(row.entity_id for row in base_rows)
            continue
        rows, features, batch_failures, batch_affected = apply_operations(
            rows,
            features,
            batch.operations,
            config=config,
            raw_fields=raw_fields,
        )
        failures.extend(batch_failures)
        affected.update(batch_affected)
    return rows, features, failures, affected


def materialize_plan_state(
    base_rows: list[Row],
    base_features: dict[str, FeatureDef],
    batches: list[PlanBatch],
    *,
    completed_batches: list[int] | None = None,
    config: AppConfig,
    raw_fields: Iterable[str],
) -> tuple[list[Row], dict[str, FeatureDef], list[dict[str, Any]]]:
    """Return base+plan state after the given completed batch ids.

    This is the independent oracle used to validate snapshot resumes and full
    backfill equivalence.
    """

    if completed_batches is None:
        cursor = len(batches) - 1
    else:
        completed = set(completed_batches)
        cursor = -1
        for index, batch in enumerate(batches):
            if batch.batch_id in completed:
                cursor = index
            else:
                break
    if cursor < 0:
        return list(base_rows), dict(base_features), []
    rows, features, failures, _ = _batch_rows_for_cursor(
        base_rows,
        base_features,
        batches,
        cursor,
        config=config,
        raw_fields=raw_fields,
    )
    return rows, features, failures


def plan_hash(batches: list[PlanBatch]) -> str:
    return plan_fingerprint(batches)
