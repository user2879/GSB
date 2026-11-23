from __future__ import annotations

import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json
from .clock import FrozenClock
from .compute import record_bytes
from .engine import ConfigEngine
from .errors import DiagnosticError
from .structured_log import NullLogger
from .validation import load_bundle


def _comparison_value(record: Mapping[str, Any], field: str) -> Any:
    value = record.get(field)
    if field in ("basis", "formula") and not isinstance(value, str):
        return canonical_json(value)
    if field == "refs":
        return list(value)
    return value


def _first_record_difference(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    for key in sorted(set(left) | set(right)):
        if key not in left or key not in right:
            return {
                "key": key,
                "field": "$record",
                "left": "<missing>" if key not in left else "present",
                "right": "<missing>" if key not in right else "present",
            }
        for field in (
            "value",
            "content_digest",
            "version",
            "basis",
            "status",
            "refs",
            "formula",
            "expires_mono_ns",
        ):
            left_value = _comparison_value(left[key], field)
            right_value = _comparison_value(right[key], field)
            if left_value != right_value:
                return {
                    "key": key,
                    "field": field,
                    "left": left_value,
                    "right": right_value,
                }
    return None


def _first_trace_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(left) != len(right):
        return {
            "journal_index": min(len(left), len(right)),
            "field": "invalidation_trace_count",
            "left": len(left),
            "right": len(right),
        }
    for index, (left_item, right_item) in enumerate(zip(left, right)):
        left_trace = left_item.get("trace")
        right_trace = right_item.get("trace")
        if left_trace != right_trace:
            first_affected = None
            if isinstance(left_trace, dict) and left_trace.get("affected"):
                first_affected = left_trace["affected"][0]
            return {
                "journal_index": index,
                "journal_id": left_item.get("journal_id"),
                "operation_id": left_item.get("operation_id"),
                "field": "invalidation_trace",
                "key": first_affected,
                "left": left_trace,
                "right": right_trace,
            }
    return None


def compare_incremental_and_full(
    samples_dir: Path | str,
    workspace: Path | str,
    *,
    inject_diverge: bool = False,
    now_ns: int | None = None,
) -> dict[str, Any]:
    spec, clients, changes = load_bundle(samples_dir)
    runtime = Path(workspace).resolve() / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    moment = int(now_ns) if now_ns is not None else __import__("time").monotonic_ns()
    clock = FrozenClock(moment)
    inc_engine: ConfigEngine | None = None
    full_engine: ConfigEngine | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="compare-inc-", dir=str(runtime)) as tmp1, \
             tempfile.TemporaryDirectory(prefix="compare-full-", dir=str(runtime)) as tmp2, \
             ExitStack() as stack:
            inc_engine = stack.enter_context(
                ConfigEngine(
                    tmp1,
                    db_path=Path(tmp1) / "ledger.sqlite3",
                    logger=NullLogger(),
                    clock=clock,
                )
            )
            full_engine = stack.enter_context(
                ConfigEngine(
                    tmp2,
                    db_path=Path(tmp2) / "ledger.sqlite3",
                    logger=NullLogger(),
                    clock=clock,
                )
            )
            inc_engine.initialize(spec, clients, now_ns=moment)
            full_engine.initialize(spec, clients, now_ns=moment)
            independently_checked_sets = []
            for operation in changes.operations:
                incremental_result = inc_engine.apply_operation(operation, now_ns=moment)
                before_full = full_engine.ledger.records()
                full_engine.apply_operation(operation, now_ns=moment, force_full_recompute=True)
                after_full = full_engine.ledger.records()
                full_diff = {
                    key
                    for key in sorted(set(before_full) | set(after_full))
                    if key not in before_full
                    or key not in after_full
                    or record_bytes(before_full[key]) != record_bytes(after_full[key])
                }
                expected_set = set(incremental_result["affected"])
                if full_diff != expected_set:
                    missing = sorted(expected_set - full_diff)
                    extra = sorted(full_diff - expected_set)
                    raise DiagnosticError(
                        "E_DIVERGE",
                        "增量失效集合与全量前后快照差异不一致",
                        key=(missing + extra)[0] if (missing or extra) else None,
                        field="invalidation_set",
                        context={
                            "incremental": sorted(expected_set),
                            "full_rebuild": sorted(full_diff),
                            "missing_from_full_diff": missing,
                            "extra_in_full_diff": extra,
                            "operation_id": operation.operation_id,
                        },
                    )
                independently_checked_sets.append(
                    {"operation_id": operation.operation_id, "changed_keys": sorted(full_diff)}
                )

            if inject_diverge:
                candidate = sorted(
                    key for key, record in full_engine.ledger.records().items()
                    if record["status"] == "active"
                )[0]
                full_engine.debug_tamper_record(candidate, delta=1)

            left = inc_engine.ledger.records()
            right = full_engine.ledger.records()
            difference = _first_record_difference(left, right)
            if difference:
                raise DiagnosticError(
                    "E_DIVERGE",
                    "增量路径与全量重建路径首个逐键差异",
                    key=difference.get("key"),
                    field=difference.get("field"),
                    context={
                        "incremental": difference.get("left"),
                        "full_rebuild": difference.get("right"),
                        "cause": "field_mismatch",
                    },
                )
            inc_meta = inc_engine.ledger.meta()
            full_meta = full_engine.ledger.meta()
            for field in ("config_set_version", "derivation_version", "change_cursor", "ledger_version"):
                if inc_meta[field] != full_meta[field]:
                    raise DiagnosticError(
                        "E_DIVERGE",
                        "增量路径与全量重建路径元数据不一致",
                        key=None,
                        field=field,
                        context={"incremental": inc_meta[field], "full_rebuild": full_meta[field]},
                    )
            inc_journal = [
                item for item in inc_engine.ledger.journal_rows("committed")
            ]
            full_journal = [
                item for item in full_engine.ledger.journal_rows("committed")
            ]
            trace_diff = _first_trace_difference(inc_journal, full_journal)
            if trace_diff:
                raise DiagnosticError(
                    "E_DIVERGE",
                    "增量路径与全量重建路径失效轨迹不一致",
                    key=trace_diff.get("key"),
                    field=trace_diff.get("field"),
                    context={
                        "operation_id": trace_diff.get("operation_id"),
                        "incremental": trace_diff.get("left"),
                        "full_rebuild": trace_diff.get("right"),
                    },
                )
            return {
                "ok": True,
                "keys_compared": len(left),
                "operations_compared": len(inc_journal),
                "incremental_ledger_version": inc_meta["ledger_version"],
                "full_ledger_version": full_meta["ledger_version"],
                "content_digest": inc_meta["content_digest"],
                "invalidation_sets_equal": True,
                "independent_invalidation_sets_checked": independently_checked_sets,
                "fields_compared": [
                    "value",
                    "content_digest",
                    "version",
                    "basis",
                    "status",
                    "refs",
                    "formula",
                    "expires_mono_ns",
                ],
                "inject_diverge": inject_diverge,
            }
    finally:
        if inc_engine is not None:
            inc_engine.close()
        if full_engine is not None:
            full_engine.close()
