from __future__ import annotations

import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json
from .clock import FrozenClock
from .engine import ConfigEngine, clients_from_payload, operation_from_payload, spec_from_payload
from .errors import DiagnosticError
from .structured_log import JsonlLogger, NullLogger


def _value_for_compare(record: Mapping[str, Any], field: str) -> Any:
    value = record.get(field)
    if field in ("basis", "formula"):
        return canonical_json(value)
    if field == "refs":
        return list(value)
    return value


def _raise_replay(key: str | None, field: str, ledger_value: Any, replay_value: Any, reason: str) -> None:
    raise DiagnosticError(
        "E_REPLAY",
        "逐键重放校验不一致",
        key=key,
        field=field,
        context={
            "ledger": ledger_value,
            "replay": replay_value,
            "basis": reason,
        },
    )


def recover_and_replay(
    engine: ConfigEngine,
    workspace: Path | str,
    *,
    logger: JsonlLogger | NullLogger | None = None,
) -> dict[str, Any]:
    engine._ensure_initialized()
    log = logger or engine.logger
    source_meta = engine.ledger.meta()
    unknown = engine.ledger.unresolved_unknown()
    committed = engine.ledger.journal_rows("committed")
    initial_spec = spec_from_payload(json.loads(source_meta["initial_spec_json"]))
    initial_clients = clients_from_payload(json.loads(source_meta["initial_clients_json"]))
    runtime = Path(workspace).resolve() / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    replay_engine: ConfigEngine | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="replay-", dir=str(runtime)) as tmp, ExitStack() as stack:
            replay_engine = stack.enter_context(
                ConfigEngine(
                    tmp,
                    db_path=Path(tmp) / "replay.sqlite3",
                    logger=NullLogger(),
                    clock=FrozenClock(source_meta["initialized_mono_ns"]),
                )
            )
            replay_engine.initialize(
                initial_spec,
                initial_clients,
                now_ns=source_meta["initialized_mono_ns"],
            )
            try:
                for journal in committed:
                    if journal["kind"] == "update":
                        operation = operation_from_payload(journal["payload"]["operation"])
                        replay_engine.apply_operation(
                            operation,
                            now_ns=journal["started_mono_ns"],
                        )
                    elif journal["kind"] == "expire":
                        replay_engine.collect_expired(
                            now_ns=journal["payload"]["reclaim_mono_ns"],
                            forced_keys=journal["payload"]["reclaimed"],
                        )
                    else:
                        raise DiagnosticError(
                            "E_REPLAY",
                            "账本包含不支持的变更类型",
                            key=journal.get("operation_id"),
                            field="kind",
                            context={"kind": journal["kind"]},
                        )
            except DiagnosticError as exc:
                if exc.code == "E_REPLAY":
                    raise
                raise DiagnosticError(
                    "E_REPLAY",
                    "重放执行失败，无法重建账本状态",
                    key=journal.get("operation_id"),
                    field="journal_replay",
                    context={
                        "original_code": exc.code,
                        "original_detail": exc.detail,
                        "original_context": exc.context,
                    },
                ) from exc

            source_records = engine.ledger.records()
            replay_records = replay_engine.ledger.records()
            for key in sorted(set(source_records) | set(replay_records)):
                if key not in source_records or key not in replay_records:
                    _raise_replay(
                        key,
                        "$record",
                        "<missing>" if key not in source_records else "present",
                        "<missing>" if key not in replay_records else "present",
                        "key_set_mismatch",
                    )
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
                    left = _value_for_compare(source_records[key], field)
                    right = _value_for_compare(replay_records[key], field)
                    if left != right:
                        _raise_replay(key, field, left, right, "replay_field_mismatch")

            source_meta_now = engine.ledger.meta()
            replay_meta = replay_engine.ledger.meta()
            for field in (
                "format_version",
                "config_set_version",
                "derivation_version",
                "change_cursor",
            ):
                if source_meta_now[field] != replay_meta[field]:
                    _raise_replay(None, field, source_meta_now[field], replay_meta[field], "replay_meta_mismatch")
            if source_meta_now["content_digest"] != replay_meta["content_digest"]:
                _raise_replay(
                    None,
                    "ledger.content_digest",
                    source_meta_now["content_digest"],
                    replay_meta["content_digest"],
                    "replay_ledger_digest_mismatch",
                )

            source_traces = engine.ledger.journal_rows("committed")
            replay_traces = replay_engine.ledger.journal_rows("committed")
            if len(source_traces) != len(replay_traces):
                _raise_replay(
                    None,
                    "invalidation_trace_count",
                    len(source_traces),
                    len(replay_traces),
                    "replay_trace_count_mismatch",
                )
            for index, (left_item, right_item) in enumerate(zip(source_traces, replay_traces)):
                if left_item["trace"] != right_item["trace"]:
                    first_key = None
                    if left_item["trace"] and left_item["trace"].get("affected"):
                        first_key = left_item["trace"]["affected"][0]
                    _raise_replay(
                        first_key,
                        "invalidation_trace",
                        left_item["trace"],
                        right_item["trace"],
                        f"replay_trace_mismatch_at_index_{index}_operation_{left_item['operation_id']}",
                    )

            source_view = engine.current_view()
            replay_view = replay_engine.current_view()
            for field in ("config_set_version", "derivation_version", "view_digest", "active", "invalid", "expired"):
                if source_view.get(field) != replay_view.get(field):
                    _raise_replay(None, f"current_view.{field}", source_view.get(field), replay_view.get(field), "replay_view_mismatch")

            result = {
                "ok": True,
                "code": "E_REPLAY_OK",
                "keys_compared": len(source_records),
                "operations_replayed": len(committed),
                "committed_change_cursor": source_meta_now["change_cursor"],
                "content_digest": source_meta_now["content_digest"],
                "unknown_transactions": [
                    {
                        "journal_id": item["journal_id"],
                        "operation_id": item["operation_id"],
                        "kind": item["kind"],
                        "requires_rerun": True,
                    }
                    for item in unknown
                ],
                "needs_rerun": bool(unknown),
                # Recovery never asserts that an exit report exists or can be
                # synthesized.  It is based solely on committed ledger rows and
                # committed change-journal entries.
                "exit_report_available": False,
                "exit_report_note": (
                    "进程被强制终止，未生成退出报告；仅按已提交记录恢复并要求重跑未知事务"
                    if unknown
                    else "恢复不生成退出报告；仅依据完整提交的账本记录与变更游标"
                ),
                "fields_compared": ["value", "content_digest", "version", "basis", "invalidation_trace"],
                "invalidation_traces_compared": len(source_traces),
            }
            log.emit(
                "E_REPLAY_OK",
                "replay_verified",
                message="逐键重放校验通过",
                keys_compared=result["keys_compared"],
                operations_replayed=result["operations_replayed"],
                needs_rerun=result["needs_rerun"],
                content_digest=result["content_digest"],
            )
            return result
    except DiagnosticError as exc:
        if exc.code == "E_REPLAY":
            log.emit(
                "E_REPLAY",
                "replay_mismatch",
                severity="ERROR",
                message=exc.detail,
                key=exc.key,
                field=exc.field,
                **exc.context,
            )
        raise
    finally:
        if replay_engine is not None:
            replay_engine.close()
