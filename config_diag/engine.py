from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .clock import MonotonicClock
from .compute import (
    build_initial_records,
    build_ledger_digest,
    build_view,
    compute_all,
    derive_record,
    snapshot_payload,
)
from .errors import DiagnosticError
from .graph import build_propagation, validate_acyclic
from .ledger import Ledger
from .model import (
    ChangeEdit,
    ChangeOperation,
    ChangeSequence,
    ConfigSpec,
    Formula,
    FormulaTerm,
    KeySpec,
    SubscriptionClient,
    SubscriptionSample,
)
from .structured_log import JsonlLogger, NullLogger
from .validation import load_config_spec, load_subscription_sample


def _formula_from_obj(obj: Mapping[str, Any]) -> Formula:
    if obj["op"] == "identity":
        return Formula(op="identity")
    return Formula(
        op=obj["op"],
        terms=tuple(
            FormulaTerm(ref=item["ref"], weight=int(item["weight"]))
            for item in obj["terms"]
        ),
        divisor=int(obj["divisor"]),
        bias=int(obj["bias"]),
    )


def spec_from_payload(payload: Mapping[str, Any]) -> ConfigSpec:
    keys = tuple(
        KeySpec(
            key=item["key"],
            value=int(item["value"]),
            version=int(item["version"]),
            ttl_ms=int(item["ttl_ms"]),
            refs=tuple(item["refs"]),
            formula=_formula_from_obj(item["formula"]),
        )
        for item in payload["keys"]
    )
    return ConfigSpec(
        schema_version=int(payload["schema_version"]),
        derivation_version=int(payload["derivation_version"]),
        keys=keys,
    )


def clients_from_payload(payload: Mapping[str, Any]) -> SubscriptionSample:
    return SubscriptionSample(
        schema_version=int(payload["schema_version"]),
        clients=tuple(
            SubscriptionClient(
                client_id=item["client_id"],
                seed=int(item["seed"]),
                poll_interval_ms=int(item["poll_interval_ms"]),
                keys=tuple(item["keys"]),
                sample_size=int(item["sample_size"]),
            )
            for item in payload["clients"]
        ),
    )


def operation_from_payload(payload: Mapping[str, Any]) -> ChangeOperation:
    edits = []
    for item in payload["changes"]:
        edits.append(
            ChangeEdit(
                key=item["key"],
                version=int(item["version"]),
                value=int(item["value"]) if "value" in item else None,
                ttl_ms=int(item["ttl_ms"]) if "ttl_ms" in item else None,
                refs=tuple(item["refs"]) if "refs" in item else None,
                formula=_formula_from_obj(item["formula"]) if "formula" in item else None,
            )
        )
    return ChangeOperation(
        operation_id=payload["operation_id"],
        sequence=int(payload["sequence"]),
        description=payload.get("description", ""),
        changes=tuple(edits),
    )


class ConfigEngine:
    def __init__(
        self,
        workspace: Path | str,
        *,
        db_path: Path | str | None = None,
        logger: JsonlLogger | NullLogger | None = None,
        clock: MonotonicClock | Any | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = self.workspace / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path is not None else self.runtime_dir / "ledger.sqlite3"
        if not self.db_path.is_absolute():
            self.db_path = self.workspace / self.db_path
        self.logger = logger or JsonlLogger(self.workspace / "logs" / "diagnostics.jsonl")
        self.clock = clock or MonotonicClock()
        self.ledger = Ledger(self.db_path, logger=self.logger)
        self.open_events = self.ledger.open_existing()
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.ledger.close()

    def __enter__(self) -> "ConfigEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _reset_database(self) -> str | None:
        if not self.db_path.exists():
            return None
        backup_dir = self.db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = backup_dir / f"{self.db_path.stem}.pre-reset.{time.monotonic_ns()}.sqlite3"
        self.ledger.conn.execute("PRAGMA wal_checkpoint(FULL)")
        shutil.copy2(self.db_path, target)
        os.chmod(target, 0o444)
        self.ledger.close()
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.db_path) + suffix)
            if candidate.exists():
                candidate.unlink()
        self.ledger = Ledger(self.db_path, logger=self.logger)
        self.open_events = []
        return str(target)

    def initialize_from_paths(
        self,
        spec_path: Path | str,
        clients_path: Path | str,
        *,
        reset: bool = False,
        now_ns: int | None = None,
    ) -> dict[str, Any]:
        spec = load_config_spec(spec_path)
        clients = load_subscription_sample(clients_path, set(spec.by_key))
        return self.initialize(spec, clients, reset=reset, now_ns=now_ns)

    def initialize(
        self,
        spec: ConfigSpec,
        clients: SubscriptionSample,
        *,
        reset: bool = False,
        now_ns: int | None = None,
    ) -> dict[str, Any]:
        reset_backup = None
        if reset and self.ledger.has_state():
            reset_backup = self._reset_database()
        if self.ledger.has_state():
            return {
                "ok": True,
                "already_initialized": True,
                "ledger": self.meta_summary(),
                "open_events": self.open_events,
            }
        moment = int(now_ns) if now_ns is not None else int(self.clock.now_ns())
        records = build_initial_records(spec, moment)
        result = self.ledger.initialize(spec, clients, records, moment)
        return {
            "ok": True,
            "initialized": True,
            "reset_backup": reset_backup,
            **result,
            "key_count": len(records),
            "open_events": self.open_events,
        }

    def _initial_models(self) -> tuple[ConfigSpec, SubscriptionSample]:
        meta = self.ledger.meta()
        spec = spec_from_payload(json.loads(meta["initial_spec_json"]))
        clients = clients_from_payload(json.loads(meta["initial_clients_json"]))
        return spec, clients

    def meta_summary(self) -> dict[str, Any]:
        meta = self.ledger.meta()
        return {
            "format_version": meta["format_version"],
            "ledger_version": meta["ledger_version"],
            "config_set_version": meta["config_set_version"],
            "derivation_version": meta["derivation_version"],
            "change_cursor": meta["change_cursor"],
            "content_digest": meta["content_digest"],
            "initialized_mono_ns": meta["initialized_mono_ns"],
            "last_mutation_mono_ns": meta["last_mutation_mono_ns"],
        }

    def _ensure_initialized(self) -> None:
        if not self.ledger.has_state():
            raise DiagnosticError("E_STATE", "账本尚未初始化，请先执行 init", field="ledger_meta")

    def _unknown_for_operation(self, operation_id: str) -> list[dict[str, Any]]:
        unknown = self.ledger.unresolved_unknown()
        return [item for item in unknown if item["operation_id"] != operation_id]

    def _max_view_version(self) -> int:
        row = self.ledger.conn.execute("SELECT COALESCE(MAX(view_version), 0) AS v FROM views").fetchone()
        return int(row["v"] if row else 0)

    def _new_view(
        self,
        records: Mapping[str, Mapping[str, Any]],
        meta: Mapping[str, Any],
        now_ns: int,
    ) -> dict[str, Any]:
        previous = self.ledger.current_view()
        return build_view(
            records,
            view_version=self._max_view_version() + 1,
            config_set_version=meta["config_set_version"],
            derivation_version=meta["derivation_version"],
            created_mono_ns=now_ns,
            previous_view=previous,
        )

    def _commit_mutation(
        self,
        *,
        meta: dict[str, Any],
        records: Mapping[str, Mapping[str, Any]],
        changed_keys: Iterable[str],
        view: Mapping[str, Any],
        journal_id: int,
        trace: Mapping[str, Any],
        now_ns: int,
        snapshot_reason: str,
        before_commit: Callable[[], None] | None = None,
        crash_point: str | None = None,
    ) -> None:
        changed = [records[key] for key in sorted(set(changed_keys))]
        with self.ledger.transaction():
            for record in changed:
                self.ledger._upsert_record(record)
            self.ledger._write_meta(meta)
            self.ledger._insert_current_view(view)
            self.ledger.mark_journal_committed(journal_id, meta, trace, now_ns)
            snap = snapshot_payload(meta, records, view)
            self.ledger._insert_snapshot(meta, snap, now_ns, snapshot_reason)
            if before_commit is not None:
                before_commit()
            if crash_point == "before-commit":
                os._exit(137)
        if crash_point == "after-commit":
            os._exit(137)

    @staticmethod
    def _edit_is_same_version_noop(edit: ChangeEdit, current: Mapping[str, Any] | None) -> bool:
        if current is None or edit.version != current["version"]:
            return False
        if edit.value is not None and edit.value != current["raw_value"]:
            return False
        if edit.ttl_ms is not None and edit.ttl_ms != current["ttl_ms"]:
            return False
        if edit.refs is not None and sorted(edit.refs) != sorted(current["refs"]):
            return False
        if edit.formula is not None and edit.formula.to_obj() != current["formula"]:
            return False
        return True

    def apply_operation(
        self,
        operation: ChangeOperation,
        *,
        now_ns: int | None = None,
        force_full_recompute: bool = False,
        before_commit: Callable[[], None] | None = None,
        crash_point: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        if crash_point not in (None, "after-intent", "before-commit", "after-commit"):
            raise DiagnosticError("E_STATE", "未知崩溃注入点", field="crash_point")

        existing = self.ledger.get_journal_by_operation(operation.operation_id)
        if existing and existing["status"] == "committed":
            return {
                "ok": True,
                "idempotent": True,
                "operation_id": operation.operation_id,
                "journal_id": existing["journal_id"],
                "ledger_version": self.ledger.meta()["ledger_version"],
                "trace": existing["trace"],
            }
        blocked = self._unknown_for_operation(operation.operation_id)
        if blocked:
            raise DiagnosticError(
                "E_PENDING",
                "存在未完成事务；必须先以原 operation_id 重跑，不能越过未知事务继续提交",
                field="change_journal.status",
                context={"unknown_operations": [item["operation_id"] for item in blocked]},
            )

        meta_before = self.ledger.meta()
        records_before = self.ledger.records()
        if existing is None and all(
            self._edit_is_same_version_noop(edit, records_before.get(edit.key))
            for edit in operation.changes
        ):
            view = self.current_view()
            return {
                "ok": True,
                "idempotent": True,
                "idempotency_reason": "same_key_version_same_payload",
                "operation_id": operation.operation_id,
                "journal_id": existing["journal_id"] if existing else None,
                "ledger_version": meta_before["ledger_version"],
                "config_set_version": meta_before["config_set_version"],
                "derivation_version": meta_before["derivation_version"],
                "affected": [],
                "view_version": view["view_version"],
                "view_digest": view["view_digest"],
            }

        max_seq = self.ledger.max_committed_source_sequence()
        if operation.sequence <= max_seq:
            raise DiagnosticError(
                "E_VERSION",
                "操作 sequence 必须大于已提交游标",
                field="sequence",
                context={"committed_sequence": max_seq, "incoming": operation.sequence},
            )

        moment = int(now_ns) if now_ns is not None else int(self.clock.now_ns())
        working = {
            key: {
                **record,
                "refs": list(record["refs"]),
                "formula": dict(record["formula"]),
                "basis": dict(record["basis"]),
            }
            for key, record in records_before.items()
        }

        direct_fields: dict[str, list[str]] = {}
        for edit in operation.changes:
            current = working.get(edit.key)
            if current is None:
                raise DiagnosticError("E_STATE", "变更引用未知键", key=edit.key, field="key")
            if current["status"] == "expired":
                raise DiagnosticError("E_STATE", "过期键不可直接编辑，需重新装载配置", key=edit.key, field="status")
            if edit.version < current["version"]:
                raise DiagnosticError(
                    "E_VERSION",
                    "键版本号必须单调递增",
                    key=edit.key,
                    field="version",
                    context={"current": current["version"], "incoming": edit.version},
                )
            if edit.version == current["version"]:
                if self._edit_is_same_version_noop(edit, current):
                    continue
                raise DiagnosticError(
                    "E_VERSION",
                    "同版本写入仅允许完全相同的幂等载荷",
                    key=edit.key,
                    field="version",
                    context={"current": current["version"], "incoming": edit.version},
                )
            current["version"] = edit.version
            if edit.value is not None:
                current["raw_value"] = edit.value
            if edit.ttl_ms is not None:
                current["ttl_ms"] = edit.ttl_ms
                current["expires_mono_ns"] = moment + edit.ttl_ms * 1_000_000
            if edit.refs is not None:
                current["refs"] = list(sorted(edit.refs))
            if edit.formula is not None:
                current["formula"] = edit.formula.to_obj()
            direct_fields[edit.key] = edit.changed_fields()

        refs_by_key = {key: tuple(record["refs"]) for key, record in working.items()}
        validate_acyclic(refs_by_key)
        affected, propagation = build_propagation(
            refs_by_key,
            direct_fields.keys(),
            direct_fields=direct_fields,
            invalid_status="recompute",
        )
        direct = set(direct_fields)
        for key in affected:
            record = working[key]
            if key not in direct:
                record["version"] += 1
            working[key] = derive_record(record, working)
        if force_full_recompute:
            working = compute_all(working)
            changed_keys = sorted(working)
        else:
            changed_keys = affected

        derivation_increment = 1 if any(edit.formula is not None for edit in operation.changes) else 0
        meta = dict(meta_before)
        meta["ledger_version"] = meta_before["ledger_version"] + 1
        meta["config_set_version"] = meta_before["config_set_version"] + 1
        meta["derivation_version"] = meta_before["derivation_version"] + derivation_increment
        meta["last_mutation_mono_ns"] = moment
        view = self._new_view(working, meta, moment)

        journal_payload = {"operation": operation.to_obj()}
        journal_id, is_new = self.ledger.prepare_journal(
            operation_id=operation.operation_id,
            kind="update",
            source_sequence=operation.sequence,
            payload=journal_payload,
            meta=meta_before,
            started_mono_ns=moment,
        )
        if not is_new:
            current = self.ledger.get_journal_by_operation(operation.operation_id)
            return {
                "ok": True,
                "idempotent": True,
                "operation_id": operation.operation_id,
                "journal_id": journal_id,
                "trace": current["trace"] if current else None,
            }
        if crash_point == "after-intent":
            os._exit(137)

        meta["change_cursor"] = journal_id
        meta["content_digest"] = build_ledger_digest(meta, working)
        view = build_view(
            working,
            view_version=view["view_version"],
            config_set_version=meta["config_set_version"],
            derivation_version=meta["derivation_version"],
            created_mono_ns=moment,
            previous_view=self.ledger.current_view(),
        )
        trace = {
            "kind": "update",
            "operation_id": operation.operation_id,
            "sequence": operation.sequence,
            "direct": sorted(direct),
            "direct_fields": {key: sorted(value) for key, value in sorted(direct_fields.items())},
            "affected": affected,
            "propagation": propagation,
            "config_set_version_before": meta_before["config_set_version"],
            "config_set_version_after": meta["config_set_version"],
            "derivation_version_before": meta_before["derivation_version"],
            "derivation_version_after": meta["derivation_version"],
        }
        self._commit_mutation(
            meta=meta,
            records=working,
            changed_keys=changed_keys,
            view=view,
            journal_id=journal_id,
            trace=trace,
            now_ns=moment,
            snapshot_reason=f"update:{operation.operation_id}",
            before_commit=before_commit,
            crash_point=crash_point,
        )
        self.logger.emit(
            "INFO_UPDATE",
            "operation_committed",
            message="增量变更已整体提交",
            operation_id=operation.operation_id,
            journal_id=journal_id,
            affected=affected,
            ledger_version=meta["ledger_version"],
            content_digest=meta["content_digest"],
        )
        return {
            "ok": True,
            "idempotent": False,
            "operation_id": operation.operation_id,
            "sequence": operation.sequence,
            "journal_id": journal_id,
            "ledger_version": meta["ledger_version"],
            "config_set_version": meta["config_set_version"],
            "derivation_version": meta["derivation_version"],
            "content_digest": meta["content_digest"],
            "direct": sorted(direct),
            "affected": affected,
            "propagation": propagation,
            "view_version": view["view_version"],
            "view_digest": view["view_digest"],
            "view_added": view["added"],
            "view_deleted": view["deleted"],
            "view_invalidated": view["invalidated"],
        }

    def apply_sequence(
        self,
        changes: ChangeSequence,
        *,
        sequence: int | None = None,
        all_operations: bool = False,
        now_ns: int | None = None,
        crash_point: str | None = None,
        rerun_unknown: bool = False,
    ) -> list[dict[str, Any]]:
        self._ensure_initialized()
        if not all_operations:
            if sequence is None:
                raise DiagnosticError("E_STATE", "必须指定 sequence 或 --all", field="sequence")
            ops = [item for item in changes.operations if item.sequence == sequence]
            if not ops:
                raise DiagnosticError("E_STATE", "变更序列中不存在该 sequence", field="sequence", context={"sequence": sequence})
        else:
            ops = list(changes.operations)
        results = []
        for index, operation in enumerate(ops):
            if index == 0 and crash_point is not None:
                results.append(
                    self.apply_operation(
                        operation,
                        now_ns=now_ns,
                        crash_point=crash_point,
                    )
                )
            else:
                results.append(self.apply_operation(operation, now_ns=now_ns))
        return results

    def collect_expired(
        self,
        *,
        now_ns: int | None = None,
        forced_keys: Iterable[str] | None = None,
        before_commit: Callable[[], None] | None = None,
        crash_point: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        blocked = self._unknown_for_operation("__expire_never_matches__")
        if blocked:
            raise DiagnosticError(
                "E_PENDING",
                "存在未完成事务；必须先重跑原操作，才能执行回收",
                field="change_journal.status",
                context={"unknown_operations": [item["operation_id"] for item in blocked]},
            )
        moment = int(now_ns) if now_ns is not None else int(self.clock.now_ns())
        meta_before = self.ledger.meta()
        records_before = self.ledger.records()
        if forced_keys is None:
            reclaimed = sorted(
                key
                for key, record in records_before.items()
                if record["status"] != "expired" and record["expires_mono_ns"] <= moment
            )
        else:
            reclaimed = sorted(set(forced_keys))
            missing = [key for key in reclaimed if key not in records_before]
            if missing:
                raise DiagnosticError("E_STATE", "回收集合引用未知键", key=missing[0], field="reclaimed")
        if not reclaimed:
            return {
                "ok": True,
                "reclaimed": [],
                "downstream": [],
                "affected": [],
                "reclaim_mono_ns": moment,
                "message": "本轮没有已过期键",
            }

        working = {
            key: {
                **record,
                "refs": list(record["refs"]),
                "formula": dict(record["formula"]),
                "basis": dict(record["basis"]),
            }
            for key, record in records_before.items()
        }
        refs_by_key = {key: tuple(record["refs"]) for key, record in working.items()}
        affected, propagation = build_propagation(
            refs_by_key,
            reclaimed,
            direct_fields={key: ["ttl_ms", "expires_mono_ns"] for key in reclaimed},
            invalid_status="expire",
        )
        reclaimed_set = set(reclaimed)
        for key in affected:
            record = working[key]
            record["version"] += 1
            if key in reclaimed_set:
                record["status"] = "expired"
                record["value"] = None
                record["basis"] = {
                    "op": "expired",
                    "reason": "ttl_elapsed",
                    "expires_mono_ns": record["expires_mono_ns"],
                    "reclaim_mono_ns": moment,
                }
            else:
                record["status"] = "invalid"
                record["value"] = None
                expired_ancestors = sorted(
                    ref for ref in record["refs"] if ref in reclaimed_set
                )
                record["basis"] = {
                    "op": "invalid",
                    "reason": "expired_dependency" if expired_ancestors else "expired_ancestor",
                    "expired_refs": expired_ancestors,
                    "reclaim_mono_ns": moment,
                }
            working[key] = derive_record(record, working)
        downstream = sorted(set(affected) - reclaimed_set)

        derivation_increment = 0
        meta = dict(meta_before)
        meta["ledger_version"] = meta_before["ledger_version"] + 1
        meta["config_set_version"] = meta_before["config_set_version"] + 1
        meta["derivation_version"] = meta_before["derivation_version"] + derivation_increment
        meta["last_mutation_mono_ns"] = moment
        view = self._new_view(working, meta, moment)
        base_operation_id = f"expire-{moment}"
        operation_id = base_operation_id
        suffix = 1
        while self.ledger.get_journal_by_operation(operation_id) is not None:
            suffix += 1
            operation_id = f"{base_operation_id}-{suffix}"
        journal_payload = {
            "reclaimed": reclaimed,
            "downstream": downstream,
            "affected": affected,
            "reclaim_mono_ns": moment,
        }
        journal_id, _ = self.ledger.prepare_journal(
            operation_id=operation_id,
            kind="expire",
            source_sequence=None,
            payload=journal_payload,
            meta=meta_before,
            started_mono_ns=moment,
        )
        if crash_point == "after-intent":
            os._exit(137)
        meta["change_cursor"] = journal_id
        meta["content_digest"] = build_ledger_digest(meta, working)
        view = build_view(
            working,
            view_version=view["view_version"],
            config_set_version=meta["config_set_version"],
            derivation_version=meta["derivation_version"],
            created_mono_ns=moment,
            previous_view=self.ledger.current_view(),
        )
        for item in propagation:
            item["action"] = "expired" if item["key"] in reclaimed_set else "invalidated"
        trace = {
            "kind": "expire",
            "operation_id": operation_id,
            "reclaim_mono_ns": moment,
            "reclaimed": reclaimed,
            "downstream": downstream,
            "affected": affected,
            "propagation": propagation,
            "config_set_version_before": meta_before["config_set_version"],
            "config_set_version_after": meta["config_set_version"],
            "derivation_version_before": meta_before["derivation_version"],
            "derivation_version_after": meta["derivation_version"],
        }
        self._commit_mutation(
            meta=meta,
            records=working,
            changed_keys=affected,
            view=view,
            journal_id=journal_id,
            trace=trace,
            now_ns=moment,
            snapshot_reason=f"expire:{operation_id}",
            before_commit=before_commit,
            crash_point=crash_point,
        )
        self.logger.emit(
            "INFO_EXPIRE",
            "expired_keys_reclaimed",
            message="单调时钟回收完成，下游已失效",
            reclaimed=reclaimed,
            downstream=downstream,
            reclaim_mono_ns=moment,
            ledger_version=meta["ledger_version"],
        )
        return {
            "ok": True,
            "operation_id": operation_id,
            "journal_id": journal_id,
            "reclaim_mono_ns": moment,
            "reclaimed": reclaimed,
            "downstream": downstream,
            "affected": affected,
            "propagation": propagation,
            "ledger_version": meta["ledger_version"],
            "config_set_version": meta["config_set_version"],
            "view_version": view["view_version"],
            "view_digest": view["view_digest"],
            "view_added": view["added"],
            "view_deleted": view["deleted"],
            "view_invalidated": view["invalidated"],
        }

    def current_view(self) -> dict[str, Any]:
        self._ensure_initialized()
        view = self.ledger.current_view()
        if view is None:
            raise DiagnosticError("E_STATE", "当前订阅视图不存在", field="views")
        return view

    def list_state(self) -> dict[str, Any]:
        self._ensure_initialized()
        records = self.ledger.records()
        keys = []
        for key in sorted(records):
            record = records[key]
            keys.append(
                {
                    "key": key,
                    "raw_value": record["raw_value"],
                    "value": record["value"],
                    "version": record["version"],
                    "status": record["status"],
                    "refs": sorted(record["refs"]),
                    "formula": record["formula"],
                    "ttl_ms": record["ttl_ms"],
                    "expires_mono_ns": record["expires_mono_ns"],
                    "content_digest": record["content_digest"],
                    "basis": record["basis"],
                }
            )
        views = self.ledger.views(limit=20)
        stale_count = sum(1 for item in views if item["status"] == "stale")
        snapshots = [
            {
                "snapshot_id": item["snapshot_id"],
                "ledger_version": item["ledger_version"],
                "config_set_version": item["config_set_version"],
                "derivation_version": item["derivation_version"],
                "change_cursor": item["change_cursor"],
                "content_digest": item["content_digest"],
                "created_mono_ns": item["created_mono_ns"],
                "reason": item["reason"],
            }
            for item in self.ledger.snapshots(limit=20)
        ]
        journal = self.ledger.journal_rows()
        stale_trace_count = sum(1 for item in journal if item.get("stale_since_view") is not None)
        latest_trace = None
        for item in reversed(journal):
            if item["trace"] is not None and item.get("stale_since_view") is None:
                latest_trace = item["trace"]
                break
        return {
            "ok": True,
            "ledger": self.meta_summary(),
            "keys": keys,
            "current_view": self.current_view(),
            "views": [
                {
                    "view_version": item["view_version"],
                    "config_set_version": item["config_set_version"],
                    "derivation_version": item["derivation_version"],
                    "status": item["status"],
                    "view_digest": item["view_digest"],
                    "created_mono_ns": item["created_mono_ns"],
                }
                for item in views
            ],
            "stale_view_count": stale_count,
            "stale_trace_count": stale_trace_count,
            "snapshots": snapshots,
            "latest_trace": latest_trace,
            "journal": [
                {
                    "journal_id": item["journal_id"],
                    "operation_id": item["operation_id"],
                    "kind": item["kind"],
                    "source_sequence": item["source_sequence"],
                    "status": item["status"],
                    "trace_stale": item.get("stale_since_view") is not None,
                    "stale_since_view": item.get("stale_since_view"),
                }
                for item in journal
            ],
            "open_events": self.open_events,
        }

    def debug_tamper_record(self, key: str, *, delta: int = 1, fail_digest: bool = False) -> None:
        """Test/demo hook to make the diverge and replay diagnostics observable."""

        records = self.ledger.records()
        if key not in records:
            raise KeyError(key)
        records[key]["value"] = int(records[key]["value"] or 0) + delta
        from .compute import seal_record

        seal_record(records[key])
        meta = self.ledger.meta()
        meta["content_digest"] = build_ledger_digest(meta, records)
        if fail_digest:
            meta["content_digest"] = "0" * 64
        with self.ledger.transaction():
            self.ledger._upsert_record(records[key])
            self.ledger._write_meta(meta)
