"""Coordinated incremental, authoritative, recovery and latency paths."""

from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import atomic_write_json, digest, sha256_bytes
from .errors import (
    E_DIVERGE,
    E_LATENCY,
    E_LEDGER,
    E_REF,
    E_REPLAY,
    E_SNAPSHOT,
    E_SPEC,
    AppError,
)
from .ledger import Ledger
from .logging_utils import StructuredLogger
from .plan import (
    PlanTree,
    apply_edit,
    compile_base,
    first_plan_bytes_difference,
    first_plan_difference,
    node_to_dict,
    plan_diff,
)
from .spec import (
    QueryDef,
    WorkspaceSpec,
    load_workspace,
    replace_query_sql,
    replace_scale_factor,
    replace_tables,
)
from .snapshots import create_snapshot, read_snapshot, snapshot_differences


@dataclass
class QueryState:
    query: QueryDef
    plan: PlanTree
    edits: list[dict[str, Any]]
    edit_seq: int
    epoch: int
    parameter_version: int
    sql_version: int
    edits_version: int
    versions: dict[str, int]
    view: dict[str, Any]
    last_batch_id: int | None = None
    last_cursor: int = 0


@dataclass(frozen=True)
class ReplayOutcome:
    query: QueryDef
    plan: PlanTree
    edits: tuple[dict[str, Any], ...]
    edit_seq: int
    epoch: int
    parameter_version: int
    versions: dict[str, int]
    last_batch_id: int | None
    last_cursor: int
    reports: tuple[dict[str, Any], ...]


def _view_digest(plan: PlanTree, parameter_version: int, compile_status: str) -> str:
    return digest(
        {
            "plan_bytes_sha256": sha256_bytes(plan.to_bytes()),
            "parameter_version": int(parameter_version),
            "compile_status": compile_status,
        }
    )


def _make_view(
    state: QueryState,
    *,
    compile_status: str = "READY",
    affected: Sequence[str] = (),
    propagation: Sequence[Mapping[str, Any]] = (),
    diff: Mapping[str, Any] | None = None,
    latency: Mapping[str, Any] | None = None,
    verification: Mapping[str, Any] | None = None,
    stale: bool = False,
) -> dict[str, Any]:
    root = node_to_dict(state.plan.root)
    nodes = [
        {
            "node_id": node.id,
            "kind": node.kind,
            "version": node.version,
            "summary": node.summary,
            "estimated_rows": node.estimated_rows,
            "children": [child.id for child in node.children],
        }
        for node in sorted(state.plan.nodes.values(), key=lambda item: item.id.encode("utf-8"))
    ]
    view = {
        "query_name": state.query.name,
        "sql": state.query.sql,
        "parameter_version": int(state.parameter_version),
        "versions": dict(state.versions),
        "epoch": int(state.epoch),
        "plan_summary": state.plan.summary,
        "estimated_rows": state.plan.estimated_rows,
        "compile_status": compile_status,
        "stale": bool(stale),
        "root": root,
        "nodes": nodes,
        "node_count": len(nodes),
        "affected_nodes": list(affected),
        "propagation": [dict(item) for item in propagation],
        "diff": dict(diff or {"added": [], "removed": [], "summary_changed": [], "changed_details": []}),
        "latency": dict(latency or {}),
        "verification": dict(verification or {}),
        "generated_mono_ns": time.monotonic_ns(),
        "view_digest": _view_digest(state.plan, state.parameter_version, compile_status),
        "last_batch_id": state.last_batch_id,
        "ledger_cursor": state.last_cursor,
    }
    return view


def _stale_view(view: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    stale = dict(view)
    stale["stale"] = True
    stale["compile_status"] = "STALE"
    stale["stale_reason"] = reason
    stale["active_in_summary"] = False
    return stale


class PlanEngine:
    def __init__(
        self,
        workspace: WorkspaceSpec | str | Path,
        *,
        ledger_path: str | Path,
        snapshot_dir: str | Path,
        log_path: str | Path | None = None,
        budget_ms: float | None = None,
    ) -> None:
        self.workspace = load_workspace(workspace) if not isinstance(workspace, WorkspaceSpec) else workspace
        self.ledger = Ledger(ledger_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.logger = StructuredLogger(log_path)
        self.budget_ms = float(self.workspace.interaction_budget_ms if budget_ms is None else budget_ms)
        if self.budget_ms < 0:
            raise ValueError("budget_ms must be non-negative")
        self.lock = threading.RLock()
        self._background_events: dict[int, threading.Event] = {}
        self._background_results: dict[int, dict[str, Any]] = {}
        self.stale_views: list[dict[str, Any]] = []
        self.states: dict[str, QueryState] = {}
        self.versions = dict(self.workspace.versions)
        self.parameter_counter = 1
        self.epoch = 1
        self.ledger.initialize()
        self.recovery = {
            "unknown_attempts": self.ledger.recover_unknown_attempts(self.workspace.workspace_id),
            "restored_queries": [],
            "replay_reports": [],
            "pending_verification_batches": [],
            "recovery_verifications": [],
            "exit_report_available": False,
            "note": "Only fully committed batches are restored; RUNNING attempts are marked UNKNOWN and must be rerun.",
        }
        if self.recovery["unknown_attempts"]:
            self.logger.emit(
                "recovery_unknown_attempts",
                code="E_UNKNOWN",
                details={"attempts": self.recovery["unknown_attempts"]},
            )
        self._initialise_states()
        self._complete_pending_verifications()
        self.logger.emit(
            "engine_ready",
            code="OK",
            details={
                "workspace_id": self.workspace.workspace_id,
                "queries": sorted(self.states),
                "ledger_cursor": self.ledger.latest_cursor(self.workspace.workspace_id),
                "unknown_attempts": len(self.recovery["unknown_attempts"]),
            },
        )

    @property
    def workspace_id(self) -> str:
        return self.workspace.workspace_id

    def _latest_version_snapshot(self) -> dict[str, int]:
        event = self.ledger.latest_parameter_event(self.workspace_id)
        if event is not None and isinstance(event.get("versions"), dict) and event["versions"]:
            versions = dict(self.workspace.versions)
            for key, value in event["versions"].items():
                if key in versions and isinstance(value, int) and value >= 1:
                    versions[key] = value
            return versions
        return dict(self.workspace.versions)

    def _initialise_states(self) -> None:
        self.versions = self._latest_version_snapshot()
        table_map = self.workspace.table_map()
        restored: dict[str, ReplayOutcome] = {}
        missing: list[tuple[QueryDef, PlanTree]] = []
        init_start = time.monotonic_ns()
        for query in self.workspace.queries:
            base = compile_base(
                query.parsed,
                table_map[query.parsed.table],
                self.workspace.scale_factor,
                query.name,
            )
            context = self.workspace.context_digest(query.name)
            records = self.ledger.batches_for_query(self.workspace_id, query.name, context)
            if records:
                replay_start_ns = time.monotonic_ns()
                outcome = self._replay_records(query, base, records)
                replay_end_ns = time.monotonic_ns()
                restored[query.name] = outcome
                self.ledger.record_verification(
                    workspace_id=self.workspace_id,
                    batch_id=outcome.last_batch_id,
                    mode="RESTORE",
                    status="MATCH",
                    elapsed_ms=(replay_end_ns - replay_start_ns) / 1_000_000.0,
                    expected_summary=outcome.plan.summary,
                    actual_summary=outcome.plan.summary,
                    first_diff=None,
                    mono_start_ns=replay_start_ns,
                    mono_end_ns=replay_end_ns,
                    update_batch=False,
                )
                self.recovery["restored_queries"].append(
                    {
                        "query": query.name,
                        "batches": len(records),
                        "last_batch_id": outcome.last_batch_id,
                        "ledger_cursor": outcome.last_cursor,
                        "edit_seq": outcome.edit_seq,
                    }
                )
            else:
                missing.append((query, base))

        max_pv = max(self.ledger.max_parameter_version(self.workspace_id), 1)
        existing_epoch = self.ledger.max_epoch(self.workspace_id)
        existing_cursor = self.ledger.latest_cursor(self.workspace_id)
        next_epoch = max(existing_epoch, 1)
        if restored:
            next_epoch = max(existing_epoch, max(item.epoch for item in restored.values()))
        elif existing_cursor > 0:
            next_epoch = existing_epoch + 1
        baseline_pv = max_pv if not restored and self.ledger.latest_cursor(self.workspace_id) == 0 else max_pv + (1 if missing else 0)
        baseline_records: list[dict[str, Any]] = []
        baseline_latencies: list[dict[str, Any]] = []
        compile_end = time.monotonic_ns()
        baseline_ms = (compile_end - init_start) / 1_000_000.0
        cursor = self.ledger.next_cursor(self.workspace_id)
        for index, (query, plan) in enumerate(missing):
            context = self.workspace.context_digest(query.name)
            query_pv = baseline_pv + index
            affected = sorted(plan.nodes)
            plan_bytes = plan.to_bytes()
            record = {
                "workspace_id": self.workspace_id,
                "query_name": query.name,
                "epoch": next_epoch,
                "cursor": cursor + index,
                "batch_kind": "BASE",
                "parameter_version": query_pv,
                "tables_version": self.versions["tables"],
                "sql_version": self.versions["sql"],
                "edits_version": self.versions["edits"],
                "scale_version": self.versions["scale"],
                "table_digest": self.workspace.table_digest(),
                "sql_digest": self.workspace.sql_digest(),
                "scale_digest": self.workspace.scale_digest(),
                "context_digest": context,
                "edit_seq_start": 0,
                "edit_seq_end": 0,
                "plan_summary": plan.summary,
                "plan_bytes": plan_bytes,
                "node_count": len(plan.nodes),
                "estimated_rows": plan.estimated_rows,
                "affected": affected,
                "propagation": [{"node": node_id, "reason": "baseline_compile"} for node_id in affected],
                "edits": [],
                "incremental_ms": baseline_ms,
                "full_ms": baseline_ms,
                "verify_status": "MATCH",
                "verify": {"status": "MATCH", "basis": "base SQL compiled from scratch"},
                "progress_cursor": cursor + index,
            }
            baseline_records.append(record)
            over = baseline_ms > self.budget_ms
            baseline_latencies.append(
                {
                    "workspace_id": self.workspace_id,
                    "kind": "baseline_compile",
                    "query_name": query.name,
                    "batch_index": index,
                    "code": E_LATENCY if over else "OK",
                    "within_budget": not over,
                    "budget_ms": self.budget_ms,
                    "elapsed_ms": baseline_ms,
                    "mono_start_ns": init_start,
                    "mono_end_ns": compile_end,
                    "truth_source": "ledger:latency_records",
                    "details": {"full_base_compile": True},
                }
            )
        if baseline_records:
            event = {
                "workspace_id": self.workspace_id,
                "event_type": "baseline",
                "parameter_version": max(record["parameter_version"] for record in baseline_records),
                "versions": dict(self.versions),
                "reason": "initial committed baseline",
                "details": {"queries": [record["query_name"] for record in baseline_records]},
                "mono_ns": compile_end,
            }
            batch_ids = self.ledger.commit_batches(
                baseline_records,
                latencies=baseline_latencies,
                parameter_event=event,
            )
            for index, ((query, plan), batch_id) in enumerate(zip(missing, batch_ids)):
                record = baseline_records[index]
                state = QueryState(
                    query=query,
                    plan=plan,
                    edits=[],
                    edit_seq=0,
                    epoch=next_epoch,
                    parameter_version=int(record["parameter_version"]),
                    sql_version=int(record["sql_version"]),
                    edits_version=int(record["edits_version"]),
                    versions=dict(self.versions),
                    view={},
                    last_batch_id=batch_id,
                    last_cursor=int(record["cursor"]),
                )
                state.view = _make_view(state, verification={"status": "MATCH", "basis": "baseline"})
                self.states[query.name] = state
        for query_name, outcome in restored.items():
            state = QueryState(
                query=outcome.query,
                plan=outcome.plan,
                edits=list(outcome.edits),
                edit_seq=outcome.edit_seq,
                epoch=outcome.epoch,
                parameter_version=outcome.parameter_version,
                sql_version=outcome.versions.get("sql", self.versions["sql"]),
                edits_version=outcome.versions.get("edits", self.versions["edits"]),
                versions=dict(outcome.versions),
                view={},
                last_batch_id=outcome.last_batch_id,
                last_cursor=outcome.last_cursor,
            )
            latest_verify = outcome.reports[-1].get("verify_status", "MATCH") if outcome.reports else "MATCH"
            compile_status = "READY" if latest_verify == "MATCH" else latest_verify
            state.view = _make_view(
                state,
                compile_status=compile_status,
                verification={"status": latest_verify, "basis": "ledger replay"},
            )
            self.states[query_name] = state
            self.recovery["replay_reports"].append(
                {"query": query_name, "status": "MATCH", "batches": len(outcome.reports)}
            )
        self.parameter_counter = max(
            self.ledger.max_parameter_version(self.workspace_id),
            max((state.parameter_version for state in self.states.values()), default=1),
        )
        self.epoch = max(
            self.ledger.max_epoch(self.workspace_id),
            max((state.epoch for state in self.states.values()), default=1),
        )

    def _complete_pending_verifications(self) -> None:
        """Finish committed batches whose background verdict was interrupted."""

        table_map = self.workspace.table_map()
        for query in self.workspace.queries:
            records = self.ledger.batches_for_query(
                self.workspace_id, query.name, self.workspace.context_digest(query.name)
            )
            pending = [record for record in records if record["verify_status"] == "PENDING"]
            if not pending:
                continue
            base = compile_base(
                query.parsed,
                table_map[query.parsed.table],
                self.workspace.scale_factor,
                query.name,
            )
            plan = base
            applied: list[dict[str, Any]] = []
            edit_seq = 0
            for record in records:
                if record["batch_kind"] == "EDIT":
                    for operation in record["edits"]:
                        edit_seq += 1
                        plan, _effect = apply_edit(plan, operation, edit_seq)
                        applied.append(copy.deepcopy(operation))
                if record["verify_status"] != "PENDING":
                    continue
                self.recovery["pending_verification_batches"].append(
                    {
                        "query_name": query.name,
                        "batch_id": record["batch_id"],
                        "edit_seq": int(record["edit_seq_start"]),
                    }
                )
                result = self._perform_verification(
                    batch_id=int(record["batch_id"]),
                    query=query,
                    table=base.table,
                    scale_factor=base.scale_factor,
                    edits=list(applied),
                    expected_plan=plan,
                    edit_seq=edit_seq,
                    mode="RECOVERY",
                )
                self.recovery["recovery_verifications"].append(result)
                self.logger.emit(
                    "recovery_verification_completed",
                    code=result["status"] if result["status"] != "MATCH" else "OK",
                    edit_seq=edit_seq,
                    details={"batch_id": record["batch_id"], "mode": "RECOVERY"},
                )

    def _next_parameter_version(self) -> int:
        self.parameter_counter = max(
            self.parameter_counter,
            self.ledger.max_parameter_version(self.workspace_id),
        ) + 1
        return self.parameter_counter

    def _replay_records(
        self,
        query: QueryDef,
        base: PlanTree,
        records: Sequence[Mapping[str, Any]],
    ) -> ReplayOutcome:
        plan = base
        edits: list[dict[str, Any]] = []
        edit_seq = 0
        reports: list[dict[str, Any]] = []
        first = True
        last_record: Mapping[str, Any] | None = None
        for record in records:
            if first:
                if record["batch_kind"] != "BASE":
                    raise AppError(
                        E_REPLAY,
                        "ledger history does not start with a committed baseline",
                        details={"batch_id": record["batch_id"], "edit_seq": 0, "node_id": "$"},
                    )
                first = False
            elif record["batch_kind"] == "BASE":
                raise AppError(
                    E_REPLAY,
                    "unexpected second baseline in ledger epoch",
                    details={"batch_id": record["batch_id"], "edit_seq": edit_seq, "node_id": "$"},
                )
            if record["batch_kind"] == "EDIT":
                expected_start = edit_seq + 1
                if int(record["edit_seq_start"]) != expected_start:
                    raise AppError(
                        E_REPLAY,
                        "non-contiguous edit sequence in ledger",
                        edit_seq=int(record["edit_seq_start"]),
                        node_id="$",
                        details={
                            "batch_id": record["batch_id"],
                            "expected_edit_seq": expected_start,
                            "actual_edit_seq": int(record["edit_seq_start"]),
                            "basis": "ledger edit sequence continuity",
                        },
                    )
                for operation in record["edits"]:
                    edit_seq += 1
                    try:
                        plan, _effect = apply_edit(plan, operation, edit_seq)
                    except AppError as error:
                        raise AppError(
                            E_REPLAY,
                            f"replay edit failed: {error.message}",
                            edit_seq=edit_seq,
                            node_id=error.node_id,
                            details={
                                "batch_id": record["batch_id"],
                                "code_during_replay": error.code,
                                "basis": "raw SQL plus ledger edit trace",
                            },
                        ) from error
                    edits.append(copy.deepcopy(operation))
                if int(record["edit_seq_end"]) != edit_seq:
                    raise AppError(
                        E_REPLAY,
                        "ledger edit sequence end does not match replayed operations",
                        edit_seq=edit_seq,
                        node_id="$",
                        details={"batch_id": record["batch_id"]},
                    )
            actual_bytes = plan.to_bytes()
            difference = first_plan_bytes_difference(record["plan_bytes"], actual_bytes)
            if difference is not None or plan.summary != record["plan_summary"]:
                raise AppError(
                    E_REPLAY,
                    "replayed plan differs from committed ledger plan",
                    edit_seq=int(record["edit_seq_start"]),
                    node_id=(difference or {}).get("node_id"),
                    details={
                        "batch_id": record["batch_id"],
                        "query_name": query.name,
                        "first_divergence": difference,
                        "expected_summary": record["plan_summary"],
                        "actual_summary": plan.summary,
                        "basis": "raw SQL plus ordered committed edit trace",
                    },
                )
            reports.append(
                {
                    "batch_id": record["batch_id"],
                    "cursor": record["cursor"],
                    "edit_seq_start": record["edit_seq_start"],
                    "edit_seq_end": record["edit_seq_end"],
                    "plan_summary": record["plan_summary"],
                    "verify_status": record["verify_status"],
                }
            )
            last_record = record
        if last_record is None:
            raise AppError(E_REPLAY, "empty ledger history", node_id="$")
        return ReplayOutcome(
            query=query,
            plan=plan,
            edits=tuple(edits),
            edit_seq=edit_seq,
            epoch=int(last_record["epoch"]),
            parameter_version=int(last_record["parameter_version"]),
            versions={
                "tables": int(last_record["tables_version"]),
                "sql": int(last_record["sql_version"]),
                "edits": int(last_record["edits_version"]),
                "scale": int(last_record["scale_version"]),
            },
            last_batch_id=int(last_record["batch_id"]),
            last_cursor=int(last_record["cursor"]),
            reports=tuple(reports),
        )

    def _resolve_query(self, query_name: str | None) -> QueryState:
        if query_name is None:
            query_name = self.workspace.queries[0].name if self.workspace.queries else ""
        if query_name not in self.states:
            raise AppError(E_REF, f"unknown query {query_name!r}", node_id="$")
        return self.states[query_name]

    def current_view(self, query_name: str | None = None) -> dict[str, Any]:
        with self.lock:
            state = self._resolve_query(query_name)
            if state.view.get("stale"):
                raise AppError(E_LEDGER, "current view was marked stale and must be replaced")
            return copy.deepcopy(state.view)

    def all_views(self) -> dict[str, dict[str, Any]]:
        with self.lock:
            return {name: copy.deepcopy(state.view) for name, state in sorted(self.states.items())}

    def next_edit_seq(self, query_name: str | None = None) -> int:
        with self.lock:
            return self._resolve_query(query_name).edit_seq + 1

    def table_declaration(self) -> list[dict[str, Any]]:
        return [table.to_dict() for table in self.workspace.tables]

    def export_workspace(self, path: str | Path) -> dict[str, Any]:
        with self.lock:
            current = replace(self.workspace, versions=dict(self.versions))
            target = Path(path)
            document = current.to_workspace_dict()
            atomic_write_json(target, document)
            result = {
                "ok": True,
                "path": str(target),
                "spec_digest": current.spec_digest(),
                "versions": dict(current.versions),
                "queries": [query.name for query in current.queries],
            }
            self.logger.emit("workspace_exported", code="OK", details=result)
            return result

    def _versioned_workspace(self) -> WorkspaceSpec:
        """Use current monotonic versions, not the stale file-declared baseline."""

        return replace(self.workspace, versions=dict(self.versions))

    def _compile_full(self, state: QueryState, edits: Sequence[Mapping[str, Any]] | None = None) -> PlanTree:
        query = state.query
        table = self.workspace.table_map()[query.parsed.table]
        plan = compile_base(query.parsed, table, self.workspace.scale_factor, query.name)
        for index, operation in enumerate(state.edits if edits is None else edits, start=1):
            plan, _effect = apply_edit(plan, operation, index)
        return plan

    def submit_edits(
        self,
        edits: Sequence[Mapping[str, Any]],
        *,
        query_name: str | None = None,
        background_verify: bool = False,
    ) -> dict[str, Any]:
        with self.lock:
            state = self._resolve_query(query_name)
            if not isinstance(edits, (list, tuple)) or not edits:
                error = AppError(E_SPEC, "edits must be a non-empty array", json_path="$.edits")
                self.logger.error(error, "edit_rejected", edit_seq=state.edit_seq + 1, query=state.query.name)
                raise error
            normalized_edits: list[dict[str, Any]] = []
            for index, edit in enumerate(edits):
                if not isinstance(edit, Mapping):
                    error = AppError(E_SPEC, "each edit must be an object", json_path=f"$.edits[{index}]")
                    self.logger.error(error, "edit_rejected", edit_seq=state.edit_seq + index + 1, query=state.query.name)
                    raise error
                normalized_edits.append(copy.deepcopy(dict(edit)))

            edit_seq_start = state.edit_seq + 1
            prospective_pv = self._next_parameter_version()
            cursor = self.ledger.next_cursor(self.workspace_id)
            interaction_start_ns = time.monotonic_ns()
            attempt_id = self.ledger.begin_attempt(
                workspace_id=self.workspace_id,
                cursor=cursor,
                query_name=state.query.name,
                edits=normalized_edits,
                parameter_version=prospective_pv,
                next_edit_seq=edit_seq_start,
            )
            compile_start_ns = time.monotonic_ns()
            plan = state.plan
            history = list(state.edits)
            effects: list[dict[str, Any]] = []
            try:
                for index, operation in enumerate(normalized_edits):
                    seq = edit_seq_start + index
                    plan, effect = apply_edit(plan, operation, seq)
                    effects.append(effect)
                    history.append(copy.deepcopy(operation))
            except AppError as error:
                self.ledger.fail_attempt(attempt_id, error.code, error.to_dict())
                self.logger.error(
                    error,
                    "edit_rejected",
                    edit_seq=error.edit_seq or edit_seq_start,
                    query=state.query.name,
                )
                raise
            except Exception as error:  # defensive boundary, still records the failed intent
                self.ledger.fail_attempt(attempt_id, "E_INTERNAL", {"message": str(error)})
                wrapped = AppError("E_INTERNAL", str(error), edit_seq=edit_seq_start)
                self.logger.error(wrapped, "edit_rejected", edit_seq=edit_seq_start, query=state.query.name)
                raise wrapped from error

            compile_end_ns = time.monotonic_ns()
            incremental_ms = (compile_end_ns - compile_start_ns) / 1_000_000.0
            interaction_ms = (compile_end_ns - interaction_start_ns) / 1_000_000.0
            edit_seq_end = edit_seq_start + len(normalized_edits) - 1
            affected = sorted({node_id for effect in effects for node_id in effect["affected_nodes"]})
            new_versions = dict(self.versions)
            new_versions["edits"] = int(new_versions.get("edits", 1)) + 1
            new_state = QueryState(
                query=state.query,
                plan=plan,
                edits=history,
                edit_seq=edit_seq_end,
                epoch=state.epoch,
                parameter_version=prospective_pv,
                sql_version=int(new_versions["sql"]),
                edits_version=int(new_versions["edits"]),
                versions=new_versions,
                view={},
                last_batch_id=None,
                last_cursor=cursor,
            )
            diff = plan_diff(state.plan, plan)
            latency_payload = {
                "workspace_id": self.workspace_id,
                "kind": "edit_compile",
                "query_name": state.query.name,
                "batch_index": 0,
                "attempt_id": attempt_id,
                "code": E_LATENCY if interaction_ms > self.budget_ms else "OK",
                "within_budget": interaction_ms <= self.budget_ms,
                "budget_ms": self.budget_ms,
                "elapsed_ms": interaction_ms,
                "mono_start_ns": interaction_start_ns,
                "mono_end_ns": compile_end_ns,
                "truth_source": "ledger:latency_records",
                "details": {
                    "edit_seq_start": edit_seq_start,
                    "edit_seq_end": edit_seq_end,
                    "incremental_ms": incremental_ms,
                    "affected_nodes": affected,
                    "full_recompile_triggered": False,
                },
            }
            context = self.workspace.context_digest(state.query.name)
            plan_bytes = plan.to_bytes()
            batch_record = {
                "workspace_id": self.workspace_id,
                "query_name": state.query.name,
                "epoch": state.epoch,
                "cursor": cursor,
                "batch_kind": "EDIT",
                "parameter_version": prospective_pv,
                "tables_version": new_versions["tables"],
                "sql_version": new_versions["sql"],
                "edits_version": new_versions["edits"],
                "scale_version": new_versions["scale"],
                "table_digest": self.workspace.table_digest(),
                "sql_digest": self.workspace.sql_digest(),
                "scale_digest": self.workspace.scale_digest(),
                "context_digest": context,
                "edit_seq_start": edit_seq_start,
                "edit_seq_end": edit_seq_end,
                "plan_summary": plan.summary,
                "plan_bytes": plan_bytes,
                "node_count": len(plan.nodes),
                "estimated_rows": plan.estimated_rows,
                "affected": affected,
                "propagation": effects,
                "edits": normalized_edits,
                "incremental_ms": incremental_ms,
                "full_ms": None,
                "verify_status": "PENDING",
                "verify": {"status": "PENDING"},
                "progress_cursor": cursor,
            }
            parameter_event = {
                "workspace_id": self.workspace_id,
                "event_type": "edit_trace",
                "parameter_version": prospective_pv,
                "versions": new_versions,
                "reason": "edit batch submitted",
                "details": {
                    "query_name": state.query.name,
                    "edit_seq_start": edit_seq_start,
                    "edit_seq_end": edit_seq_end,
                },
                "mono_ns": compile_end_ns,
            }
            try:
                batch_ids = self.ledger.commit_batches(
                    [batch_record],
                    latencies=[latency_payload],
                    parameter_event=parameter_event,
                    attempt_id=attempt_id,
                    stale_before_parameter_version=(self.workspace_id, state.query.name, prospective_pv),
                )
            except AppError:
                self.ledger.fail_attempt(attempt_id, E_LEDGER, {"message": "batch commit failed"})
                raise
            batch_id = batch_ids[0]
            new_state.last_batch_id = batch_id
            new_state.view = _make_view(
                new_state,
                compile_status="READY_VERIFY_PENDING",
                affected=affected,
                propagation=effects,
                diff=diff,
                latency={**latency_payload, "code": latency_payload["code"]},
                verification={"status": "PENDING"},
            )
            old_stale = _stale_view(state.view, reason="new parameter_version replaced the whole view")
            self.stale_views.append(old_stale)
            self.states[state.query.name] = new_state
            self.versions = new_versions
            if interaction_ms > self.budget_ms:
                self.logger.emit(
                    "edit_latency_over_budget",
                    code=E_LATENCY,
                    edit_seq=edit_seq_end,
                    details={
                        "elapsed_ms": interaction_ms,
                        "incremental_ms": incremental_ms,
                        "budget_ms": self.budget_ms,
                        "truth_source": "ledger:latency_records",
                    },
                )
            self.logger.emit(
                "edit_committed",
                code="OK",
                edit_seq=edit_seq_end,
                details={
                    "batch_id": batch_id,
                    "affected_nodes": affected,
                    "incremental_ms": incremental_ms,
                    "plan_summary": plan.summary,
                },
            )

            result = {
                "ok": True,
                "batch_id": batch_id,
                "query_name": state.query.name,
                "edit_seq_start": edit_seq_start,
                "edit_seq_end": edit_seq_end,
                "parameter_version": prospective_pv,
                "view": copy.deepcopy(new_state.view),
                "effects": effects,
                "incremental_ms": incremental_ms,
                "latency_code": latency_payload["code"],
                "verification": {"status": "PENDING"},
            }
            if background_verify:
                gate = threading.Event()
                self._background_events[batch_id] = gate
                thread = threading.Thread(
                    target=self._background_verify,
                    args=(
                        batch_id,
                        state.query,
                        state.plan.table,
                        state.plan.scale_factor,
                        copy.deepcopy(history),
                        plan,
                        edit_seq_end,
                        gate,
                    ),
                    name=f"verify-batch-{batch_id}",
                    daemon=True,
                )
                thread.start()
                result["verification"] = {"status": "PENDING", "background": True}
            else:
                verification = self._perform_verification(
                    batch_id=batch_id,
                    query=state.query,
                    table=state.plan.table,
                    scale_factor=state.plan.scale_factor,
                    edits=history,
                    expected_plan=plan,
                    edit_seq=edit_seq_end,
                    mode="SYNC",
                )
                result["verification"] = verification
                result["view"] = self.current_view(state.query.name)
                if verification["status"] == E_DIVERGE:
                    raise AppError(
                        E_DIVERGE,
                        "incremental plan differs from full recompilation",
                        edit_seq=edit_seq_end,
                        node_id=verification.get("first_diff", {}).get("node_id"),
                        details=verification,
                    )
            return result

    def _background_verify(
        self,
        batch_id: int,
        query: QueryDef,
        table: Any,
        scale_factor: float,
        edits: Sequence[Mapping[str, Any]],
        expected_plan: PlanTree,
        edit_seq: int,
        gate: threading.Event,
    ) -> None:
        try:
            result = self._perform_verification(
                batch_id=batch_id,
                query=query,
                table=table,
                scale_factor=scale_factor,
                edits=edits,
                expected_plan=expected_plan,
                edit_seq=edit_seq,
                mode="BACKGROUND",
            )
            self._background_results[batch_id] = result
        except Exception as error:  # background failures must never terminate Tk
            self._background_results[batch_id] = {
                "status": "E_INTERNAL",
                "message": str(error),
                "batch_id": batch_id,
            }
            self.logger.emit(
                "background_verify_failed",
                code=getattr(error, "code", "E_INTERNAL"),
                edit_seq=edit_seq,
                details={"batch_id": batch_id, "message": str(error)},
            )
        finally:
            gate.set()

    def wait_for_verification(self, batch_id: int, timeout: float | None = None) -> dict[str, Any] | None:
        gate = self._background_events.get(batch_id)
        if gate is None:
            return self._background_results.get(batch_id)
        gate.wait(timeout)
        return self._background_results.get(batch_id)

    def _perform_verification(
        self,
        *,
        batch_id: int,
        query: QueryDef,
        table: Any,
        scale_factor: float,
        edits: Sequence[Mapping[str, Any]],
        expected_plan: PlanTree,
        edit_seq: int,
        mode: str,
    ) -> dict[str, Any]:
        start_ns = time.monotonic_ns()
        full_plan: PlanTree | None = None
        failure: dict[str, Any] | None = None
        try:
            full_plan = compile_base(query.parsed, table, scale_factor, query.name)
            for index, operation in enumerate(edits, start=1):
                full_plan, _effect = apply_edit(full_plan, operation, index)
            failure = first_plan_difference(expected_plan, full_plan)
        except AppError as error:
            full_plan = None
            failure = {
                "node_id": error.node_id or "$",
                "reason": "full_recompile_error",
                "code": error.code,
                "message": error.message,
            }
        end_ns = time.monotonic_ns()
        elapsed_ms = (end_ns - start_ns) / 1_000_000.0
        actual_summary = full_plan.summary if full_plan is not None else "$ERROR"
        status = "MATCH" if failure is None else E_DIVERGE
        self.ledger.record_verification(
            workspace_id=self.workspace_id,
            batch_id=batch_id,
            mode=mode,
            status=status,
            elapsed_ms=elapsed_ms,
            expected_summary=expected_plan.summary,
            actual_summary=actual_summary,
            first_diff=failure,
            mono_start_ns=start_ns,
            mono_end_ns=end_ns,
            update_batch=True,
        )
        self.ledger.record_latency(
            {
                "workspace_id": self.workspace_id,
                "kind": "full_recompile",
                "query_name": query.name,
                "batch_id": batch_id,
                "code": "OK" if failure is None else E_DIVERGE,
                "within_budget": elapsed_ms <= self.budget_ms,
                "budget_ms": self.budget_ms,
                "elapsed_ms": elapsed_ms,
                "mono_start_ns": start_ns,
                "mono_end_ns": end_ns,
                "truth_source": "ledger:latency_records",
                "details": {
                    "mode": mode,
                    "expected_summary": expected_plan.summary,
                    "actual_summary": actual_summary,
                    "first_diff": failure,
                },
            }
        )
        result = {
            "status": status,
            "batch_id": batch_id,
            "mode": mode,
            "full_ms": elapsed_ms,
            "expected_summary": expected_plan.summary,
            "actual_summary": actual_summary,
            "first_diff": failure,
            "truth_source": "ledger:verification_records",
        }
        with self.lock:
            state = self.states.get(query.name)
            if state is not None and state.last_batch_id == batch_id:
                new_view = dict(state.view)
                new_view["verification"] = dict(result)
                new_view["compile_status"] = "READY" if status == "MATCH" else E_DIVERGE
                new_view["view_digest"] = _view_digest(state.plan, state.parameter_version, new_view["compile_status"])
                state.view = new_view
        if failure is None:
            self.logger.emit(
                "plan_verification_match",
                code="OK",
                edit_seq=edit_seq,
                details={"batch_id": batch_id, "mode": mode, "full_ms": elapsed_ms, "summary": expected_plan.summary},
            )
        else:
            self.logger.emit(
                "plan_verification_diverge",
                code=E_DIVERGE,
                edit_seq=edit_seq,
                node_id=failure.get("node_id"),
                details={"batch_id": batch_id, "mode": mode, "first_diff": failure},
            )
        return result

    def quick_query(self, query_name: str | None = None) -> dict[str, Any]:
        start_ns = time.monotonic_ns()
        with self.lock:
            try:
                state = self._resolve_query(query_name)
                view = copy.deepcopy(state.view)
                status = view.get("compile_status", "NOT_READY")
                payload = {
                    "ok": True,
                    "query_name": state.query.name,
                    "parameter_version": state.parameter_version,
                    "plan_summary": view.get("plan_summary"),
                    "estimated_rows": view.get("estimated_rows"),
                    "compile_status": status,
                    "verification": view.get("verification", {}),
                    "view_digest": view.get("view_digest"),
                    "fast_path": "incremental_plan_cache_read",
                    "full_recompile_triggered": False,
                }
            except AppError as error:
                state = None
                error_payload = error.to_dict()
                payload = {
                    "ok": False,
                    **error_payload,
                    "compile_status": "NOT_READY",
                    "fast_path": "incremental_plan_cache_read",
                    "full_recompile_triggered": False,
                }
        end_ns = time.monotonic_ns()
        elapsed_ms = (end_ns - start_ns) / 1_000_000.0
        over = elapsed_ms > self.budget_ms
        code = E_LATENCY if over else ("OK" if payload.get("ok") else payload.get("code", "E_LEDGER"))
        latency = {
            "workspace_id": self.workspace_id,
            "kind": "interactive_query",
            "query_name": payload.get("query_name") or query_name,
            "code": code,
            "within_budget": not over,
            "budget_ms": self.budget_ms,
            "elapsed_ms": elapsed_ms,
            "mono_start_ns": start_ns,
            "mono_end_ns": end_ns,
            "truth_source": "ledger:latency_records",
            "details": {
                "plan_summary": payload.get("plan_summary"),
                "compile_status": payload.get("compile_status"),
                "full_recompile_triggered": False,
            },
        }
        latency_id = self.ledger.record_latency(latency)
        payload.update(
            {
                "elapsed_ms": elapsed_ms,
                "budget_ms": self.budget_ms,
                "within_budget": not over,
                "warning_code": E_LATENCY if over else None,
                "latency_id": latency_id,
                "mono_start_ns": start_ns,
                "mono_end_ns": end_ns,
                "truth_source": f"ledger:latency_records:{latency_id}",
                "source": "ledger",
            }
        )
        if over:
            self.logger.emit(
                "interactive_latency_over_budget",
                code=E_LATENCY,
                details={
                    "elapsed_ms": elapsed_ms,
                    "budget_ms": self.budget_ms,
                    "latency_id": latency_id,
                    "truth_source": "ledger:latency_records",
                },
            )
        return payload

    def authoritative_recompute(self, query_name: str | None = None) -> dict[str, Any]:
        with self.lock:
            state = self._resolve_query(query_name)
            start_ns = time.monotonic_ns()
            full_plan = self._compile_full(state)
            end_ns = time.monotonic_ns()
            elapsed_ms = (end_ns - start_ns) / 1_000_000.0
            difference = first_plan_difference(state.plan, full_plan)
            status = "MATCH" if difference is None else E_DIVERGE
            self.ledger.record_verification(
                workspace_id=self.workspace_id,
                batch_id=state.last_batch_id,
                mode="AUTHORITATIVE",
                status=status,
                elapsed_ms=elapsed_ms,
                expected_summary=state.plan.summary,
                actual_summary=full_plan.summary,
                first_diff=difference,
                mono_start_ns=start_ns,
                mono_end_ns=end_ns,
                update_batch=False,
            )
            over = elapsed_ms > self.budget_ms
            latency_id = self.ledger.record_latency(
                {
                    "workspace_id": self.workspace_id,
                    "kind": "authoritative_recompute",
                    "query_name": state.query.name,
                    "batch_id": state.last_batch_id,
                    "code": E_LATENCY if over else ("OK" if difference is None else E_DIVERGE),
                    "within_budget": not over,
                    "budget_ms": self.budget_ms,
                    "elapsed_ms": elapsed_ms,
                    "mono_start_ns": start_ns,
                    "mono_end_ns": end_ns,
                    "truth_source": "ledger:latency_records",
                    "details": {
                        "status": status,
                        "first_diff": difference,
                        "source": "raw SQL plus complete edit trace",
                    },
                }
            )
            if difference is not None:
                view = dict(state.view)
                view["compile_status"] = E_DIVERGE
                view["verification"] = {
                    "status": E_DIVERGE,
                    "mode": "AUTHORITATIVE",
                    "first_diff": difference,
                }
                state.view = view
                error = AppError(
                    E_DIVERGE,
                    "authoritative recomputation differs from cached incremental plan",
                    edit_seq=state.edit_seq,
                    node_id=difference.get("node_id"),
                    details={"first_diff": difference, "latency_id": latency_id},
                )
                self.logger.error(error, "authoritative_diverged", edit_seq=state.edit_seq, query=state.query.name)
                raise error
            message = {
                "ok": True,
                "query_name": state.query.name,
                "status": "MATCH",
                "plan_summary": full_plan.summary,
                "estimated_rows": full_plan.estimated_rows,
                "elapsed_ms": elapsed_ms,
                "budget_ms": self.budget_ms,
                "within_budget": not over,
                "warning_code": E_LATENCY if over else None,
                "truth_source": f"ledger:latency_records:{latency_id}",
                "recomputed_from": "raw SQL plus complete edit trace",
                "edit_seq": state.edit_seq,
            }
            self.logger.emit(
                "authoritative_recompute_match",
                code=E_LATENCY if over else "OK",
                edit_seq=state.edit_seq,
                details={
                    "elapsed_ms": elapsed_ms,
                    "budget_ms": self.budget_ms,
                    "latency_id": latency_id,
                    "plan_summary": full_plan.summary,
                },
            )
            return message

    def _commit_parameter_rebuild(
        self,
        new_spec: WorkspaceSpec,
        *,
        event_type: str,
        reason: str,
    ) -> dict[str, Any]:
        start_ns = time.monotonic_ns()
        table_map = new_spec.table_map()
        new_plans: dict[str, PlanTree] = {}
        for query in new_spec.queries:
            new_plans[query.name] = compile_base(
                query.parsed,
                table_map[query.parsed.table],
                new_spec.scale_factor,
                query.name,
            )
        end_ns = time.monotonic_ns()
        elapsed_ms = (end_ns - start_ns) / 1_000_000.0
        new_pv = self._next_parameter_version()
        new_epoch = self.epoch + 1
        new_versions = dict(new_spec.versions)
        cursor = self.ledger.next_cursor(self.workspace_id)
        records: list[dict[str, Any]] = []
        latencies: list[dict[str, Any]] = []
        staged_states: dict[str, QueryState] = {}
        previous_states = dict(self.states)
        for index, query in enumerate(new_spec.queries):
            plan = new_plans[query.name]
            context = new_spec.context_digest(query.name)
            affected = sorted(plan.nodes)
            record = {
                "workspace_id": self.workspace_id,
                "query_name": query.name,
                "epoch": new_epoch,
                "cursor": cursor + index,
                "batch_kind": "BASE",
                "parameter_version": new_pv,
                "tables_version": new_versions["tables"],
                "sql_version": new_versions["sql"],
                "edits_version": new_versions["edits"],
                "scale_version": new_versions["scale"],
                "table_digest": new_spec.table_digest(),
                "sql_digest": new_spec.sql_digest(),
                "scale_digest": new_spec.scale_digest(),
                "context_digest": context,
                "edit_seq_start": 0,
                "edit_seq_end": 0,
                "plan_summary": plan.summary,
                "plan_bytes": plan.to_bytes(),
                "node_count": len(plan.nodes),
                "estimated_rows": plan.estimated_rows,
                "affected": affected,
                "propagation": [{"node": node_id, "reason": "parameter_version_rebuild"} for node_id in affected],
                "edits": [],
                "incremental_ms": elapsed_ms,
                "full_ms": elapsed_ms,
                "verify_status": "MATCH",
                "verify": {"status": "MATCH", "basis": "full base rebuild after parameter change"},
                "progress_cursor": cursor + index,
            }
            records.append(record)
            over = elapsed_ms > self.budget_ms
            latencies.append(
                {
                    "workspace_id": self.workspace_id,
                    "kind": "parameter_recompile",
                    "query_name": query.name,
                    "batch_index": index,
                    "code": E_LATENCY if over else "OK",
                    "within_budget": not over,
                    "budget_ms": self.budget_ms,
                    "elapsed_ms": elapsed_ms,
                    "mono_start_ns": start_ns,
                    "mono_end_ns": end_ns,
                    "truth_source": "ledger:latency_records",
                    "details": {"reason": reason, "parameter_version": new_pv},
                }
            )
            staged = QueryState(
                query=query,
                plan=plan,
                edits=[],
                edit_seq=0,
                epoch=new_epoch,
                parameter_version=new_pv,
                sql_version=int(new_versions["sql"]),
                edits_version=int(new_versions["edits"]),
                versions=dict(new_versions),
                view={},
                last_batch_id=None,
                last_cursor=cursor + index,
            )
            staged_states[query.name] = staged
        parameter_event = {
            "workspace_id": self.workspace_id,
            "event_type": event_type,
            "parameter_version": new_pv,
            "versions": new_versions,
            "reason": reason,
            "details": {"epoch": new_epoch, "queries": sorted(staged_states)},
            "mono_ns": end_ns,
        }
        batch_ids = self.ledger.commit_batches(
            records,
            latencies=latencies,
            parameter_event=parameter_event,
            stale_before_parameter_version=(self.workspace_id, new_pv),
        )
        diffs: dict[str, Any] = {}
        for index, (query_name, staged) in enumerate(staged_states.items()):
            old_state = previous_states.get(query_name)
            if old_state is not None:
                self.stale_views.append(
                    _stale_view(old_state.view, reason=f"whole replacement for {event_type}")
                )
                diffs[query_name] = plan_diff(old_state.plan, staged.plan)
            else:
                diffs[query_name] = plan_diff(None, staged.plan)
            staged.last_batch_id = batch_ids[index]
            staged.view = _make_view(
                staged,
                compile_status="READY",
                affected=sorted(staged.plan.nodes),
                propagation=[{"node": "all", "reason": "whole_view_replacement"}],
                diff=diffs[query_name],
                verification={"status": "MATCH", "basis": "parameter rebuild"},
            )
        self.workspace = new_spec
        self.states = staged_states
        self.versions = new_versions
        self.parameter_counter = new_pv
        self.epoch = new_epoch
        self.logger.emit(
            "parameter_rebuild_committed",
            code=E_LATENCY if elapsed_ms > self.budget_ms else "OK",
            details={
                "event_type": event_type,
                "parameter_version": new_pv,
                "epoch": new_epoch,
                "elapsed_ms": elapsed_ms,
                "diffs": diffs,
            },
        )
        return {
            "ok": True,
            "replacement": True,
            "parameter_version": new_pv,
            "epoch": new_epoch,
            "versions": new_versions,
            "diffs": diffs,
            "views": self.all_views(),
            "elapsed_ms": elapsed_ms,
        }

    def update_query_sql(self, query_name: str, sql: str) -> dict[str, Any]:
        with self.lock:
            state = self._resolve_query(query_name)
            try:
                new_spec = replace_query_sql(self._versioned_workspace(), state.query.name, sql)
            except AppError as error:
                self.logger.error(error, "sql_update_rejected", edit_seq=state.edit_seq + 1, query=state.query.name)
                raise
            return self._commit_parameter_rebuild(new_spec, event_type="sql_text", reason=f"SQL text changed for {query_name}")

    def update_tables(self, tables: Any) -> dict[str, Any]:
        with self.lock:
            state = self.workspace.queries[0].name if self.workspace.queries else None
            if isinstance(tables, str):
                try:
                    tables = json.loads(tables)
                except json.JSONDecodeError as error:
                    raise AppError(E_SPEC, f"invalid table JSON: {error.msg}", json_path=f"$@{error.pos}") from error
            try:
                new_spec = replace_tables(self._versioned_workspace(), tables)
            except AppError as error:
                self.logger.error(error, "table_update_rejected", edit_seq=self.next_edit_seq(state), query=state)
                raise
            return self._commit_parameter_rebuild(new_spec, event_type="table_declaration", reason="table declaration changed")

    def update_scale_factor(self, scale_factor: float) -> dict[str, Any]:
        with self.lock:
            try:
                new_spec = replace_scale_factor(self._versioned_workspace(), scale_factor)
            except AppError as error:
                self.logger.error(error, "scale_update_rejected", edit_seq=self.next_edit_seq())
                raise
            return self._commit_parameter_rebuild(new_spec, event_type="scale_factor", reason="scale factor changed")

    def replay_ledger(self, cursor: int | None = None) -> dict[str, Any]:
        with self.lock:
            target_cursor = self.ledger.latest_cursor(self.workspace_id) if cursor is None else int(cursor)
            table_map = self.workspace.table_map()
            reports: dict[str, Any] = {}
            for query in self.workspace.queries:
                base = compile_base(
                    query.parsed,
                    table_map[query.parsed.table],
                    self.workspace.scale_factor,
                    query.name,
                )
                records = [
                    record
                    for record in self.ledger.batches_for_query(
                        self.workspace_id, query.name, self.workspace.context_digest(query.name)
                    )
                    if int(record["cursor"]) <= target_cursor
                ]
                if not records:
                    continue
                replay_start_ns = time.monotonic_ns()
                try:
                    outcome = self._replay_records(query, base, records)
                except AppError as error:
                    batch_id = error.details.get("batch_id") if isinstance(error.details, dict) else None
                    if batch_id is not None:
                        self.ledger.record_verification(
                            workspace_id=self.workspace_id,
                            batch_id=int(batch_id),
                            mode="REPLAY",
                            status=E_REPLAY,
                            elapsed_ms=0.0,
                            expected_summary="",
                            actual_summary="",
                            first_diff={"node_id": error.node_id, "reason": error.message},
                            mono_start_ns=time.monotonic_ns(),
                            mono_end_ns=time.monotonic_ns(),
                            update_batch=False,
                        )
                    self.logger.error(error, "ledger_replay_failed", edit_seq=error.edit_seq, query=query.name)
                    raise
                replay_end_ns = time.monotonic_ns()
                self.ledger.record_verification(
                    workspace_id=self.workspace_id,
                    batch_id=outcome.last_batch_id,
                    mode="REPLAY",
                    status="MATCH",
                    elapsed_ms=(replay_end_ns - replay_start_ns) / 1_000_000.0,
                    expected_summary=outcome.plan.summary,
                    actual_summary=outcome.plan.summary,
                    first_diff=None,
                    mono_start_ns=replay_start_ns,
                    mono_end_ns=replay_end_ns,
                    update_batch=False,
                )
                reports[query.name] = {
                    "status": "MATCH",
                    "batches": len(outcome.reports),
                    "last_batch_id": outcome.last_batch_id,
                    "last_cursor": outcome.last_cursor,
                    "edit_seq": outcome.edit_seq,
                    "plan_summary": outcome.plan.summary,
                }
            self.logger.emit(
                "ledger_replay_match",
                code="OK",
                details={"cursor": target_cursor, "queries": reports},
            )
            return {"ok": True, "status": "MATCH", "cursor": target_cursor, "queries": reports}

    def _replay_outcomes(self, cursor: int) -> dict[str, ReplayOutcome]:
        table_map = self.workspace.table_map()
        outcomes: dict[str, ReplayOutcome] = {}
        for query in self.workspace.queries:
            base = compile_base(
                query.parsed,
                table_map[query.parsed.table],
                self.workspace.scale_factor,
                query.name,
            )
            records = [
                record
                for record in self.ledger.batches_for_query(
                    self.workspace_id, query.name, self.workspace.context_digest(query.name)
                )
                if int(record["cursor"]) <= int(cursor)
            ]
            if records:
                outcomes[query.name] = self._replay_records(query, base, records)
        return outcomes

    def snapshot_state(self, active_query: str | None = None) -> dict[str, Any]:
        with self.lock:
            active = self._resolve_query(active_query)
            return {
                "workspace_id": self.workspace_id,
                "workspace_name": self.workspace.name,
                "active_query": active.query.name,
                "versions": dict(self.versions),
                "epoch": self.epoch,
                "table_digest": self.workspace.table_digest(),
                "sql_digest": self.workspace.sql_digest(),
                "scale_digest": self.workspace.scale_digest(),
                "ledger_cursor": self.ledger.latest_cursor(self.workspace_id),
                "logical_plan_summary": active.plan.summary,
                "views": {
                    name: {
                        "sql": state.query.sql,
                        "parameter_version": state.parameter_version,
                        "epoch": state.epoch,
                        "edit_seq": state.edit_seq,
                        "edits_version": state.edits_version,
                        "plan_summary": state.plan.summary,
                        "view_digest": state.view.get("view_digest"),
                        "ledger_cursor": state.last_cursor,
                    }
                    for name, state in sorted(self.states.items())
                },
            }

    def restore_replayed_outcomes(self, cursor: int, outcomes: Mapping[str, ReplayOutcome]) -> dict[str, Any]:
        with self.lock:
            new_states: dict[str, QueryState] = {}
            for query_name, outcome in outcomes.items():
                old_state = self.states.get(query_name)
                if old_state is not None:
                    self.stale_views.append(_stale_view(old_state.view, reason="snapshot restore whole replacement"))
                state = QueryState(
                    query=outcome.query,
                    plan=outcome.plan,
                    edits=list(outcome.edits),
                    edit_seq=outcome.edit_seq,
                    epoch=outcome.epoch,
                    parameter_version=outcome.parameter_version,
                    sql_version=outcome.versions.get("sql", self.versions["sql"]),
                    edits_version=outcome.versions.get("edits", self.versions["edits"]),
                    versions=dict(outcome.versions),
                    view={},
                    last_batch_id=outcome.last_batch_id,
                    last_cursor=outcome.last_cursor,
                )
                state.view = _make_view(
                    state,
                    verification={"status": "MATCH", "basis": "snapshot ledger replay"},
                    diff=plan_diff(old_state.plan if old_state else None, state.plan),
                )
                new_states[query_name] = state
            if new_states:
                self.states = new_states
            self.ledger_cursor = int(cursor)
            return self.all_views()

    def force_replace_current_views(self) -> dict[str, dict[str, Any]]:
        """Replace all in-memory view objects without merging old node state."""

        with self.lock:
            for name, state in list(self.states.items()):
                self.stale_views.append(_stale_view(state.view, reason="authoritative whole-view replacement"))
                state.view = _make_view(
                    state,
                    compile_status=state.view.get("compile_status", "READY")
                    if state.view.get("compile_status") != "STALE"
                    else "READY",
                    verification=state.view.get("verification", {}),
                )
            return self.all_views()

    def create_snapshot(self, active_query: str | None = None, path: str | Path | None = None) -> dict[str, Any]:
        return create_snapshot(self, active_query, path)

    def restore_snapshot(self, path: str | Path) -> dict[str, Any]:
        with self.lock:
            document = read_snapshot(path)
            requested_active = document.get("active_query")
            active_name = requested_active if requested_active in self.states else self.workspace.queries[0].name
            current = self.snapshot_state(active_name)
            versions = current["versions"]
            current_header = {
                **current,
                "tables_version": int(versions["tables"]),
                "sql_version": int(versions["sql"]),
                "edits_version": int(versions["edits"]),
                "scale_version": int(versions["scale"]),
                "parameter_version": int(current["views"][active_name]["parameter_version"]),
            }
            current_edit_seq = max(
                (int(view.get("edit_seq", 0)) for view in current_header["views"].values()),
                default=0,
            )
            snapshot_edit_seq = max(
                (
                    int(view.get("edit_seq", 0))
                    for view in document.get("views", {}).values()
                    if isinstance(view, dict)
                ),
                default=0,
            )
            mismatch_edit_seq = max(current_edit_seq + 1, snapshot_edit_seq)
            differences = snapshot_differences(document, current_header)
            if not differences:
                snapshot_views = document.get("views", {})
                for query_name, current_view in current_header["views"].items():
                    snapshot_view = snapshot_views.get(query_name)
                    if not isinstance(snapshot_view, dict):
                        differences.append(
                            {
                                "field": f"views.{query_name}",
                                "snapshot": None,
                                "current": current_view.get("plan_summary"),
                            }
                        )
            if differences:
                views = self.force_replace_current_views()
                error = AppError(
                    E_SNAPSHOT,
                    "snapshot versions or digests do not match the current session",
                    edit_seq=mismatch_edit_seq,
                    details={
                        "differences": differences,
                        "replacement_mode": "whole_current_ledger_view",
                        "snapshot_ignored": True,
                        "replacement_view_digests": {
                            name: {
                                "plan_summary": view.get("plan_summary"),
                                "view_digest": view.get("view_digest"),
                            }
                            for name, view in views.items()
                        },
                    },
                )
                self.logger.error(error, "snapshot_rejected", edit_seq=mismatch_edit_seq)
                raise error

            target_cursor = int(document["ledger_cursor"])
            replay_report = self.replay_ledger(target_cursor)
            outcomes = self._replay_outcomes(target_cursor)
            snapshot_views = document.get("views", {})
            view_differences: list[dict[str, Any]] = []
            for query_name, outcome in outcomes.items():
                snapshot_view = snapshot_views.get(query_name, {})
                if snapshot_view.get("plan_summary") != outcome.plan.summary:
                    view_differences.append(
                        {
                            "field": f"views.{query_name}.plan_summary",
                            "snapshot": snapshot_view.get("plan_summary"),
                            "current": outcome.plan.summary,
                        }
                    )
            if view_differences:
                self.force_replace_current_views()
                error = AppError(
                    E_SNAPSHOT,
                    "snapshot plan summary does not match ledger replay",
                    edit_seq=mismatch_edit_seq,
                    details={
                        "differences": view_differences,
                        "replacement_mode": "whole_current_ledger_view",
                        "snapshot_ignored": True,
                    },
                )
                self.logger.error(error, "snapshot_rejected", edit_seq=mismatch_edit_seq)
                raise error
            views = self.restore_replayed_outcomes(target_cursor, outcomes)
            self.versions = {
                "tables": int(document["tables_version"]),
                "sql": int(document["sql_version"]),
                "edits": int(document["edits_version"]),
                "scale": int(document["scale_version"]),
            }
            self.epoch = int(document.get("epoch", self.epoch))
            self.parameter_counter = max(
                self.parameter_counter,
                max((view.get("parameter_version", 1) for view in views.values()), default=1),
            )
            result = {
                "ok": True,
                "restored": True,
                "replacement": True,
                "path": str(path),
                "active_query": document.get("active_query"),
                "ledger_cursor": target_cursor,
                "content_digest": document.get("content_digest"),
                "views": views,
                "replay": replay_report,
            }
            self.logger.emit(
                "snapshot_restored",
                code="OK",
                edit_seq=max((view.get("edit_seq", 0) for view in views.values()), default=0),
                details={"path": str(path), "cursor": target_cursor, "replay": replay_report},
            )
            return result
