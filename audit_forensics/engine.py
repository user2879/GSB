"""Transactional import, recovery, queries, export, and checkpoint validation."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import (
    CORE_EVENT_FIELDS,
    canonical_json,
    event_fingerprint,
    sha256_text,
)
from .causal import analyze_sequences, reconstruct_traces
from .errors import (
    E_CHECKPOINT_INVALID,
    E_CASE_NOT_FOUND,
    E_LOCK,
    E_NO_STATE,
    E_REPORT_WRITE,
    E_RULES_CONFIG_CHANGED,
    E_RULES_VERSION_CHANGED,
    E_SNAPSHOT_STALE,
    E_STATE_SCHEMA_MISMATCH,
    LockConflictError,
    business_error,
)
from .ingest import ParsedFile, parse_files
from .locks import WorkspaceLock
from .report import build_graph, build_report
from .rules import RuleConfig, evaluate_rules
from .storage import (
    WorkspacePaths,
    atomic_write_json,
    cleanup_temporary_files,
    load_case_snapshot,
    load_checkpoint,
    write_case_snapshot,
    write_checkpoint,
)


EMPTY_RULES = RuleConfig.from_dict({"rules_version": "0", "rules": []})


@dataclass
class CommandResult:
    payload: dict[str, Any]
    exit_code: int


def _error_key(error: dict[str, Any]) -> str:
    return canonical_json(error)


def _dedupe_errors(errors: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for error in errors:
        key = _error_key(error)
        if key not in seen:
            seen.add(key)
            result.append(error)
    result.sort(
        key=lambda error: (
            str(error.get("source", "")),
            str(error.get("position", "")),
            str(error.get("trace_id", "")),
            str(error.get("event_id", "")),
            str(error.get("code", "")),
            canonical_json(error),
        )
    )
    return result


def _normalized_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy only structurally sound state fields; rebuild all derived fields."""

    events = payload.get("events")
    if not isinstance(events, list):
        events = []
    sound_events: list[dict[str, Any]] = []
    for state in events:
        if (
            isinstance(state, dict)
            and isinstance(state.get("event"), dict)
            and isinstance(state["event"].get("event_id"), str)
            and all(field in state["event"] for field in CORE_EVENT_FIELDS)
        ):
            state = copy.deepcopy(state)
            state.setdefault("status", "accepted")
            state.setdefault("attributes", {"severity": None, "source": None})
            state.setdefault("provenance", [])
            state.setdefault("primary_sync_marker", None)
            sound_events.append(state)

    conflicts = payload.get("conflicts")
    if not isinstance(conflicts, list):
        conflicts = []
    file_results = payload.get("file_results")
    if not isinstance(file_results, list):
        file_results = []
    parse_errors = payload.get("parse_errors")
    if not isinstance(parse_errors, list):
        parse_errors = []
    generation = payload.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        generation = 0
    return {
        "generation": generation,
        "events": sound_events,
        "conflicts": copy.deepcopy(conflicts),
        "file_results": copy.deepcopy(file_results),
        "parse_errors": _dedupe_errors(parse_errors),
        "rules": copy.deepcopy(payload.get("rules"))
        if isinstance(payload.get("rules"), dict)
        else {},
        "trace_chains": copy.deepcopy(payload.get("trace_chains"))
        if isinstance(payload.get("trace_chains"), list)
        else [],
        "diagnostics": copy.deepcopy(payload.get("diagnostics"))
        if isinstance(payload.get("diagnostics"), list)
        else [],
        "cases": copy.deepcopy(payload.get("cases"))
        if isinstance(payload.get("cases"), list)
        else [],
        "graph": copy.deepcopy(payload.get("graph"))
        if isinstance(payload.get("graph"), dict)
        else {},
    }


def _severity_attributes(provenance: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rank = {
        "critical": 5,
        "high": 4,
        "medium": 3,
        "low": 2,
        "info": 1,
        "informational": 1,
    }
    severities = sorted(
        {
            item.get("severity")
            for item in provenance
            if isinstance(item.get("severity"), str) and item.get("severity")
        }
    )
    severity: str | None = None
    if severities:
        severity = max(severities, key=lambda item: (rank.get(item.lower(), 0), item))
    sources = sorted(
        {
            item.get("source_name")
            for item in provenance
            if isinstance(item.get("source_name"), str) and item.get("source_name")
        }
    )
    return {
        "severity": severity,
        "source": sources[0] if sources else None,
    }


def _provenance_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item.get("source", "")),
        str(item.get("position", "")),
        str(item.get("sync_marker", "")),
    )


def _merge_occurrences(
    existing_events: list[dict[str, Any]],
    parsed_files: list[ParsedFile],
    *,
    arrival_base: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    events_by_id: dict[str, dict[str, Any]] = {
        state["event"]["event_id"]: state for state in existing_events
    }
    conflicts_by_id: dict[str, dict[str, Any]] = {}
    for state in existing_events:
        for conflict in state.get("_conflicts", []):
            conflicts_by_id[conflict["conflict_id"]] = conflict
    # Conflicts are stored separately in checkpoints and copied here by caller.
    for state in existing_events:
        state.pop("_conflicts", None)

    new_occurrences: list[dict[str, Any]] = []
    next_arrival = arrival_base
    for parsed_file in parsed_files:
        for occurrence in parsed_file.occurrences:
            provenance = occurrence.as_provenance()
            provenance["source_order"] = next_arrival
            provenance["position_order"] = occurrence.position_order
            new_occurrences.append(provenance)
            next_arrival += 1
    new_occurrences.sort(
        key=lambda item: (
            item["source_order"],
            item["position_order"],
            item["sync_marker"],
        )
    )

    for occurrence in new_occurrences:
        core = occurrence["event_core"]
        event_id = core["event_id"]
        state = events_by_id.get(event_id)
        if state is None:
            state = {
                "status": "accepted",
                "event": copy.deepcopy(core),
                "attributes": {
                    "severity": occurrence.get("severity"),
                    "source": occurrence.get("source_name"),
                },
                "primary_sync_marker": occurrence["sync_marker"],
                "provenance": [occurrence],
            }
            events_by_id[event_id] = state
            continue

        existing_markers = {
            (item.get("sync_marker"), item.get("source"))
            for item in state.get("provenance", [])
        }
        if (occurrence["sync_marker"], occurrence.get("source")) not in existing_markers:
            state.setdefault("provenance", []).append(occurrence)

        winner_core = state["event"]
        if event_fingerprint(winner_core) != event_fingerprint(core):
            differing = [
                field
                for field in CORE_EVENT_FIELDS
                if field != "event_id" and winner_core.get(field) != core.get(field)
            ]
            variant_fingerprint = event_fingerprint(core)
            conflict_id = "CONFLICT-" + sha256_text(
                canonical_json(
                    {
                        "event_id": event_id,
                        "winner": event_fingerprint(winner_core),
                        "incoming": variant_fingerprint,
                    }
                )
            )[:24].upper()
            primary = _primary_provenance(state)
            conflict = {
                "conflict_id": conflict_id,
                "event_id": event_id,
                "fields": differing,
                "winner": _occurrence_identity(primary, winner_core),
                "incoming": _occurrence_identity(occurrence, core),
                "source_formats": sorted(
                    {
                        primary.get("source_format", "unknown"),
                        occurrence.get("source_format", "unknown"),
                    }
                ),
            }
            conflicts_by_id[conflict_id] = conflict
            state["status"] = "conflict"
        elif state.get("status") != "conflict":
            state["status"] = (
                "duplicate" if len(state.get("provenance", [])) > 1 else "accepted"
            )

        state["provenance"].sort(key=_provenance_sort_key)
        state["attributes"] = _severity_attributes(state["provenance"])

    for state in events_by_id.values():
        state["provenance"].sort(key=_provenance_sort_key)
        state["attributes"] = _severity_attributes(state["provenance"])

    merged = sorted(events_by_id.values(), key=lambda item: item["event"]["event_id"])
    conflicts = sorted(conflicts_by_id.values(), key=lambda item: item["conflict_id"])
    return merged, conflicts, next_arrival


def _primary_provenance(state: dict[str, Any]) -> dict[str, Any]:
    marker = state.get("primary_sync_marker")
    for occurrence in state.get("provenance", []):
        if occurrence.get("sync_marker") == marker:
            return occurrence
    provenance = state.get("provenance", [])
    return provenance[0] if provenance else {}


def _occurrence_identity(
    occurrence: dict[str, Any], core: dict[str, Any]
) -> dict[str, Any]:
    return {
        "source_format": occurrence.get("source_format"),
        "source": occurrence.get("source"),
        "position": occurrence.get("position"),
        "sync_marker": occurrence.get("sync_marker"),
        "event": copy.deepcopy(core),
    }


def _file_result(parsed_file: ParsedFile, source_id: str) -> dict[str, Any]:
    if not parsed_file.errors:
        status = "ok"
    elif parsed_file.occurrences:
        status = "partial"
    else:
        status = "failed"
    return {
        "source_id": source_id,
        "source": parsed_file.path,
        "sha256": parsed_file.sha256,
        "size_bytes": parsed_file.size_bytes,
        "detected_format": parsed_file.detected_format,
        "status": status,
        "occurrences": len(parsed_file.occurrences),
        "error_count": len(parsed_file.errors),
        "error_codes": sorted({error["code"] for error in parsed_file.errors}),
    }


def _watermark(
    events: list[dict[str, Any]], generation: int, occurrence_count: int
) -> dict[str, Any]:
    max_ts = max(
        (state["event"]["ts"] for state in events),
        default=None,
    )
    max_seq_by_trace: dict[str, int] = {}
    max_arrival = -1
    for state in events:
        event = state["event"]
        previous = max_seq_by_trace.get(event["trace_id"])
        if previous is None or event["seq"] > previous:
            max_seq_by_trace[event["trace_id"]] = event["seq"]
        for occurrence in state.get("provenance", []):
            value = occurrence.get("source_order")
            if isinstance(value, int) and value > max_arrival:
                max_arrival = value
    return {
        "generation": generation,
        "unique_events": len(events),
        "processed_occurrences": occurrence_count,
        "arrival_watermark": max_arrival,
        "max_ts": max_ts,
        "max_seq_by_trace": {
            trace_id: max_seq_by_trace[trace_id]
            for trace_id in sorted(max_seq_by_trace)
        },
    }


class AuditEngine:
    def __init__(
        self,
        workspace: str | Path,
        *,
        rules: RuleConfig | None = None,
    ) -> None:
        self.paths = WorkspacePaths(Path(workspace).resolve())
        self.rules = rules

    def import_streams(
        self,
        inputs: Iterable[str | Path],
        *,
        workers: int = 1,
        report_path: str | Path | None = None,
        graph_path: str | Path | None = None,
    ) -> CommandResult:
        input_paths = [str(Path(item)) for item in inputs]
        parsed_files = parse_files(input_paths, workers=workers)
        self.paths.ensure()
        try:
            with WorkspaceLock(self.paths.lock_file):
                cleanup_temporary_files(self.paths.state_dir)
                return self._import_locked(
                    parsed_files,
                    report_path=Path(report_path) if report_path else self.paths.report,
                    graph_path=Path(graph_path) if graph_path else self.paths.graph,
                )
        except LockConflictError as exc:
            error = business_error(E_LOCK, str(exc), source=str(self.paths.lock_file))
            return CommandResult(
                payload={
                    "report_schema_version": "audit-report.v1",
                    "status": "error",
                    "exit_code": 1,
                    "errors": [error],
                },
                exit_code=1,
            )

    def _import_locked(
        self,
        parsed_files: list[ParsedFile],
        *,
        report_path: Path,
        graph_path: Path,
    ) -> CommandResult:
        checkpoint = load_checkpoint(self.paths)
        extra_errors: list[dict[str, Any]] = list(checkpoint.errors)
        notices: list[dict[str, Any]] = list(checkpoint.notices)
        has_state = checkpoint.has_state
        base = (
            _normalized_state_payload(checkpoint.payload)
            if has_state
            else _normalized_state_payload({})
        )
        if not has_state and not parsed_files:
            error = business_error(
                E_NO_STATE,
                "no checkpoint exists and no input streams were supplied",
            )
            payload = {
                "report_schema_version": "audit-report.v1",
                "status": "error",
                "exit_code": 1,
                "errors": [error, *extra_errors],
            }
            return CommandResult(payload=payload, exit_code=1)

        stored_rules_raw = None
        if has_state:
            rules_state = checkpoint.payload.get("rules")
            if isinstance(rules_state, dict):
                stored_rules_raw = rules_state.get("config")
        stored_rules: RuleConfig | None = None
        if isinstance(stored_rules_raw, dict):
            try:
                stored_rules = RuleConfig.from_dict(stored_rules_raw)
            except Exception:
                stored_rules = None
                notices.append(
                    business_error(
                        E_CHECKPOINT_INVALID,
                        "stored rules configuration is invalid; using supplied/default rules",
                        source=checkpoint.path,
                    )
                )

        current_rules = self.rules or stored_rules or EMPTY_RULES
        rebuild_reason: str | None = None
        rebuild_details: dict[str, Any] = {}
        if checkpoint.force_rebuild:
            rebuild_reason = "checkpoint_schema_changed"
            rebuild_details = {
                "found": checkpoint.schema_version,
                "expected": "audit-checkpoint.v1",
            }
        elif stored_rules is not None and stored_rules.version != current_rules.version:
            rebuild_reason = "rules_version_changed"
            rebuild_details = {
                "previous": stored_rules.version,
                "current": current_rules.version,
            }
            notices.append(
                business_error(
                    E_RULES_VERSION_CHANGED,
                    "rules version changed; all cases are recomputed from all events",
                    details=rebuild_details,
                )
            )
        elif stored_rules is not None and stored_rules.digest != current_rules.digest:
            rebuild_reason = "rules_config_changed"
            rebuild_details = {
                "previous_digest": stored_rules.digest,
                "current_digest": current_rules.digest,
            }
            notices.append(
                business_error(
                    E_RULES_CONFIG_CHANGED,
                    "rules content changed under the same version; all cases are recomputed",
                    details=rebuild_details,
                )
            )

        existing_events = base["events"]
        conflict_by_id = {
            conflict["conflict_id"]: conflict for conflict in base["conflicts"]
        }
        for state in existing_events:
            state["_conflicts"] = [
                conflict
                for conflict in base["conflicts"]
                if conflict.get("event_id") == state["event"]["event_id"]
            ]
        arrival_base = 0
        for state in existing_events:
            for occurrence in state.get("provenance", []):
                value = occurrence.get("source_order")
                if isinstance(value, int) and value >= arrival_base:
                    arrival_base = value + 1

        merged_events, new_conflicts, next_arrival = _merge_occurrences(
            existing_events,
            parsed_files,
            arrival_base=arrival_base,
        )
        for conflict in new_conflicts:
            conflict_by_id[conflict["conflict_id"]] = conflict
        conflicts = sorted(conflict_by_id.values(), key=lambda item: item["conflict_id"])

        file_results_by_id = {
            item.get("source_id", canonical_json(item)): item
            for item in base["file_results"]
            if isinstance(item, dict)
        }
        parse_errors = list(base["parse_errors"])
        for index, parsed_file in enumerate(parsed_files):
            source_id = sha256_text(
                canonical_json(
                    {
                        "source": parsed_file.path,
                        "sha256": parsed_file.sha256,
                    }
                )
            )
            file_results_by_id[source_id] = _file_result(parsed_file, source_id)
            parse_errors.extend(parsed_file.errors)
        file_results = sorted(
            file_results_by_id.values(),
            key=lambda item: (str(item.get("source", "")), item.get("source_id", "")),
        )
        parse_errors = _dedupe_errors(parse_errors)

        trace_chains, causal_errors = reconstruct_traces(merged_events)
        provenance = [
            occurrence
            for state in merged_events
            for occurrence in state.get("provenance", [])
        ]
        sequence_errors = analyze_sequences(provenance, merged_events)
        diagnostics = _dedupe_errors(causal_errors + sequence_errors)
        conflicted_event_ids = {
            state["event"]["event_id"]
            for state in merged_events
            if state.get("status") == "conflict"
        }
        cases = evaluate_rules(
            merged_events,
            current_rules,
            excluded_event_ids=conflicted_event_ids,
        )
        graph = build_graph(merged_events, trace_chains, cases)
        generation = base["generation"] + 1
        occurrence_count = sum(
            len(state.get("provenance", [])) for state in merged_events
        )
        state_payload = {
            "generation": generation,
            "watermark": _watermark(merged_events, generation, occurrence_count),
            "rules": current_rules.as_state(),
            "events": merged_events,
            "conflicts": conflicts,
            "file_results": file_results,
            "parse_errors": parse_errors,
            "trace_chains": trace_chains,
            "diagnostics": diagnostics,
            "cases": cases,
            "graph": graph,
            "rebuild": {
                "required": rebuild_reason is not None,
                "reason": rebuild_reason or "none",
                "details": rebuild_details,
            },
        }
        envelope = write_checkpoint(self.paths, state_payload)
        write_case_snapshot(
            self.paths,
            {
                "generation": generation,
                "checkpoint_checksum": envelope["checksum"],
                "rules": {
                    "version": current_rules.version,
                    "digest": current_rules.digest,
                },
                "cases": cases,
                "graph": graph,
            },
        )
        report = build_report(
            state_payload,
            checkpoint_checksum=envelope["checksum"],
            checkpoint_path=str(self.paths.checkpoint),
            extra_errors=extra_errors,
            notices=notices,
        )
        try:
            atomic_write_json(report_path, report, pause_key="report")
            atomic_write_json(graph_path, graph, pause_key="graph")
        except OSError as exc:
            error = business_error(
                E_REPORT_WRITE,
                f"cannot write report or graph: {exc}",
                source=str(report_path),
            )
            report["errors"].append(error)
            report["status"] = "partial"
            report["exit_code"] = 1
        return CommandResult(payload=report, exit_code=report["exit_code"])

    def _read_checkpoint(self) -> tuple[dict[str, Any], str, str, list[dict[str, Any]], list[dict[str, Any]]]:
        checkpoint = load_checkpoint(self.paths)
        if not checkpoint.has_state:
            error = business_error(
                E_NO_STATE,
                "workspace does not contain a readable checkpoint",
                details={"errors": checkpoint.errors},
            )
            raise RuntimeError(canonical_json(error))
        state = _normalized_state_payload(checkpoint.payload)
        notices = list(checkpoint.notices)
        if checkpoint.force_rebuild:
            notices.append(
                business_error(
                    E_CHECKPOINT_INVALID,
                    "checkpoint schema is not current; derived fields were rebuilt in memory",
                    source=checkpoint.path,
                )
            )
        return state, checkpoint.checksum, checkpoint.path, checkpoint.errors, notices

    def _read_state_with_snapshot(self) -> dict[str, Any]:
        state, checksum, path, errors, notices = self._read_checkpoint()
        snapshot = load_case_snapshot(
            self.paths,
            checkpoint_checksum=checksum,
            checkpoint_generation=state["generation"],
        )
        notices.extend(snapshot.notices)
        if snapshot.valid:
            state["cases"] = snapshot.payload.get("cases", [])
            state["graph"] = snapshot.payload.get("graph", {})
        else:
            notices.extend(snapshot.errors)
            rules_raw = None
            check_payload = load_checkpoint(self.paths).payload
            rules_state = check_payload.get("rules")
            if isinstance(rules_state, dict):
                rules_raw = rules_state.get("config")
            try:
                rules = (
                    RuleConfig.from_dict(rules_raw)
                    if isinstance(rules_raw, dict)
                    else EMPTY_RULES
                )
            except Exception:
                rules = EMPTY_RULES
            conflicted_event_ids = {
                item["event"]["event_id"]
                for item in state["events"]
                if item.get("status") == "conflict"
            }
            state["cases"] = evaluate_rules(
                state["events"], rules, excluded_event_ids=conflicted_event_ids
            )
            state["graph"] = build_graph(
                state["events"], state.get("trace_chains", []), state["cases"]
            )
        if not state.get("trace_chains"):
            trace_chains, causal_errors = reconstruct_traces(state["events"])
            state["trace_chains"] = trace_chains
            state["diagnostics"] = _dedupe_errors(
                list(state.get("diagnostics", [])) + causal_errors
            )
            if not snapshot.valid:
                state["graph"] = build_graph(
                    state["events"], trace_chains, state["cases"]
                )
        state["checkpoint_checksum"] = checksum
        state["checkpoint_path"] = path
        state["_read_errors"] = errors
        state["_notices"] = notices
        return state

    def query_case(self, case_id: str | None = None) -> CommandResult:
        try:
            state = self._read_state_with_snapshot()
        except RuntimeError as exc:
            error = json.loads(str(exc))
            return CommandResult(
                payload={
                    "report_schema_version": "audit-report.v1",
                    "status": "error",
                    "exit_code": 1,
                    "errors": [error],
                },
                exit_code=1,
            )
        cases = state["cases"]
        if case_id is not None:
            selected = [case for case in cases if case["case_id"] == case_id]
            if not selected:
                error = business_error(
                    E_CASE_NOT_FOUND,
                    f"case not found: {case_id}",
                    details={"case_id": case_id},
                )
                return CommandResult(
                    payload={
                        "report_schema_version": "audit-report.v1",
                        "status": "error",
                        "exit_code": 1,
                        "errors": [error],
                    },
                    exit_code=1,
                )
            payload = {
                "status": "ok",
                "exit_code": 0,
                "rules_version": state.get("rules", {}).get("version"),
                "case": selected[0],
                "notices": state["_notices"],
            }
            return CommandResult(payload=payload, exit_code=0)
        payload = {
            "status": "ok",
            "exit_code": 0,
            "rules_version": state.get("rules", {}).get("version"),
            "case_count": len(cases),
            "cases": cases,
            "notices": state["_notices"],
        }
        return CommandResult(payload=payload, exit_code=0)

    def export_report(
        self,
        *,
        output: str | Path | None = None,
        graph_output: str | Path | None = None,
    ) -> CommandResult:
        try:
            state = self._read_state_with_snapshot()
        except RuntimeError as exc:
            error = json.loads(str(exc))
            return CommandResult(
                payload={
                    "report_schema_version": "audit-report.v1",
                    "status": "error",
                    "exit_code": 1,
                    "errors": [error],
                },
                exit_code=1,
            )
        report = build_report(
            state,
            checkpoint_checksum=state["checkpoint_checksum"],
            checkpoint_path=state["checkpoint_path"],
            extra_errors=state["_read_errors"],
            notices=state["_notices"],
        )
        target = Path(output) if output else self.paths.report
        graph_target = Path(graph_output) if graph_output else self.paths.graph
        try:
            atomic_write_json(target, report, pause_key="report")
            atomic_write_json(graph_target, state["graph"], pause_key="graph")
        except OSError as exc:
            error = business_error(
                E_REPORT_WRITE,
                f"cannot write report or graph: {exc}",
                source=str(target),
            )
            report["errors"].append(error)
            report["status"] = "partial"
            report["exit_code"] = 1
        return CommandResult(payload=report, exit_code=report["exit_code"])

    def verify_checkpoint(self) -> CommandResult:
        checkpoint = load_checkpoint(self.paths)
        errors: list[dict[str, Any]] = list(checkpoint.errors)
        notices: list[dict[str, Any]] = list(checkpoint.notices)
        if not checkpoint.has_state:
            errors.append(
                business_error(
                    "E_CHECKPOINT_MISSING",
                    "no readable checkpoint exists",
                    source=str(self.paths.checkpoint),
                )
            )
            payload = {
                "status": "invalid",
                "exit_code": 1,
                "valid": False,
                "errors": _dedupe_errors(errors),
                "notices": notices,
            }
            return CommandResult(payload=payload, exit_code=1)

        if checkpoint.force_rebuild:
            errors.append(
                business_error(
                    E_STATE_SCHEMA_MISMATCH,
                    "checkpoint structure version is not current",
                    source=checkpoint.path,
                    details={
                        "found": checkpoint.schema_version,
                        "expected": "audit-checkpoint.v1",
                    },
                )
            )

        state = _normalized_state_payload(checkpoint.payload)
        if not state["events"] and checkpoint.payload.get("events"):
            errors.append(
                business_error(
                    E_CHECKPOINT_INVALID,
                    "checkpoint event state is structurally invalid",
                    source=checkpoint.path,
                )
            )
        event_ids = [item["event"]["event_id"] for item in state["events"]]
        if len(event_ids) != len(set(event_ids)):
            errors.append(
                business_error(
                    E_CHECKPOINT_INVALID,
                    "checkpoint contains duplicate normalized event IDs",
                    source=checkpoint.path,
                )
            )
        snapshot = load_case_snapshot(
            self.paths,
            checkpoint_checksum=checkpoint.checksum,
            checkpoint_generation=state["generation"],
        )
        errors.extend(snapshot.errors)
        notices.extend(snapshot.notices)

        stored_rules_raw = None
        rules_state = checkpoint.payload.get("rules")
        if isinstance(rules_state, dict):
            stored_rules_raw = rules_state.get("config")
        try:
            rules = (
                RuleConfig.from_dict(stored_rules_raw)
                if isinstance(stored_rules_raw, dict)
                else EMPTY_RULES
            )
        except Exception as exc:
            errors.append(
                business_error(
                    E_CHECKPOINT_INVALID,
                    f"stored rules configuration is invalid: {exc}",
                    source=checkpoint.path,
                )
            )
            rules = EMPTY_RULES
        else:
            expected_version = checkpoint.payload.get("rules", {}).get("version")
            if str(expected_version) != rules.version:
                errors.append(
                    business_error(
                        E_CHECKPOINT_INVALID,
                        "checkpoint rules version does not match embedded rules",
                        source=checkpoint.path,
                    )
                )
            expected_digest = checkpoint.payload.get("rules", {}).get("digest")
            if expected_digest != rules.digest:
                errors.append(
                    business_error(
                        E_CHECKPOINT_INVALID,
                        "checkpoint rules digest does not match embedded rules",
                        source=checkpoint.path,
                    )
                )

        conflicted = {
            item["event"]["event_id"]
            for item in state["events"]
            if item.get("status") == "conflict"
        }
        recomputed_cases = evaluate_rules(
            state["events"], rules, excluded_event_ids=conflicted
        )
        checkpoint_cases = checkpoint.payload.get("cases")
        if not isinstance(checkpoint_cases, list) or canonical_json(
            checkpoint_cases
        ) != canonical_json(recomputed_cases):
            errors.append(
                business_error(
                    E_CHECKPOINT_INVALID,
                    "checkpoint cases do not match a full recomputation",
                    source=checkpoint.path,
                )
            )
        if snapshot.valid and canonical_json(snapshot.payload.get("cases")) != canonical_json(
            recomputed_cases
        ):
            errors.append(
                business_error(
                    E_SNAPSHOT_STALE,
                    "case snapshot does not match checkpoint event state",
                    source=str(self.paths.case_snapshot),
                )
            )

        trace_chains = checkpoint.payload.get("trace_chains")
        if not isinstance(trace_chains, list):
            trace_chains, _ = reconstruct_traces(state["events"])
        recomputed_graph = build_graph(
            state["events"], trace_chains, recomputed_cases
        )
        if canonical_json(checkpoint.payload.get("graph")) != canonical_json(
            recomputed_graph
        ):
            errors.append(
                business_error(
                    E_CHECKPOINT_INVALID,
                    "checkpoint graph does not match a full recomputation",
                    source=checkpoint.path,
                )
            )
        if snapshot.valid and canonical_json(snapshot.payload.get("graph")) != canonical_json(
            recomputed_graph
        ):
            errors.append(
                business_error(
                    E_SNAPSHOT_STALE,
                    "case snapshot graph does not match checkpoint event state",
                    source=str(self.paths.case_snapshot),
                )
            )

        occurrence_count = sum(
            len(item.get("provenance", [])) for item in state["events"]
        )
        expected_watermark = _watermark(
            state["events"], state["generation"], occurrence_count
        )
        if canonical_json(checkpoint.payload.get("watermark")) != canonical_json(
            expected_watermark
        ):
            errors.append(
                business_error(
                    E_CHECKPOINT_INVALID,
                    "checkpoint watermark does not match persisted event state",
                    source=checkpoint.path,
                )
            )

        errors = _dedupe_errors(errors)
        valid = not errors
        payload = {
            "status": "ok" if valid else "invalid",
            "exit_code": 0 if valid else 1,
            "valid": valid,
            "checkpoint": {
                "path": checkpoint.path,
                "schema_version": checkpoint.schema_version,
                "checksum": checkpoint.checksum,
                "generation": state["generation"],
                "recovered_from_previous": checkpoint.recovered_from_previous,
                "watermark": checkpoint.payload.get("watermark", {}),
            },
            "snapshot": {
                "valid": snapshot.valid,
                "checksum": snapshot.checksum,
                "cases": len(snapshot.payload.get("cases", []))
                if snapshot.valid
                else 0,
            },
            "rules": {
                "version": rules.version,
                "digest": rules.digest,
            },
            "events": len(state["events"]),
            "cases": len(recomputed_cases),
            "errors": errors,
            "notices": notices,
        }
        return CommandResult(payload=payload, exit_code=0 if valid else 1)
