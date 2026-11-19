"""Deterministic forensic report and case-graph construction."""

from __future__ import annotations

from typing import Any, Iterable

from .canonical import canonical_json, sha256_text
from .errors import E_EVENT_CONFLICT, business_error


REPORT_SCHEMA_VERSION = "audit-report.v1"
GRAPH_SCHEMA_VERSION = "audit-case-graph.v1"


def _error_sort_key(error: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(error.get("source", "")),
        str(error.get("position", "")),
        str(error.get("trace_id", "")),
        str(error.get("event_id", "")),
        str(error.get("code", "")),
        canonical_json(error),
    )


def conflict_errors(conflicts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for conflict in conflicts:
        errors.append(
            business_error(
                E_EVENT_CONFLICT,
                "the same event_id has materially different normalized content",
                event_id=conflict["event_id"],
                details={
                    "conflict_id": conflict["conflict_id"],
                    "fields": conflict["fields"],
                    "source_formats": conflict["source_formats"],
                },
            )
        )
    return errors


def build_graph(
    event_states: Iterable[dict[str, Any]],
    trace_chains: Iterable[dict[str, Any]],
    cases: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    event_ids: set[str] = set()

    for state in sorted(
        event_states, key=lambda item: item.get("event", {}).get("event_id", "")
    ):
        event = state["event"]
        event_id = event["event_id"]
        event_ids.add(event_id)
        nodes.append(
            {
                "node_id": f"event:{event_id}",
                "node_type": "event",
                "event_id": event_id,
                "trace_id": event["trace_id"],
                "ts": event["ts"],
                "seq": event["seq"],
                "status": state.get("status", "accepted"),
            }
        )

    for case in sorted(cases, key=lambda item: item["case_id"]):
        nodes.append(
            {
                "node_id": f"case:{case['case_id']}",
                "node_type": "case",
                "case_id": case["case_id"],
                "rule_id": case["rule_id"],
                "rules_version": case["rules_version"],
                "member_count": len(case["member_event_ids"]),
            }
        )
        for event_id in case["member_event_ids"]:
            if event_id not in event_ids:
                continue
            basis = {
                "rule_id": case["rule_id"],
                "rules_version": case["rules_version"],
                "group": case["group"],
                "member_event_id": event_id,
            }
            edge_id = "EDGE-" + sha256_text(
                canonical_json(
                    {
                        "from": f"case:{case['case_id']}",
                        "to": f"event:{event_id}",
                        "kind": "case_member",
                    }
                )
            )[:24].upper()
            edges.append(
                {
                    "edge_id": edge_id,
                    "from": f"case:{case['case_id']}",
                    "to": f"event:{event_id}",
                    "kind": "case_member",
                    "basis": basis,
                }
            )

    for trace in sorted(trace_chains, key=lambda item: item["trace_id"]):
        for node in trace["nodes"]:
            parent_id = node.get("parent_event_id")
            if not parent_id or parent_id not in event_ids:
                continue
            edge_id = "EDGE-" + sha256_text(
                canonical_json(
                    {
                        "from": f"event:{parent_id}",
                        "to": f"event:{node['event_id']}",
                        "kind": "causal_parent",
                    }
                )
            )[:24].upper()
            edges.append(
                {
                    "edge_id": edge_id,
                    "from": f"event:{parent_id}",
                    "to": f"event:{node['event_id']}",
                    "kind": "causal_parent",
                    "basis": {
                        "field": "parent_event_id",
                        "parent_event_id": parent_id,
                        "trace_id": trace["trace_id"],
                    },
                }
            )

    nodes.sort(key=lambda node: node["node_id"])
    edges.sort(key=lambda edge: edge["edge_id"])
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
    }


def _event_status_records(event_states: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for state in sorted(
        event_states, key=lambda item: item.get("event", {}).get("event_id", "")
    ):
        event = state["event"]
        count = len(state.get("provenance", []))
        status = state.get("status", "accepted")
        if count > 1 and status == "accepted":
            status = "duplicate"
        records.append(
            {
                "status": status,
                "event": event,
                "attributes": state.get("attributes", {}),
                "occurrence_count": count,
                "primary_sync_marker": state.get("primary_sync_marker"),
                "provenance": state.get("provenance", []),
            }
        )
    return records


def build_report(
    state: dict[str, Any],
    *,
    checkpoint_checksum: str,
    checkpoint_path: str,
    extra_errors: Iterable[dict[str, Any]] = (),
    notices: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    event_states = state.get("events", [])
    conflicts = state.get("conflicts", [])
    diagnostics = state.get("diagnostics", [])
    parse_errors = state.get("parse_errors", [])
    errors = list(parse_errors) + list(diagnostics) + conflict_errors(conflicts)
    errors.extend(extra_errors)
    errors.sort(key=_error_sort_key)
    notice_list = sorted(list(notices), key=_error_sort_key)

    cases = state.get("cases", [])
    files = state.get("file_results", [])
    failed_files = [item for item in files if item.get("status") != "ok"]
    status = "ok" if not errors else "partial"
    if failed_files and not files:
        status = "failed"
    exit_code = 0 if status == "ok" else 1
    event_records = _event_status_records(event_states)
    error_codes = sorted({error["code"] for error in errors})

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "rules": state.get("rules", {}),
        "checkpoint": {
            "path": checkpoint_path,
            "checksum": checkpoint_checksum,
            "generation": state.get("generation", 0),
            "watermark": state.get("watermark", {}),
        },
        "rebuild": state.get("rebuild", {}),
        "conclusions": {
            "files_total": len(files),
            "files_ok": len(files) - len(failed_files),
            "files_partial_or_failed": len(failed_files),
            "events_normalized": len(event_records),
            "events_conflicted": len(conflicts),
            "traces": len(state.get("trace_chains", [])),
            "cases": len(cases),
            "business_error_count": len(errors),
            "business_error_codes": error_codes,
        },
        "files": files,
        "events": event_records,
        "conflicts": conflicts,
        "traces": state.get("trace_chains", []),
        "cases": cases,
        "errors": errors,
        "notices": notice_list,
        "graph": state.get("graph", {}),
    }
