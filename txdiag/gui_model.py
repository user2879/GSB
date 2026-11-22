"""Pure, deterministic view model shared by Tkinter and headless tests."""

from __future__ import annotations

from typing import Any

from .engine import ExecutionResult
from .model import Spec


def build_gui_model(spec: Spec, result: ExecutionResult) -> dict[str, Any]:
    transactions = []
    access_edges: list[dict[str, str]] = []
    for tx in sorted(spec.transactions, key=lambda item: item.priority):
        rows = sorted({op.row for op in tx.operations}, key=spec.row_index.get)
        transactions.append({
            "txid": tx.txid,
            "start_seq": tx.start_seq,
            "isolation": tx.isolation,
            "operations": [
                {"seq": op.seq, "kind": op.kind, "row": op.row}
                for op in tx.operations
            ],
            "outcome": result.result["transactions"].get(tx.txid, "unknown"),
        })
        for row in rows:
            access_edges.append({"from": tx.txid, "to": row, "kind": "access"})

    rows = [
        {"row": row, "index": idx}
        for idx, row in enumerate(spec.row_keys)
    ]
    dependency_edges = [
        {"from": edge.source, "to": edge.target, "kind": edge.kind}
        for edge in sorted(spec.dependency_edges, key=lambda e: (e.source, e.target, e.kind))
    ]

    last_round = result.rounds[-1] if result.rounds else {
        "locks": [], "waiters": [], "wait_graph": {"nodes": [], "edges": [], "cycles": []}
    }
    lock_rows = []
    for lock in sorted(last_round.get("locks", []),
                       key=lambda item: (spec.row_index.get(item["row"], 10**9),
                                         item["txid"], item["modes"])):
        lock_rows.append({
            "row": lock["row"],
            "txid": lock["txid"],
            "modes": ",".join(lock.get("modes", [])),
            "state": "held",
        })
    for wait in sorted(last_round.get("waiters", []),
                       key=lambda item: (spec.row_index.get(item["row"], 10**9),
                                         item["txid"], item["mode"])):
        lock_rows.append({
            "row": wait["row"],
            "txid": wait["txid"],
            "modes": wait["mode"],
            "state": "waiting",
        })

    wait_graph_rounds = [
        {
            "round": payload["round"],
            "nodes": list(payload.get("wait_graph", {}).get("nodes", [])),
            "edges": [list(edge) for edge in payload.get("wait_graph", {}).get("edges", [])],
            "cycles": [list(cycle) for cycle in payload.get("wait_graph", {}).get("cycles", [])],
        }
        for payload in result.rounds
    ]
    active_cycles = last_round.get("wait_graph", {}).get("cycles", [])
    deadlock_rows: list[dict[str, Any]] = []
    for record in sorted(result.deadlocks, key=lambda item: (item["round"], item["victim"])):
        deadlock_rows.append({
            "round": record["round"],
            "cycle": list(record["cycle"]),
            "victim": record["victim"],
            "basis": record["basis"],
            "edges": [list(edge) for edge in record.get("wait_graph_edges", [])],
        })

    quota_rows = []
    for payload in result.rounds:
        quota = payload.get("quota", {})
        iso = quota.get("isolation", {})
        quota_rows.append({
            "round": payload["round"],
            "logical_ms": payload["logical_ms"],
            "running": list(payload.get("running", [])),
            "global_used": quota.get("global", {}).get("used", 0),
            "global_limit": quota.get("global", {}).get("limit", 0),
            "isolation": {
                name: f"{value.get('used', 0)}/{value.get('limit', 0)}"
                for name, value in sorted(iso.items())
            },
            "queued": [item["txid"] for item in payload.get("admission_queue", [])],
        })

    event_rows = []
    for payload in result.rounds:
        for event in payload.get("events", []):
            event_rows.append({
                "round": payload["round"],
                "event_seq": event.get("event_seq", 0),
                "at_ms": event.get("at_ms", 0),
                "event": event.get("event", "unknown"),
                "txid": event.get("txid"),
                "row": event.get("row"),
                "detail": _event_detail(event),
            })
    event_rows.sort(key=lambda item: (item["round"], item["event_seq"]))
    for payload in result.rounds:
        for event in payload.get("events", []):
            if event.get("event") not in (
                    "lock_acquired", "lock_release", "lock_conflict", "wait_start", "wait_end"):
                continue
            mode = event.get("mode")
            if mode is None and event.get("modes"):
                mode = ",".join(event["modes"])
            lock_rows.append({
                "row": event.get("row") or "",
                "txid": event.get("txid") or "",
                "modes": mode or "",
                "state": f'{event.get("event")}@r{payload["round"]}',
            })

    summary = result.result
    stable_summary = {
        "format_version": summary.get("format_version"),
        "status": summary.get("status"),
        "spec_sha256": summary.get("spec_sha256"),
        "path": summary.get("path"),
        "round_count": summary.get("round_count"),
        "transactions": summary.get("transactions"),
        "final_versions": summary.get("final_versions"),
        "final_locks": summary.get("final_locks"),
        "deadlocks": [
            {key: item[key] for key in ("round", "cycle", "victim", "basis")}
            for item in summary.get("deadlocks", [])
        ],
        "timeouts": [
            {key: item[key] for key in ("round", "txid", "row", "mode", "wait_ms", "code")}
            for item in summary.get("timeouts", [])
        ],
        "wait_budget": {
            "limit_ms": summary.get("wait_budget", {}).get("limit_ms"),
            "canonical_used_ms": summary.get("wait_budget", {}).get("canonical_used_ms"),
        },
        "canonical_sha256": summary.get("canonical_sha256"),
    }

    return {
        "transactions": transactions,
        "row_nodes": rows,
        "dependency_edges": dependency_edges,
        "access_edges": access_edges,
        "locks": lock_rows,
        "wait_graph": {
            "last_round": wait_graph_rounds[-1] if wait_graph_rounds else {},
            "rounds": wait_graph_rounds,
            "active_cycles": [list(cycle) for cycle in active_cycles],
        },
        "deadlocks": deadlock_rows,
        "quota_timeline": quota_rows,
        "isolation": sorted(result.isolation_conclusions, key=lambda item: (
            item["start_seq"], item["txid"])),
        "events": event_rows,
        "summary": stable_summary,
    }


def _event_detail(event: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("mode", "modes", "result", "reason", "version", "value", "source",
                "cycle", "selected_victim", "basis", "limit_ms", "wait_ms"):
        if key in event:
            value = event[key]
            if isinstance(value, (dict, list)):
                value = str(value)
            parts.append(f"{key}={value}")
    return "; ".join(parts)
