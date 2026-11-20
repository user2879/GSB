from __future__ import annotations

from typing import Any


def build_view_model(snapshot: dict[str, Any]) -> dict[str, Any]:
    summary = dict(snapshot.get("summary", {}))
    requests = list(snapshot.get("requests", []))
    total = int(summary.get("requests_total", len(requests)))
    completed = sum(1 for request in requests if request.get("status") == "completed")
    progress_percent = 100.0 if total == 0 else min(100.0, completed * 100.0 / total)

    instances: list[dict[str, Any]] = []
    for service in snapshot.get("services", []):
        for instance in service.get("instances", []):
            row = dict(instance)
            row["service"] = service.get("service")
            row["quota"] = service.get("concurrency_quota")
            instances.append(row)
    instances.sort(key=lambda row: (str(row["service"]), int(row["port"])))

    budget = snapshot.get("budget", {})
    global_budget = dict(
        budget.get(
            "global",
            {
                "limit": snapshot.get("limits", {}).get("global_concurrency", 0),
                "active": 0,
                "peak": 0,
            },
        )
    )
    budget_services = sorted(
        [dict(row) for row in budget.get("services", [])],
        key=lambda row: str(row.get("service")),
    )
    for row in budget_services:
        limit = int(row.get("limit", 0))
        peak = int(row.get("peak", 0))
        row["utilization_percent"] = 0.0 if limit == 0 else peak * 100.0 / limit
    global_limit = int(global_budget.get("limit", 0))
    global_peak = int(global_budget.get("peak", 0))
    global_budget["utilization_percent"] = (
        0.0 if global_limit == 0 else global_peak * 100.0 / global_limit
    )

    rules = snapshot.get("rules", {})
    routes = []
    for route in rules.get("routes", []):
        if isinstance(route, dict):
            row = dict(route)
            row["rule_version"] = snapshot.get("rule_version")
            routes.append(row)
    routes.sort(key=lambda row: (str(row.get("prefix", "")), str(row.get("service", ""))))

    circuit_events = []
    for event in snapshot.get("circuit_events", []):
        row = dict(event)
        row["rule_version"] = snapshot.get("rule_version")
        circuit_events.append(row)
    circuit_events.sort(
        key=lambda row: (
            str(row.get("service", "")),
            int(row.get("port", 0)),
            int(row.get("event_id", 0)),
        )
    )

    return {
        "schema_version": 1,
        "meta": {
            "snapshot_id": snapshot.get("snapshot_id"),
            "round_id": snapshot.get("round_id"),
            "rule_version": snapshot.get("rule_version"),
            "observed_rule_version": snapshot.get("observed_rule_version"),
            "rules_changed_during_round": snapshot.get(
                "rules_changed_during_round", False
            ),
            "created_at_utc": snapshot.get("created_at_utc"),
            "status": summary.get("status"),
        },
        "progress": {
            "completed": completed,
            "total": total,
            "percent": round(progress_percent, 2),
        },
        "summary": summary,
        "services": sorted(
            [dict(service) for service in snapshot.get("services", [])],
            key=lambda row: str(row.get("service")),
        ),
        "instances": instances,
        "budget": {
            "global": global_budget,
            "services": budget_services,
        },
        "circuit_events": circuit_events,
        "routes": routes,
        "render_order": {
            "instances": ["service", "port"],
            "routes": ["prefix", "service"],
            "circuit_events": ["service", "port", "event_id"],
        },
    }


def render_view_text(view: dict[str, Any]) -> str:
    meta = view["meta"]
    summary = view["summary"]
    progress = view["progress"]
    lines = [
        f"Round {meta['round_id']}  [{meta['status']}]  rule={meta['rule_version']}",
        f"Progress: {progress['completed']}/{progress['total']} ({progress['percent']:.1f}%)",
        f"Conclusion: {summary.get('conclusion', '')}",
        "",
        "SERVICE / INSTANCE     WEIGHT  TARGET  SELECT  FAIL  CIRCUIT",
    ]
    for row in view["instances"]:
        lines.append(
            f"{row['service']:<10} {row['instance']:<12} {row['weight']:>6} "
            f"{row['targeted_count']:>6} {row['selected_count']:>6} "
            f"{row['failure_count']:>5}  {row['circuit_state']}"
        )
    global_budget = view["budget"]["global"]
    lines.extend(
        [
            "",
            f"Global budget: peak={global_budget.get('peak', 0)}/"
            f"{global_budget.get('limit', 0)} active={global_budget.get('active', 0)}",
        ]
    )
    for row in view["budget"]["services"]:
        lines.append(
            f"  {row['service']}: peak={row.get('peak', 0)}/{row.get('limit', 0)} "
            f"active={row.get('active', 0)}"
        )
    return "\n".join(lines)

