from __future__ import annotations

from typing import Any

from .errors import E_POINT_BUDGET_PENDING, E_QUERY_QUOTA_EXCEEDED


def new_scheduler_state(batch_points: int, query_quota_points: int) -> dict[str, Any]:
    return {
        "todo": [],
        "completed": [],
        "quota_exceeded": [],
        "batch_points": batch_points,
        "query_quota_points": query_quota_points,
        "batch_remaining": batch_points,
        "batch_consumed": 0,
        "total_consumed_points": 0,
        "last_batch": 0,
    }


def begin_batch(state: dict[str, Any], batch_points: int | None = None) -> dict[str, Any]:
    points = int(batch_points if batch_points is not None else state["batch_points"])
    state["batch_points"] = points
    state["batch_remaining"] = points
    state["batch_consumed"] = 0
    state["last_batch"] = int(state.get("last_batch", 0)) + 1
    return state


def plan_schedule(
    state: dict[str, Any],
    query_ids: list[str],
    cost_by_query: dict[str, int],
    *,
    query_quota_points: int | None = None,
) -> dict[str, Any]:
    quota = int(
        query_quota_points
        if query_quota_points is not None
        else state["query_quota_points"]
    )
    stable_ids = sorted(query_ids)
    known = set(stable_ids)
    aborted = set(state.get("quota_exceeded", []))
    todo = [qid for qid in state.get("todo", []) if qid in known and qid not in aborted]
    for qid in stable_ids:
        if qid not in todo and qid not in state.get("completed", []) and qid not in aborted:
            todo.append(qid)

    ready: list[dict[str, Any]] = []
    quota_failures: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    remaining = int(state["batch_remaining"])
    consumed = 0
    index = 0
    while index < len(todo):
        qid = todo[index]
        cost = int(cost_by_query.get(qid, 0))
        if cost > quota:
            quota_failures.append(
                {
                    "code": E_QUERY_QUOTA_EXCEEDED,
                    "query_id": qid,
                    "cost_points": cost,
                    "query_quota_points": quota,
                    "message": (
                        f"query point cost {cost} exceeds quota {quota}; query aborted"
                    ),
                }
            )
            state.setdefault("quota_exceeded", []).append(qid)
            index += 1
            continue
        if cost > remaining:
            for pending_id in todo[index:]:
                pending_cost = int(cost_by_query.get(pending_id, 0))
                pending.append(
                    {
                        "code": E_POINT_BUDGET_PENDING,
                        "query_id": pending_id,
                        "cost_points": pending_cost,
                        "batch_remaining_points": remaining,
                        "message": "insufficient batch budget; retained in priority todo queue",
                    }
                )
            break
        ready.append({"query_id": qid, "cost_points": cost})
        remaining -= cost
        consumed += cost
        state["total_consumed_points"] = int(state.get("total_consumed_points", 0)) + cost
        index += 1

    state["todo"] = [item["query_id"] for item in pending]
    state["batch_remaining"] = remaining
    state["batch_consumed"] = int(state.get("batch_consumed", 0)) + consumed
    completed = list(state.get("completed", []))
    completed.extend(item["query_id"] for item in ready)
    state["completed"] = sorted(set(completed))
    return {
        "ready": ready,
        "quota_failures": quota_failures,
        "pending": pending,
        "todo": list(state["todo"]),
        "batch_remaining": remaining,
        "batch_consumed_this_plan": consumed,
    }
