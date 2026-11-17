"""Deterministic view grouping strategy selection."""

from .geometry import match_views


def estimate_match_density(views, features_by_view):
    """Mean mutual-nearest match count over adjacent view pairs."""
    counts = []
    ordered = sorted(views, key=lambda view: view["view_id"])
    for index in range(len(ordered) - 1):
        view_a, view_b = ordered[index], ordered[index + 1]
        matches = match_views(
            view_a, features_by_view[view_a["view_id"]],
            view_b, features_by_view[view_b["view_id"]],
        )
        counts.append(len(matches))
    return (sum(counts) / len(counts)) if counts else 0.0


def build_groups(view_ids, group_size, views_per_batch):
    """Split sorted view ids into consecutive groups of group_size."""
    ordered = sorted(view_ids)
    groups, group_index = [], 0
    for start in range(0, len(ordered), group_size):
        member_ids = ordered[start : start + group_size]
        groups.append({
            "group_index": group_index,
            "view_ids": member_ids,
            "slot_demand": len(member_ids),
            "views_per_batch": views_per_batch,
            "stable_key": (group_index,),
        })
        group_index += 1
    return groups


def _candidate_score(candidate, metrics):
    group_size = candidate["group_size"]
    views_per_batch = candidate["views_per_batch"]
    group_count = max(1, (metrics["view_count"] + group_size - 1) // group_size)
    slot_feasible = group_size <= metrics["slot_budget"]
    point_headroom = metrics["point_budget"] / max(1, group_count * max(1, metrics["avg_match_count"]))
    score = 0.0
    score += 2.0 if slot_feasible else -50.0
    score += min(2.0, point_headroom)
    score -= 0.15 * abs(
        group_size - min(metrics["view_count"], max(2, metrics["view_count"] // 4))
    )
    score -= 0.05 * group_count
    score += 1.0 if views_per_batch >= group_size else -10.0
    score += 0.25 * min(metrics["avg_match_count"], 8.0) / group_size
    return round(score, 9), group_count, slot_feasible


def select_grouping(config, views, features_by_view):
    """Pick a grouping candidate deterministically; concurrency never matters."""
    view_count = len(views)
    feature_counts = [len(features_by_view[view["view_id"]]) for view in views]
    avg_feature_density = (sum(feature_counts) / view_count) if view_count else 0.0
    avg_match_count = estimate_match_density(views, features_by_view)
    metrics = {
        "view_count": view_count,
        "avg_feature_density": round(avg_feature_density, 6),
        "avg_match_count": round(avg_match_count, 6),
        "slot_budget": config["slot_budget"],
        "point_budget": config["point_budget"],
        "thread_count": config["thread_count"],
    }
    scored = []
    for candidate in config["grouping_candidates"]:
        score, group_count, slot_feasible = _candidate_score(candidate, metrics)
        scored.append({
            "name": candidate["name"],
            "group_size": candidate["group_size"],
            "views_per_batch": candidate["views_per_batch"],
            "group_count": group_count,
            "slot_feasible": slot_feasible,
            "score": score,
        })
    feasible = [entry for entry in scored if entry["slot_feasible"]]
    pool = feasible if feasible else scored
    selected = sorted(pool, key=lambda entry: (-entry["score"], entry["name"]))[0]
    reason = (
        "highest deterministic score among slot-feasible candidates"
        if feasible
        else "no candidate fits the slot budget; highest score selected and will suspend"
    )
    groups = build_groups(
        [view["view_id"] for view in views],
        selected["group_size"],
        selected["views_per_batch"],
    )
    decision = {
        "metrics": {k: v for k, v in metrics.items() if k != "thread_count"},
        "thread_count_seen_but_ignored": metrics["thread_count"],
        "candidate_scores": sorted(scored, key=lambda entry: entry["name"]),
        "selected": selected["name"],
        "reason": reason,
        "groups": groups,
    }
    return decision
