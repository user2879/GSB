from __future__ import annotations

from typing import Any

from .engine import (
    build_partitions,
    build_window_tasks,
    feature_cache_key,
    feature_stage,
    make_ledger_book,
    schedule_windows,
    score_cache_key,
    score_stage,
    summarize,
)
from .model import Config, Issue, PreferenceRecord
from .util import fingerprint
from .ledger import downgrade_g2_to_g1, serialize_g1_book


def new_cache() -> dict[str, dict[str, dict[str, Any]]]:
    return {"features": {}, "scores": {}, "windows": {}, "partitions": {}}


def run_pipeline(
    records: list[PreferenceRecord],
    weights: dict[str, int],
    config: Config,
    issues: list[Issue] | None = None,
    cache: dict[str, Any] | None = None,
    budget: int | None = None,
    session_quota: int | None = None,
    workers: int = 1,
    pending_task_ids: list[str] | None = None,
    initial_session_points: dict[str, int] | None = None,
) -> dict[str, Any]:
    issues = list(issues or [])
    cache = cache if cache is not None else new_cache()
    cache.setdefault("features", {})
    cache.setdefault("scores", {})
    cache.setdefault("windows", {})
    cache.setdefault("partitions", {})
    cache_events: dict[str, list[dict[str, str]]] = {
        "feature_reuse": [],
        "feature_compute": [],
        "score_reuse": [],
        "score_compute": [],
        "partition_reuse": [],
        "partition_compute": [],
    }

    records = sorted(records, key=lambda r: (r.session_id, r.turn))
    features: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for record in records:
        fkey = feature_cache_key(record, config)
        if fkey in cache["features"]:
            feature = cache["features"][fkey]
            cache_events["feature_reuse"].append({"record_id": f"{record.session_id}#{record.turn:06d}", "cache_key": fkey})
        else:
            feature = feature_stage(record, config)
            cache["features"][fkey] = feature
            cache_events["feature_compute"].append({"record_id": f"{record.session_id}#{record.turn:06d}", "cache_key": fkey})
        features.append(feature)

        skey = score_cache_key(record, config, fkey, weights)
        if skey in cache["scores"]:
            score = cache["scores"][skey]
            cache_events["score_reuse"].append({"record_id": f"{record.session_id}#{record.turn:06d}", "cache_key": skey})
        else:
            score = score_stage(record, config, feature, weights)
            cache["scores"][skey] = score
            cache_events["score_compute"].append({"record_id": f"{record.session_id}#{record.turn:06d}", "cache_key": skey})
        scores.append(score)

    tasks = build_window_tasks(scores, config)
    cached_windows_by_scope: dict[str, dict[str, Any]] = {}
    for task in tasks:
        cached = cache["windows"].get(task["cache_key"])
        if cached is not None:
            cached_windows_by_scope[task["scope_id"]] = cached
    windows, scheduler_state, scheduler_issues = schedule_windows(
        tasks,
        config,
        budget=budget,
        session_quota=session_quota,
        workers=workers,
        cached_windows=cached_windows_by_scope,
        pending_task_ids=pending_task_ids,
        initial_session_points=initial_session_points,
    )
    issues.extend(scheduler_issues)
    reused_window_ids = {hit["scope_id"] for hit in scheduler_state.get("reuse_hits", [])}
    cache_events["window_reuse"] = list(scheduler_state.get("reuse_hits", []))
    cache_events["window_compute"] = [
        {"scope_id": window["scope_id"], "cache_key": window["cache_key"], "stage": "sliding_window"}
        for window in windows
        if window["scope_id"] not in reused_window_ids
    ]
    for window in windows:
        cache["windows"][window["cache_key"]] = window

    max_turns: dict[str, int] = {}
    for record in records:
        max_turns[record.session_id] = max(max_turns.get(record.session_id, 0), record.turn)
    cached_partitions = cache["partitions"]
    partitions, partition_reuse = build_partitions(windows, max_turns, config, cached_partitions)
    cache_events["partition_reuse"].extend(partition_reuse)
    current_partition_keys = {p["cache_key"] for p in partitions}
    for partition in partitions:
        if partition["cache_key"] not in cache["partitions"]:
            cache_events["partition_compute"].append({"scope_id": partition["scope_id"], "cache_key": partition["cache_key"], "stage": "partition"})
        cache["partitions"][partition["cache_key"]] = partition
    # Do not report stale cache entries as current partition reuse.
    cache_events["partition_reuse"] = [event for event in cache_events["partition_reuse"] if event["cache_key"] in current_partition_keys]

    summary = summarize(records, scores, windows, partitions, weights)
    book = make_ledger_book(windows, partitions, summary, config)
    exchange_artifacts: dict[str, Any] = {"exchange_version": config.exchange_version}
    if config.exchange_version == "g1":
        g1_book, g1_sidecar = downgrade_g2_to_g1(book)
        exchange_artifacts["ledger_g1_text"] = serialize_g1_book(g1_book, config.ledger_generation)
        exchange_artifacts["ledger_g1_sidecar"] = g1_sidecar
    else:
        exchange_artifacts["ledger_g2_text"] = book.to_g2_json(config.ledger_generation)

    bucket_summaries = [
        {
            "record_id": item["record_id"],
            "session_id": item["session_id"],
            "turn": item["turn"],
            "bucket_digest": item["bucket_digest"],
            "nonzero": item["nonzero"],
            "cache_key": item["cache_key"],
        }
        for item in sorted(features, key=lambda x: (x["session_id"], x["turn"]))
    ]
    cache_keys = {
        "config": fingerprint(config.to_dict()),
        "weights": fingerprint({k: weights[k] for k in sorted(weights)}),
        "feature": [{"id": item["record_id"], "key": item["cache_key"]} for item in bucket_summaries],
        "score": [{"id": f"{s['session_id']}#{s['turn']:06d}", "key": s["cache_key"]} for s in sorted(scores, key=lambda x: (x["session_id"], x["turn"]))],
        "window": [{"id": w["scope_id"], "key": w["cache_key"]} for w in windows],
        "partition": [{"id": p["scope_id"], "key": p["cache_key"]} for p in partitions],
        "ledger_book": fingerprint(book.to_g2_dict("L2")),
    }
    report = {
        "run_fingerprint": fingerprint(
            {
                "config": config.to_dict(),
                "records": [r.to_dict() for r in records],
                "weights": {k: weights[k] for k in sorted(weights)},
                "workers": workers,
            }
        ),
        "config": config.to_dict(),
        "issues": [issue.to_dict() for issue in issues],
        "failure_list": [issue.to_dict() for issue in issues],
        "bucket_summaries": bucket_summaries,
        "record_scores": sorted(scores, key=lambda x: (x["session_id"], x["turn"])),
        "windows": windows,
        "partitions": partitions,
        **summary,
        "cache_keys": cache_keys,
        "cache_events": cache_events,
        "scheduler": scheduler_state,
        "ledger_book": book.to_g2_dict(config.ledger_generation),
        "exchange_artifacts": exchange_artifacts,
    }
    return {
        "report": report,
        "book": book,
        "cache": cache,
        "issues": issues,
        "tasks": tasks,
    }


def semantic_comparison_report(report: dict[str, Any]) -> dict[str, Any]:
    """Project to fields that must match between incremental and full runs."""
    keys = [
        "config",
        "issues",
        "failure_list",
        "bucket_summaries",
        "record_scores",
        "windows",
        "partitions",
        "annotator_summary",
        "session_summary",
        "batch_summary",
        "cache_keys",
    ]
    projected = {key: report[key] for key in keys}
    scheduler = report["scheduler"]
    projected["scheduler"] = {
        key: value
        for key, value in scheduler.items()
        if key in {"aborted_sessions", "pending_task_ids", "quota_pending_task_ids"}
    }
    return projected
