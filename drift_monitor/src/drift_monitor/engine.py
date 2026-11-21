"""Orchestration engine: slicing, concurrent metrics, fixed-point decisions.

The engine derives a cache key for every stage artifact from upstream content
fingerprints plus the parameter fingerprint.  Incremental runs recompute only
keys missing from the cache; the recompute/reuse sets are identical to what a
clean full rebuild would report because they are computed by key comparison,
not by edit history.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .cache import StageCache, derived_key, fingerprint
from .config import MonitorConfig
from .datagen import CUR, REF, Dataset, GeneratedColumn
from .flip import TierEvaluation, diagnose_flips
from .metrics import (
    HIST_BINS,
    WindowStats,
    build_stats,
    histogram_on_grid,
    metric_score,
)
from .snapshot import Manifest, SnapshotStore
from .strategy import DecisionRecord, select_strategy
from .windows import WindowSlice, aligned_window_plan


@dataclass
class RunContext:
    config: MonitorConfig
    tier_name: str
    reference: Dataset
    current: Dataset
    columns: tuple  # tuple[ColumnSpec]
    decision: DecisionRecord
    ref_windows: tuple[WindowSlice, ...]
    cur_windows: tuple[WindowSlice, ...]
    cache: StageCache
    store: SnapshotStore
    config_hash: str
    data_fingerprint: str
    approx_quantile: bool
    effective_metrics: dict[str, str]
    budget_log: list[str] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    recomputed: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    expected_invalidated: list[str] = field(default_factory=list)
    expected_reusable: list[str] = field(default_factory=list)
    required_keys: list[str] = field(default_factory=list)


def sort_key(column: str, window_index: int) -> str:
    return f"{column}::w{window_index:06d}"


def _data_fingerprint(config: MonitorConfig, tier_name: str,
                      reference: Dataset, current: Dataset) -> str:
    payload = {
        "tier": tier_name,
        "reference": reference.data_fingerprint_inputs,
        "current": current.data_fingerprint_inputs,
        "missing_markers_ref": [
            {"column": c.name, "mask": _mask_digest(c.missing_mask)}
            for c in reference.columns
        ],
        "missing_markers_cur": [
            {"column": c.name, "mask": _mask_digest(c.missing_mask)}
            for c in current.columns
        ],
    }
    return fingerprint(payload, prefix="data:")


def _mask_digest(mask) -> str:
    return fingerprint("".join("1" if flag else "0" for flag in mask))


def _window_fingerprint(slices: tuple[WindowSlice, ...]) -> str:
    return fingerprint([(s.index, s.start, s.end, s.tag) for s in slices],
                       prefix="windows:")


def _effective_metric(ctx: RunContext, column_name: str) -> str:
    override = ctx.config.metric_overrides.get(column_name)
    if override:
        return override
    strategy_override = ctx.decision.metric_overrides.get(column_name)
    if strategy_override:
        return strategy_override
    spec = next(c for c in ctx.columns if c.name == column_name)
    return spec.metric


def _column_lookup(dataset: Dataset, name: str) -> GeneratedColumn:
    return next(c for c in dataset.columns if c.name == name)


def build_task_plan(ctx: RunContext) -> list[dict]:
    """Canonical task order: column name lexicographic, then window index."""
    names = sorted(c.name for c in ctx.columns)
    plan = []
    for name in names:
        for window_index in range(len(ctx.ref_windows)):
            plan.append({
                "sort_key": sort_key(name, window_index),
                "column": name,
                "window_index": window_index,
            })
    return plan


def _metric_cache_key(ctx: RunContext, column: str, window_index: int,
                      metric: str, approx: bool) -> str:
    ref_slice = ctx.ref_windows[window_index]
    cur_slice = ctx.cur_windows[window_index]
    ref_col = _column_lookup(ctx.reference, column)
    cur_col = _column_lookup(ctx.current, column)
    ref_obs = ref_col.values[ref_slice.start:ref_slice.end]
    cur_obs = cur_col.values[cur_slice.start:cur_slice.end]
    content = fingerprint({
        "ref": [None if v is None else str(v) for v in ref_obs],
        "cur": [None if v is None else str(v) for v in cur_obs],
        "ref_window": (ref_slice.start, ref_slice.end, ref_slice.tag),
        "cur_window": (cur_slice.start, cur_slice.end, cur_slice.tag),
    })
    params = {"metric": metric, "approx_quantile": approx,
              "kind": ref_col.kind, "window_index": window_index}
    return derived_key(content, params, "metric")


def _compute_one(ctx: RunContext, task: dict) -> dict:
    column = task["column"]
    window_index = task["window_index"]
    metric = _effective_metric(ctx, column)
    approx = ctx.approx_quantile
    key = _metric_cache_key(ctx, column, window_index, metric, approx)
    cached = ctx.cache.get("metric", key)
    if cached is not None:
        return {"task": task, "key": key, "reused": True, "artifact": cached}
    ref_slice = ctx.ref_windows[window_index]
    cur_slice = ctx.cur_windows[window_index]
    ref_col = _column_lookup(ctx.reference, column)
    cur_col = _column_lookup(ctx.current, column)
    ref_stats = build_stats(ref_col, window_index, REF,
                            ref_slice.start, ref_slice.end)
    cur_stats = build_stats(cur_col, window_index, CUR,
                            cur_slice.start, cur_slice.end)
    if ref_col.kind == "numeric" and metric == "hist_density":
        ref_present = [float(v) for v in ref_col.values[ref_slice.start:ref_slice.end]
                       if v is not None]
        cur_present = [float(v) for v in cur_col.values[cur_slice.start:cur_slice.end]
                       if v is not None]
        from .metrics import _histogram
        _, edges = _histogram(ref_present)
        ref_stats = WindowStats(
            **{**ref_stats.__dict__,
               "histogram": histogram_on_grid(ref_present, edges),
               "bin_edges": edges})
        cur_stats = WindowStats(
            **{**cur_stats.__dict__,
               "histogram": histogram_on_grid(cur_present, edges),
               "bin_edges": edges})
    if f"{column}#w{window_index}" in ctx.config.fail_keys:
        raise RuntimeError(f"injected failure for {column} window {window_index}")
    raw_score = metric_score(metric, ref_stats, cur_stats, approx)
    artifact = {
        "column": column,
        "kind": ref_col.kind,
        "window_index": window_index,
        "metric": metric,
        "sort_key": task["sort_key"],
        "ref_slice": [ref_slice.start, ref_slice.end, ref_slice.tag],
        "cur_slice": [cur_slice.start, cur_slice.end, cur_slice.tag],
        "raw_score": raw_score,
        "ref_n_observed": ref_stats.n_observed,
        "cur_n_observed": cur_stats.n_observed,
        "ref_n_missing": ref_stats.n_missing,
        "cur_n_missing": cur_stats.n_missing,
        "quantile_approx": list(ref_stats.quantiles_approx),
        "quantile_exact": list(ref_stats.quantiles_exact),
        "content_fingerprint": key.split(":", 1)[1],
    }
    ctx.cache.put("metric", key, artifact)
    return {"task": task, "key": key, "reused": False, "artifact": artifact}


def run_metrics(ctx: RunContext) -> list[dict]:
    plan = build_task_plan(ctx)
    expected_keys = [
        _metric_cache_key(
            ctx, task["column"], task["window_index"],
            _effective_metric(ctx, task["column"]), ctx.approx_quantile)
        for task in plan
    ]
    ctx.required_keys.extend(expected_keys)
    for key in expected_keys:
        if ctx.cache.get("metric", key) is None:
            ctx.expected_invalidated.append(key)
        else:
            ctx.expected_reusable.append(key)
    max_workers = min(ctx.config.concurrency, max(1, len(plan)))
    results: list[dict | None] = [None] * len(plan)
    if max_workers == 1:
        for idx, task in enumerate(plan):
            results[idx] = _safe_compute(ctx, task)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(_safe_compute, ctx, task): idx
                for idx, task in enumerate(plan)
            }
            for future in future_map:
                idx = future_map[future]
                results[idx] = future.result()
    # Deterministic post-processing in canonical order regardless of scheduling.
    ordered = []
    for item in results:
        if item is None:
            continue
        ordered.append(item)
        if item.get("failure"):
            ctx.failures.append(item["failure"])
            ctx.budget_log.append(
                f"deduct:1:{item['failure']['sort_key']}")
        else:
            # Canonical per-task budget deduction, independent of cache state
            # or thread scheduling: every column-window costs one unit.
            ctx.budget_log.append(
                f"deduct:1:{item['task']['sort_key']}")
            if item["reused"]:
                ctx.reused.append(item["key"])
            else:
                ctx.recomputed.append(item["key"])
    return ordered


def _safe_compute(ctx: RunContext, task: dict) -> dict:
    try:
        return _compute_one(ctx, task)
    except Exception as exc:  # noqa: BLE001 - business failure list
        return {
            "task": task,
            "key": sort_key(task["column"], task["window_index"]),
            "reused": False,
            "failure": {
                "code": "E_METRIC_FAILED",
                "column": task["column"],
                "window_index": task["window_index"],
                "sort_key": task["sort_key"],
                "message": str(exc),
            },
        }


def run_decisions(ctx: RunContext, metric_artifacts: list[dict]) -> dict:
    threshold = ctx.config.significance_threshold
    tiers = ctx.config.tiers
    active_index = ctx.config.tier_names.index(ctx.config.active_tier)
    active_tier = tiers[active_index]
    per_column: dict[str, list[dict]] = {}
    flip_records: list[dict] = []
    decision_rows = []
    for item in metric_artifacts:
        if item.get("failure"):
            continue
        artifact = item["artifact"]
        raw_score = artifact["raw_score"]
        evaluations, flips, ref_eval = diagnose_flips(
            artifact["column"], artifact["window_index"],
            raw_score, threshold, tiers)
        row = {
            "column": artifact["column"],
            "window_index": artifact["window_index"],
            "metric": artifact["metric"],
            "raw_score": raw_score,
            "sort_key": artifact["sort_key"],
            "tiers": [_eval_payload(e) for e in evaluations],
            "reference_evaluation": _eval_payload(ref_eval),
        }
        decision_rows.append(row)
        per_column.setdefault(artifact["column"], []).append(row)
        for flip in flips:
            involves_active = flip.higher_tier == active_tier.name or \
                flip.lower_tier == active_tier.name
            flip_records.append({
                "column": flip.column,
                "window_index": flip.window_index,
                "lower_tier": flip.lower_tier,
                "higher_tier": flip.higher_tier,
                "lower_score_q": flip.lower_score_q,
                "higher_score_q": flip.higher_score_q,
                "lower_retained": flip.lower_retained,
                "higher_retained": flip.higher_retained,
                "raw_score": flip.raw_score,
                "absolute_q_diff": flip.absolute_q_diff,
                "tolerance_q": flip.tolerance_q,
                "first_breach_stage": flip.first_breach_stage,
                "flipped": flip.flipped,
                "active_tier_pair": involves_active,
                "reference_score_q": flip.reference_score_q,
                "reference_retained": flip.reference_retained,
                "reference_tier": flip.reference_tier,
            })
    payload = {
        "threshold": threshold,
        "tier_order": [t.name for t in tiers],
        "active_tier": active_tier.name,
        "decisions": decision_rows,
        "flips": flip_records,
    }
    params = {"threshold": threshold,
              "tiers": [(t.name, t.decimals, t.clamp) for t in tiers],
              "active_tier": active_tier.name}
    metric_fp = fingerprint([a.get("key") for a in metric_artifacts])
    key = derived_key(metric_fp, params, "decision")
    cached = ctx.cache.get("decision", key)
    if cached is None:
        ctx.cache.put("decision", key, payload)
        ctx.recomputed.append(key)
    else:
        ctx.reused.append(key)
    return {"key": key, "payload": payload}


def _eval_payload(evaluation: TierEvaluation) -> dict:
    return {
        "tier": evaluation.tier,
        "decimals": evaluation.decimals,
        "score_q": evaluation.score_q,
        "score_retained": evaluation.score_retained,
        "threshold_q": evaluation.threshold_q,
        "significant": evaluation.significant,
    }


def make_decision_record_payload(decision: DecisionRecord) -> dict:
    return {
        "indicators": decision.indicators.__dict__,
        "candidates": [
            {"strategy": c.strategy, "score": c.score, "rationale": c.rationale}
            for c in decision.candidates
        ],
        "selected": decision.selected,
        "reasons": list(decision.reasons),
        "effective_windows": decision.effective_windows,
        "approximate_quantile": decision.approximate_quantile,
        "metric_overrides": dict(decision.metric_overrides),
    }
