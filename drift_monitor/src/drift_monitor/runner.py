"""End-to-end run pipeline and JSON diagnostic report assembly."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .cache import StageCache, canonical_json, derived_key, fingerprint
from .config import MonitorConfig
from .datagen import expanded_columns, generate_pair
from .engine import (
    RunContext,
    _data_fingerprint,
    make_decision_record_payload,
    run_decisions,
    run_metrics,
)
from .errors import SnapshotStaleError
from .snapshot import SnapshotStore
from .strategy import select_strategy
from .visualize import build_heatmap_svg
from .windows import aligned_window_plan


def config_hash(config: MonitorConfig) -> str:
    payload = {
        "window_count": config.window_count,
        "threshold": config.significance_threshold,
        "tiers": [(t.name, t.decimals, t.clamp) for t in config.tiers],
        "active_tier": config.active_tier,
        "concurrency": config.concurrency,
        "metric_overrides": dict(sorted(config.metric_overrides.items())),
        "fail_keys": list(config.fail_keys),
        "columns": [c.__dict__ for c in config.data.base_columns],
        "seed": config.data.seed,
        "missing_rate": config.data.missing_rate,
        "size_tiers": [t.__dict__ for t in config.data.size_tiers],
    }
    return fingerprint(payload, prefix="cfg:")


def run_pipeline(config: MonitorConfig, tier_name: str, output_dir: str | Path,
                 resume: bool = False) -> dict:
    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache"
    snapshot_dir = output_dir / "snapshot"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    cache = StageCache(cache_dir)
    store = SnapshotStore(snapshot_dir)

    tier = next(t for t in config.data.size_tiers if t.name == tier_name)
    reference, current = generate_pair(config.data, tier)
    columns = expanded_columns(config.data, tier)
    cfg_hash = config_hash(config)
    data_fp = _data_fingerprint(config, tier_name, reference, current)

    resume_status = "fresh"
    previous = store.load_manifest()
    if resume:
        try:
            store.verify_resume(cfg_hash, data_fp)
            resume_status = "resumed"
        except SnapshotStaleError as exc:
            resume_status = exc.code

    decision = select_strategy(config, reference, columns)
    approx = (config.enable_approx_quantile
              if config.enable_approx_quantile is not None
              else decision.approximate_quantile)
    effective_windows = (config.window_cap if config.window_cap is not None
                         else decision.effective_windows)
    ref_windows, cur_windows = aligned_window_plan(
        reference.n_samples, current.n_samples, effective_windows)

    ctx = RunContext(
        config=config,
        tier_name=tier_name,
        reference=reference,
        current=current,
        columns=columns,
        decision=decision,
        ref_windows=ref_windows,
        cur_windows=cur_windows,
        cache=cache,
        store=store,
        config_hash=cfg_hash,
        data_fingerprint=data_fp,
        approx_quantile=approx,
        effective_metrics={c.name: _metric_for(config, decision, c.name)
                           for c in columns},
    )

    slice_key = derived_key(data_fp, {
        "window_count": effective_windows,
        "ref_windows": [(s.start, s.end, s.tag) for s in ref_windows],
        "cur_windows": [(s.start, s.end, s.tag) for s in cur_windows],
    }, "slice")
    slice_artifact = cache.get("slice", slice_key)
    if slice_artifact is None:
        slice_artifact = {
            "key": slice_key,
            "ref_windows": [asdict(s) for s in ref_windows],
            "cur_windows": [asdict(s) for s in cur_windows],
        }
        cache.put("slice", slice_key, slice_artifact)
        ctx.recomputed.append(slice_key)
    else:
        ctx.reused.append(slice_key)

    metric_artifacts = run_metrics(ctx)
    decision_bundle = run_decisions(ctx, metric_artifacts)
    viz_key, svg = build_heatmap_svg(decision_bundle["payload"])
    viz_cached = cache.get("visual", viz_key)
    if viz_cached is None:
        cache.put("visual", viz_key, {"svg": svg})
        ctx.recomputed.append(viz_key)
    else:
        ctx.reused.append(viz_key)

    strategy_payload = make_decision_record_payload(decision)
    strategy_key = derived_key(fingerprint({
        "indicators": strategy_payload["indicators"],
        "window_count": config.window_count,
    }), {"strategy": decision.selected}, "decision")
    if cache.get("decision", strategy_key) is None:
        cache.put("decision", strategy_key, strategy_payload)
        ctx.recomputed.append(strategy_key)
    else:
        ctx.reused.append(strategy_key)

    report = _assemble_report(ctx, slice_artifact, decision_bundle,
                              strategy_payload, viz_key, resume_status)
    load_payload = {
        "config_hash": cfg_hash,
        "data_fingerprint": data_fp,
        "columns": [c.name for c in columns],
        "n_samples": reference.n_samples,
    }
    load_key = derived_key(data_fp, {"config": cfg_hash}, "load")
    if cache.get("load", load_key) is None:
        cache.put("load", load_key, load_payload)
        ctx.recomputed.append(load_key)
    else:
        ctx.reused.append(load_key)

    stage_payloads = {
        "load": load_payload,
        "slice": slice_artifact,
        "metric": {"items": [
            {k: v for k, v in a.items()
             if k not in ("reused", "key") and not k.startswith("_")}
            for a in metric_artifacts if not a.get("failure")
        ]},
        "decision": {**decision_bundle["payload"], "strategy": strategy_payload},
        "visual": {"svg": svg, "key": viz_key},
    }
    stable_report = {
        key: value for key, value in report.items()
        if key not in ("incremental", "budget_deduction_sequence", "resume_status")
    }
    stage_payloads["report"] = stable_report
    manifest = store.commit(stage_payloads, cfg_hash, data_fp, previous)

    report["snapshot"] = {
        "global_version": manifest.global_version,
        "config_hash": manifest.config_hash,
        "data_fingerprint": manifest.data_fingerprint,
        "stage_versions": manifest.stage_versions,
        "resume_status": resume_status,
    }
    report_path = report_dir / f"report_{tier_name}_v{manifest.global_version}.json"
    report_path.write_text(canonical_json(report), encoding="utf-8")
    latest_path = report_dir / "latest_report.json"
    latest_path.write_text(canonical_json(report), encoding="utf-8")
    (report_dir / "heatmap.svg").write_text(svg, encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _metric_for(config: MonitorConfig, decision, column_name: str) -> str:
    if column_name in config.metric_overrides:
        return config.metric_overrides[column_name]
    if column_name in decision.metric_overrides:
        return decision.metric_overrides[column_name]
    from .datagen import expanded_columns
    for size_tier in config.data.size_tiers:
        for column in expanded_columns(config.data, size_tier):
            if column.name == column_name:
                return column.metric
    raise KeyError(column_name)


def _assemble_report(ctx: RunContext, slice_artifact: dict,
                     decision_bundle: dict, strategy_payload: dict,
                     viz_key: str, resume_status: str) -> dict:
    failures_sorted = sorted(
        ctx.failures, key=lambda f: f["sort_key"])
    return {
        "schema_version": 1,
        "tier": ctx.tier_name,
        "config_hash": ctx.config_hash,
        "data_fingerprint": ctx.data_fingerprint,
        "windows": slice_artifact,
        "strategy_decision": strategy_payload,
        "approximate_quantile": ctx.approx_quantile,
        "scores_and_judgements": decision_bundle["payload"]["decisions"],
        "flip_diagnostics": decision_bundle["payload"]["flips"],
        "incremental": {
            "reused_keys": sorted(ctx.reused),
            "recomputed_keys": sorted(ctx.recomputed),
            "reused_count": len(ctx.reused),
            "recomputed_count": len(ctx.recomputed),
            "expected_invalidated_keys": sorted(ctx.expected_invalidated),
            "expected_reusable_keys": sorted(ctx.expected_reusable),
            "required_keys": sorted(ctx.required_keys),
        },
        "budget_deduction_sequence": list(ctx.budget_log),
        "failure_list": failures_sorted,
        "visualization_key": viz_key,
        "resume_status": resume_status,
        "concurrency": ctx.config.concurrency,
    }
