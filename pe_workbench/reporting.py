"""Deterministic JSON, SVG, and HTML diagnostic outputs."""

from __future__ import annotations

import html
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from .cache_keys import stage_keys
from .engine import STAGES, RunResult, WorkbenchEngine
from .hashing import sha256_json, stable_float, stable_json
from .models import Issue


def _quantile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return stable_float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return stable_float(sorted_values[lower])
    fraction = position - lower
    value = sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    return stable_float(value)


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "empty_layer": True,
            "mean": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "empty_layer": False,
        "mean": stable_float(math.fsum(ordered) / len(ordered)),
        "min": stable_float(ordered[0]),
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "p90": _quantile(ordered, 0.90),
        "p99": _quantile(ordered, 0.99),
        "max": stable_float(ordered[-1]),
    }


def _histogram(values: list[float], bins: int = 8) -> list[dict[str, Any]]:
    if not values:
        return []
    ordered = sorted(float(value) for value in values)
    minimum = ordered[0]
    maximum = ordered[-1]
    if minimum == maximum:
        return [
            {
                "lower": stable_float(minimum),
                "upper": stable_float(maximum),
                "count": len(ordered),
            }
        ]
    width = (maximum - minimum) / bins
    counts = [0] * bins
    for value in ordered:
        index = min(bins - 1, int((value - minimum) / width))
        counts[index] += 1
    return [
        {
            "lower": stable_float(minimum + index * width),
            "upper": stable_float(minimum + (index + 1) * width),
            "count": count,
        }
        for index, count in enumerate(counts)
    ]


def _layer_statistics(
    diagnostics: list[dict[str, Any]],
    train_context_length: int,
    target_length: int,
) -> list[dict[str, Any]]:
    observed_max = 0
    values: dict[tuple[int, str], list[float]] = {}
    for diagnostic in diagnostics:
        for index, score_record in enumerate(diagnostic["scores"]):
            distance = int(score_record["relative_distance"])
            observed_max = max(observed_max, distance)
            region = "extrapolation" if distance >= train_context_length else "in_training"
            values.setdefault((distance, region), []).append(float(score_record["score"]))
    maximum_distance = max(target_length - 1, observed_max)
    rows: list[dict[str, Any]] = []
    for distance in range(maximum_distance + 1):
        row: dict[str, Any] = {"distance": distance, "layers": {}}
        for region in ("in_training", "extrapolation"):
            region_values = values.get((distance, region), [])
            layer = _distribution(region_values)
            if layer["count"] == 0:
                layer = {
                    "count": 0,
                    "empty_layer": True,
                    "mean": None,
                    "min": None,
                    "max": None,
                }
            row["layers"][region] = layer
        rows.append(row)
    return rows


def _stage_distributions(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, Any] = {}
    for stage in ("position_encoding", "dot_score", "attention_weight"):
        values = [
            float(diagnostic["stage_errors"][stage])
            for diagnostic in diagnostics
            if diagnostic["stage_errors"].get(stage) is not None
        ]
        by_stage[stage] = {
            "distribution": _distribution(values),
            "histogram": _histogram(values),
        }
    return by_stage


def _over_tolerance(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage_rank = {"position_encoding": 0, "dot_score": 1, "attention_weight": 2}
    rows: list[dict[str, Any]] = []
    for diagnostic in diagnostics:
        stage = diagnostic["first_exceeded_stage"]
        if stage is None:
            continue
        rows.append(
            {
                "sample_id": diagnostic["sample_id"],
                "first_exceeded_stage": stage,
                "first_exceeded_value": diagnostic["first_exceeded_value"],
                "stage_errors": diagnostic["stage_errors"],
                "attention_overtolerance": diagnostic["attention_overtolerance"],
                "query_position": diagnostic["query_position"],
                "relative_distances": [score["relative_distance"] for score in diagnostic["scores"]],
            }
        )
    rows.sort(
        key=lambda row: (
            stage_rank.get(row["first_exceeded_stage"], 99),
            -float(row["first_exceeded_value"] or 0.0),
            row["sample_id"],
        )
    )
    return rows


def _boundary_samples(
    diagnostics: list[dict[str, Any]],
    train_context_length: int,
    limit: int = 12,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for diagnostic in diagnostics:
        distances = [int(score["relative_distance"]) for score in diagnostic["scores"]]
        if not distances:
            continue
        closest = min(distances, key=lambda distance: (abs(distance - train_context_length), distance))
        gap = abs(closest - train_context_length)
        attention = diagnostic["stage_errors"].get("attention_weight")
        rows.append(
            {
                "sample_id": diagnostic["sample_id"],
                "closest_boundary_distance": closest,
                "boundary_gap": gap,
                "first_exceeded_stage": diagnostic["first_exceeded_stage"],
                "first_exceeded_value": diagnostic["first_exceeded_value"],
                "attention_weight_error": attention,
                "position_vector_error": diagnostic["stage_errors"].get("position_encoding"),
                "dot_score_error": diagnostic["stage_errors"].get("dot_score"),
                "attention_overtolerance": diagnostic["attention_overtolerance"],
            }
        )
    rows.sort(
        key=lambda row: (
            row["boundary_gap"],
            -float(row["attention_weight_error"] or 0.0),
            row["sample_id"],
        )
    )
    return rows[:limit]


def _canonical_work_plan(engine: WorkbenchEngine) -> dict[str, Any]:
    recomputed: list[dict[str, Any]] = []
    active_keys: dict[str, list[str]] = {stage: [] for stage in STAGES}
    for sample in engine.samples:
        keys = stage_keys(sample, engine.config)
        for stage in STAGES:
            active_keys[stage].append(keys["load" if stage == "loading" else stage])
        for stage in ("position_encoding", "scoring", "error_tracking"):
            recomputed.append(
                {
                    "sample_id": sample.sample_id,
                    "scheme": engine.config.scheme,
                    "stage": stage,
                    "cache_key": keys[stage],
                }
            )
    recomputed.sort(key=lambda row: (row["sample_id"], row["stage"]))
    return {
        "contract": "canonical cold-rebuild view; actual reuse telemetry is written to state/run_trace.json",
        "ordering": "sample_id, stage",
        "canonical_reused": [],
        "canonical_recomputed": recomputed,
        "active_stage_keys": {
            stage: sorted(keys) for stage, keys in active_keys.items()
        },
    }


def build_diagnostic_report(
    engine: WorkbenchEngine,
    run_result: RunResult,
    sensitivity: dict[str, Any] | None,
    validation_issues: list[Issue],
    command_issues: list[Issue],
) -> dict[str, Any]:
    diagnostics = engine.diagnostics_for_completed(run_result.completed_samples)
    failures = [*validation_issues, *command_issues, *run_result.pending_issues]
    failures.sort(key=lambda issue: issue.sort_key)
    stage_distributions = _stage_distributions(diagnostics)
    over_tolerance = _over_tolerance(diagnostics)
    boundary = _boundary_samples(diagnostics, engine.config.train_context_length)
    layer_statistics = _layer_statistics(
        diagnostics,
        engine.config.train_context_length,
        engine.config.extrapolation_target_length,
    )
    first_stage_counts = {
        stage: 0 for stage in ("position_encoding", "dot_score", "attention_weight", "none")
    }
    for diagnostic in diagnostics:
        first_stage_counts[diagnostic["first_exceeded_stage"] or "none"] += 1
    config = engine.config
    report: dict[str, Any] = {
        "report_version": "diagnostic-report-v1",
        "metadata": {
            "scheme": config.scheme,
            "scheme_display": config.scheme_display,
            "base_frequency": stable_float(config.base_frequency),
            "scale_factor": stable_float(config.scale_factor),
            "precision_level": config.precision_level,
            "quantization_level": config.quantization_level,
            "fixed_decimal_bits": config.fixed_decimal_bits,
            "dimension": config.dimension,
            "dimension_block_size": config.dimension_block_size,
            "train_context_length": config.train_context_length,
            "extrapolation_target_length": config.extrapolation_target_length,
            "tolerances": config.tolerances.to_dict(),
            "thread_pool_size_limit": config.thread_pool_size,
            "config_hash": run_result.config_hash,
            "sample_content_hash": engine.sample_content_hash,
            "valid_sample_count": len(engine.samples),
            "completed_sample_count": len(run_result.completed_samples),
            "pending_sample_count": len(run_result.pending_samples),
            "all_samples_complete": not run_result.pending_samples,
        },
        "validation": {
            "failure_count": len(failures),
            "failures": [issue.to_dict() for issue in failures],
            "valid_sample_ids": [sample.sample_id for sample in engine.samples],
        },
        "completion": {
            "completed_sample_ids": run_result.completed_samples,
            "pending_sample_ids": run_result.pending_samples,
            "resume_cursor": run_result.resume_cursor,
            "budget_telemetry_location": "state/run_trace.json",
        },
        "sample_diagnostics": diagnostics,
        "stage_error_distributions": stage_distributions,
        "first_exceeded_stage_counts": first_stage_counts,
        "over_tolerance_samples": over_tolerance,
        "boundary_sensitive_samples": boundary,
        "layer_statistics": {
            "region_rule": "in_training when abs(query_position - key_position) < train_context_length; otherwise extrapolation",
            "empty_layer_policy": "empty layers use count=0, mean/min/max=null, empty_layer=true",
            "buckets": layer_statistics,
        },
        "sensitivity": sensitivity
        or {
            "available": False,
            "ranking": [],
            "flip_attribution": {"total_flips": 0, "cases": []},
        },
        "reuse_recompute": _canonical_work_plan(engine),
        "source_of_truth": {
            "report": "diagnostic_report.json",
            "visual_summary": "visual_summary.json",
            "svg": "summary.svg",
            "dashboard": "dashboard.html",
            "execution_trace": "state/run_trace.json",
        },
    }
    report_body = dict(report)
    report_id = sha256_json(report_body)
    report["report_id"] = report_id
    return report


def build_visual_summary(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report["metadata"]
    stage_rows = []
    for stage, payload in report["stage_error_distributions"].items():
        distribution = payload["distribution"]
        stage_rows.append(
            {
                "stage": stage,
                "mean": distribution["mean"],
                "p90": distribution["p90"],
                "max": distribution["max"],
                "count": distribution["count"],
            }
        )
    sensitivity = report["sensitivity"]
    observed_layers = [
        {
            "distance": row["distance"],
            **row["layers"],
        }
        for row in report["layer_statistics"]["buckets"]
        if row["layers"]["in_training"]["count"] or row["layers"]["extrapolation"]["count"]
    ]
    flips = sensitivity.get("flip_attribution", {})
    return {
        "visual_version": "visual-summary-v1",
        "report_id": report["report_id"],
        "scheme": metadata["scheme"],
        "scheme_display": metadata["scheme_display"],
        "sample_count": metadata["completed_sample_count"],
        "all_samples_complete": metadata["all_samples_complete"],
        "stage_error_bars": stage_rows,
        "sensitivity_ranking": sensitivity.get("ranking", []),
        "data_scale_curves": sensitivity.get("data_scale_sensitivity", []),
        "flip_attribution_counts": flips.get("counts_by_source", {}),
        "flip_attribution_total": flips.get("total_flips", 0),
        "boundary_sensitive_samples": report["boundary_sensitive_samples"],
        "over_tolerance_count": len(report["over_tolerance_samples"]),
        "observed_layer_buckets": observed_layers,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6e}"
    return str(value)


def _bar_width(value: float, maximum: float, width: float) -> float:
    if maximum <= 0:
        return 0.0
    scaled = math.log1p(max(0.0, value)) / math.log1p(maximum)
    return max(0.0, min(width, scaled * width))


def render_svg(visual: dict[str, Any]) -> str:
    width = 1200
    height = 840
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Positional encoding diagnostics">'
    )
    parts.append('<rect width="100%" height="100%" fill="#f7f8fb"/>')
    parts.append(
        '<text x="40" y="52" font-family="Segoe UI, Arial, sans-serif" font-size="26" font-weight="700" fill="#172033">Long-context Positional Encoding Diagnostics</text>'
    )
    parts.append(
        f'<text x="40" y="82" font-family="Consolas, monospace" font-size="14" fill="#4c5a73">scheme={xml_escape(str(visual.get("scheme")))} samples={visual.get("sample_count")} complete={visual.get("all_samples_complete")} report={xml_escape(str(visual.get("report_id", ""))[:20])}</text>'
    )
    # Stage errors panel.
    parts.append(
        '<rect x="32" y="108" width="548" height="310" rx="10" fill="#ffffff" stroke="#dce2ee"/>'
    )
    parts.append(
        '<text x="52" y="138" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="600" fill="#172033">Error by pipeline stage</text>'
    )
    stage_rows = visual.get("stage_error_bars", [])
    stage_max = max([float(row["max"] or 0.0) for row in stage_rows] or [0.0])
    for index, row in enumerate(stage_rows):
        y = 176 + index * 66
        value = float(row["max"] or 0.0)
        bar_width = _bar_width(value, stage_max, 330)
        parts.append(
            f'<text x="52" y="{y + 16}" font-family="Consolas, monospace" font-size="13" fill="#34405a">{xml_escape(row["stage"])}</text>'
        )
        parts.append(
            f'<rect x="220" y="{y}" width="330" height="24" rx="4" fill="#edf1f8"/>'
        )
        parts.append(
            f'<rect x="220" y="{y}" width="{bar_width:.2f}" height="24" rx="4" fill="#3268d6"/>'
        )
        parts.append(
            f'<text x="562" y="{y + 17}" text-anchor="end" font-family="Consolas, monospace" font-size="12" fill="#34405a">{_fmt(row["max"])}</text>'
        )
    # Sensitivity panel.
    parts.append(
        '<rect x="620" y="108" width="548" height="310" rx="10" fill="#ffffff" stroke="#dce2ee"/>'
    )
    parts.append(
        '<text x="640" y="138" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="600" fill="#172033">Sensitivity ranking</text>'
    )
    ranking = visual.get("sensitivity_ranking", [])[:5]
    ranking_max = max([float(row.get("impact") or 0.0) for row in ranking] or [0.0])
    for index, row in enumerate(ranking):
        y = 176 + index * 44
        value = float(row.get("impact") or 0.0)
        bar_width = _bar_width(value, ranking_max, 230)
        parts.append(
            f'<text x="640" y="{y + 16}" font-family="Consolas, monospace" font-size="12" fill="#34405a">#{row.get("rank", index + 1)} {xml_escape(str(row.get("parameter")))}</text>'
        )
        parts.append(f'<rect x="830" y="{y}" width="230" height="22" rx="4" fill="#edf1f8"/>')
        parts.append(
            f'<rect x="830" y="{y}" width="{bar_width:.2f}" height="22" rx="4" fill="#0b9b78"/>'
        )
        parts.append(
            f'<text x="1148" y="{y + 16}" text-anchor="end" font-family="Consolas, monospace" font-size="11" fill="#34405a">{_fmt(value)}</text>'
        )
    # Flip attribution panel.
    parts.append(
        '<rect x="32" y="438" width="548" height="176" rx="10" fill="#ffffff" stroke="#dce2ee"/>'
    )
    parts.append(
        '<text x="52" y="468" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="600" fill="#172033">Precision/quantization flip attribution</text>'
    )
    flips = visual.get("flip_attribution_counts", {})
    flip_max = max([float(value) for value in flips.values()] or [0.0])
    colors = {"position_encoding_angle": "#d17b00", "quantization_rounding": "#8a4fd3", "normalization_denominator": "#cf3f4f"}
    for index, (source, count) in enumerate(sorted(flips.items())):
        y = 494 + index * 34
        bar_width = (float(count) / flip_max * 300) if flip_max > 0 else 0.0
        parts.append(
            f'<text x="52" y="{y + 15}" font-family="Consolas, monospace" font-size="12" fill="#34405a">{xml_escape(source)}</text>'
        )
        parts.append(f'<rect x="300" y="{y}" width="300" height="20" rx="4" fill="#edf1f8"/>')
        parts.append(
            f'<rect x="300" y="{y}" width="{bar_width:.2f}" height="20" rx="4" fill="{colors.get(source, "#555")}"/>'
        )
        parts.append(
            f'<text x="620" y="{y + 15}" text-anchor="end" font-family="Consolas, monospace" font-size="12" fill="#34405a">{count}</text>'
        )
    # Boundary sample table.
    parts.append(
        '<rect x="620" y="438" width="548" height="360" rx="10" fill="#ffffff" stroke="#dce2ee"/>'
    )
    parts.append(
        '<text x="640" y="468" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="600" fill="#172033">Boundary-sensitive samples</text>'
    )
    parts.append(
        '<text x="640" y="488" font-family="Consolas, monospace" font-size="11" fill="#68748b">sample / gap / first stage / attention error</text>'
    )
    for index, row in enumerate(visual.get("boundary_sensitive_samples", [])[:8]):
        y = 514 + index * 32
        text = (
            f'{row["sample_id"]}  gap={row["boundary_gap"]}  '
            f'{row["first_exceeded_stage"] or "none"}  '
            f'{_fmt(row["attention_weight_error"])}'
        )
        parts.append(
            f'<text x="640" y="{y}" font-family="Consolas, monospace" font-size="12" fill="#34405a">{xml_escape(text)}</text>'
        )
    # Footer.
    parts.append(
        '<text x="32" y="824" font-family="Consolas, monospace" font-size="11" fill="#68748b">JSON report contract: deterministic ordering, empty layers marked null, content-addressed stage cache.</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def render_dashboard(report: dict[str, Any], svg: str) -> str:
    metadata = report["metadata"]
    ranking_rows = "".join(
        "<tr>"
        f"<td>{row.get('rank')}</td>"
        f"<td>{html.escape(str(row.get('parameter')))}</td>"
        f"<td>{_fmt(row.get('impact'))}</td>"
        f"<td>{_fmt(row.get('min'))}</td>"
        f"<td>{_fmt(row.get('max'))}</td>"
        "</tr>"
        for row in report["sensitivity"].get("ranking", [])
    )
    tolerance_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['sample_id'])}</td>"
        f"<td>{html.escape(str(row['first_exceeded_stage']))}</td>"
        f"<td>{_fmt(row['first_exceeded_value'])}</td>"
        f"<td>{_fmt(row['stage_errors'].get('position_encoding'))}</td>"
        f"<td>{_fmt(row['stage_errors'].get('dot_score'))}</td>"
        f"<td>{_fmt(row['stage_errors'].get('attention_weight'))}</td>"
        "</tr>"
        for row in report["over_tolerance_samples"][:40]
    )
    failure_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(issue.get('code')))}</td>"
        f"<td>{html.escape(str(issue.get('sample_id')))}</td>"
        f"<td>{html.escape(str(issue.get('field')))}</td>"
        f"<td>{html.escape(str(issue.get('message')))}</td>"
        "</tr>"
        for issue in report["validation"]["failures"][:60]
    )
    inner_svg = svg.split("?>", 1)[-1]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PE Diagnostics - {html.escape(metadata['scheme'])}</title>
<style>
body {{ margin: 0; background: #eef1f7; color: #172033; font-family: Segoe UI, Arial, sans-serif; }}
main {{ max-width: 1260px; margin: 20px auto 48px; padding: 0 20px; }}
.card {{ background: #fff; border: 1px solid #dce2ee; border-radius: 12px; padding: 20px; margin: 16px 0; box-shadow: 0 4px 18px rgba(25,38,70,.05); }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ text-align: left; padding: 7px 9px; border-bottom: 1px solid #e5e9f1; }}
th {{ background: #f4f6fa; color: #475570; }}
code {{ background: #f1f3f8; padding: 2px 5px; border-radius: 4px; }}
.meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 8px; font-size: 13px; }}
.meta div {{ background: #f7f8fb; padding: 8px 10px; border-radius: 7px; }}
svg {{ width: 100%; height: auto; border-radius: 10px; }}
</style>
</head>
<body><main>
<div class="card"><h1>长上下文位置编码外推诊断工作台</h1>
<div class="meta">
<div>方案 <b>{html.escape(metadata['scheme'])}</b> / {html.escape(metadata['scheme_display'])}</div>
<div>基频 <code>{metadata['base_frequency']}</code></div>
<div>缩放因子 <code>{metadata['scale_factor']}</code></div>
<div>训练/目标 <code>{metadata['train_context_length']} / {metadata['extrapolation_target_length']}</code></div>
<div>维度 <code>{metadata['dimension']}</code>，块 <code>{metadata['dimension_block_size']}</code></div>
<div>精度/量化/位宽 <code>{metadata['precision_level']} / {metadata['quantization_level']} / {metadata['fixed_decimal_bits']}</code></div>
<div>完成样本 <code>{metadata['completed_sample_count']} / {metadata['valid_sample_count']}</code></div>
<div>报告 ID <code>{html.escape(report['report_id'][:20])}</code></div>
</div></div>
<div class="card">{inner_svg}</div>
<div class="card"><h2>敏感性排序</h2><table><thead><tr><th>#</th><th>参数</th><th>影响</th><th>最小值</th><th>最大值</th></tr></thead><tbody>{ranking_rows}</tbody></table></div>
<div class="card"><h2>超差样本</h2><table><thead><tr><th>样本</th><th>首次超差环节</th><th>偏差</th><th>位置</th><th>打分</th><th>注意力权重</th></tr></thead><tbody>{tolerance_rows}</tbody></table></div>
<div class="card"><h2>失败/待办清单</h2><table><thead><tr><th>业务码</th><th>样本 ID</th><th>字段</th><th>说明</th></tr></thead><tbody>{failure_rows}</tbody></table></div>
</main></body></html>
"""


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def write_reports(reports_dir: str | Path, report: dict[str, Any], visual: dict[str, Any]) -> dict[str, str]:
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".reports-", dir=directory.parent))
    try:
        report_text = stable_json(report, pretty=True) + "\n"
        visual_text = stable_json(visual, pretty=True) + "\n"
        svg_text = render_svg(visual) + "\n"
        dashboard_text = render_dashboard(report, svg_text)
        (staging / "diagnostic_report.json").write_text(report_text, encoding="utf-8")
        (staging / "visual_summary.json").write_text(visual_text, encoding="utf-8")
        (staging / "summary.svg").write_text(svg_text, encoding="utf-8")
        (staging / "dashboard.html").write_text(dashboard_text, encoding="utf-8")
        paths: dict[str, str] = {}
        for name in (
            "diagnostic_report.json",
            "visual_summary.json",
            "summary.svg",
            "dashboard.html",
        ):
            destination = directory / name
            os.replace(staging / name, destination)
            paths[name] = str(destination)
        return paths
    finally:
        shutil.rmtree(staging, ignore_errors=True)
