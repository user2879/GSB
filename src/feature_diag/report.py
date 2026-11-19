"""Deterministic machine-readable and Markdown diagnostic reports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from . import ENGINE_NAME, ENGINE_VERSION
from .config import ExperimentConfig, REDUCTION_PATHS
from .errors import E_BOUNDARY_SENSITIVE
from .fingerprint import fingerprint
from .serialization import atomic_write_text, dump_json


def _comparison_public(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": comparison["scenario_id"],
        "sample_size": comparison["sample_size"],
        "order_limit": comparison["order_limit"],
        "path": comparison["path"],
        "threshold": comparison["threshold"],
        "decision_metric": comparison["decision_metric"],
        "metrics_float32": comparison["metrics_float32"],
        "metrics_float64": comparison["metrics_float64"],
        "decision_true_count_float32": comparison["decision_true_count_float32"],
        "decision_true_count_float64": comparison["decision_true_count_float64"],
        "stage_summary": comparison["stage_summary"],
        "feature_error_summary": comparison["feature_error_summary"],
        "max_stage_error": comparison["max_stage_error"],
        "comparison_fingerprint": comparison["comparison_fingerprint"],
        "stage_deltas": [
            {
                "sample_id": sample["sample_id"],
                "stage_errors": sample["stage_errors"],
                "first_exceeded_stage": sample["first_exceeded_stage"],
                "primary_error_stage": sample["primary_error_stage"],
                "metric_float32": sample["metric_float32"],
                "metric_float64": sample["metric_float64"],
                "metric_delta": sample["metric_delta"],
                "decision_float32": sample["decision_float32"],
                "decision_float64": sample["decision_float64"],
            }
            for sample in comparison["per_sample"]
        ],
    }


def _cross_diagnostics(designs: dict[str, dict[str, Any]], plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for task in plan:
        for precision in ("float32", "float64"):
            key = f"{task['scenario_id']}|n{task['sample_size']}|o{task['order_limit']}|{precision}"
            if key in seen:
                continue
            seen.add(key)
            design = designs.get(key)
            if design is None:
                continue
            output.append(
                {
                    "scenario_id": task["scenario_id"],
                    "sample_size": task["sample_size"],
                    "order_limit": task["order_limit"],
                    "precision": precision,
                    "processed_order": design["cross_processed_order"],
                    "accepted_features": [
                        {
                            "name": feature["name"],
                            "columns": feature["columns"],
                            "order": feature["order"],
                            "unique_count": feature["unique_count"],
                            "effective_unique_count": feature["effective_unique_count"],
                            "downgraded": feature["downgraded"],
                            "policy": feature["policy"],
                            "hash_prefix": feature["hash_prefix"],
                            "fingerprint": feature["fingerprint"],
                        }
                        for feature in design["cross_features"]
                    ],
                    "skipped": design["cross_skipped"],
                    "downgrades": design["cross_downgrades"],
                    "missing_counts": design["missing_counts"],
                    "design_fingerprint": design["fingerprint"],
                }
            )
    output.sort(key=lambda item: (item["scenario_id"], item["sample_size"], item["order_limit"], item["precision"]))
    return output


def _reduction_diagnostics(reductions: dict[str, dict[str, Any]], plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in plan:
        for precision in ("float32", "float64"):
            key = (
                f"{task['scenario_id']}|n{task['sample_size']}|o{task['order_limit']}|"
                f"{task['path']}|{precision}"
            )
            reduction = reductions.get(key)
            if reduction is None:
                continue
            summary = dict(reduction["summary"])
            output.append(
                {
                    "scenario_id": task["scenario_id"],
                    "sample_size": task["sample_size"],
                    "order_limit": task["order_limit"],
                    "path": task["path"],
                    "precision": precision,
                    "retained_dimension": reduction["retained_dimension"],
                    "sample_ids": [str(value) for value in reduction["sample_ids"]],
                    "projection_input_features": reduction["projection_input_features"],
                    "candidate_dimensions": reduction["candidate_dimensions"],
                    "summary": summary,
                    "fingerprint": reduction["fingerprint"],
                }
            )
    output.sort(key=lambda item: (item["scenario_id"], item["sample_size"], item["order_limit"], item["path"], item["precision"]))
    return output


def build_report(
    *,
    experiment: ExperimentConfig,
    plan: list[dict[str, Any]],
    designs: dict[str, dict[str, Any]],
    reductions: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    precision_sensitive: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    cache_ledger: dict[str, Any],
    failures: list[dict[str, Any]],
    snapshot_info: dict[str, Any],
    execution_info: dict[str, Any],
) -> dict[str, Any]:
    comparison_rows = [
        _comparison_public(comparisons[task["task_id"]])
        for task in plan
        if task["task_id"] in comparisons
    ]
    comparison_rows.sort(
        key=lambda item: (
            item["scenario_id"],
            item["sample_size"],
            item["order_limit"],
            REDUCTION_PATHS.index(item["path"]),
        )
    )
    boundary_counts = Counter(item["business_code"] or "near_tie_only" for item in boundaries)
    origin_counts = Counter(item["flip_origin"] for item in boundaries if item["flip_origin"])
    sensitive_boundaries = [item for item in boundaries if item["business_code"] == E_BOUNDARY_SENSITIVE]
    failure_rows = sorted(
        failures,
        key=lambda item: (item.get("scope", ""), item.get("field", ""), item.get("reason", ""), item.get("code", "")),
    )
    precision_rows = sorted(
        precision_sensitive,
        key=lambda item: (
            item["scenario_id"],
            item["sample_size"],
            item["order_limit"],
            item["path"],
            item["sample_id"],
        ),
    )
    cross_rows = _cross_diagnostics(designs, plan)
    reduction_rows = _reduction_diagnostics(reductions, plan)
    analysis_payload = {
        "config_hash": experiment.config_hash,
        "run_version": experiment.version,
        "schema_version": experiment.schema_version,
        "data_spec_hash": experiment.data_spec.content_hash,
        "comparisons": comparison_rows,
        "cross_diagnostics": cross_rows,
        "reduction_diagnostics": reduction_rows,
        "precision_sensitive": precision_rows,
        "boundaries": boundaries,
        "failures": failure_rows,
    }
    analysis_fingerprint = fingerprint(analysis_payload)
    report = {
        "schema_version": 1,
        "engine": {"name": ENGINE_NAME, "version": ENGINE_VERSION},
        "config": {
            "source": Path(experiment.source_path).name,
            "hash": experiment.config_hash,
            "run_version": experiment.version,
            "schema_version": experiment.schema_version,
            "seed": experiment.seed,
            "data_spec_name": experiment.data_spec.name,
            "data_spec_hash": experiment.data_spec.content_hash,
            "tolerances": experiment.tolerances,
            "boundary": experiment.boundary,
        },
        "execution": execution_info,
        "summary": {
            "valid_scheme_count": len(experiment.scenarios),
            "evaluated_task_count": len(comparison_rows),
            "failure_count": len(failure_rows),
            "precision_sensitive_sample_count": len(precision_rows),
            "boundary_candidate_count": len(boundaries),
            "boundary_sensitive_count": len(sensitive_boundaries),
            "boundary_status_counts": dict(sorted(boundary_counts.items())),
            "boundary_origin_counts": dict(sorted(origin_counts.items())),
            "cache": cache_ledger.get("summary", {}),
            "snapshot_recovery_point": snapshot_info.get("recovery_point"),
        },
        "scheme_metrics": comparison_rows,
        "precision_sensitive": precision_rows,
        "boundary_diagnostics": boundaries,
        "cross_diagnostics": cross_rows,
        "reduction_diagnostics": reduction_rows,
        "failures": failure_rows,
        "cache_ledger": cache_ledger,
        "snapshot": snapshot_info,
        "analysis_fingerprint": analysis_fingerprint,
    }
    report["report_fingerprint"] = fingerprint(report)
    return report


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    config = report["config"]
    lines = [
        "# 特征交叉与降维精度诊断报告",
        "",
        f"- 引擎：`{report['engine']['name']} {report['engine']['version']}`",
        f"- 配置哈希：`{config['hash']}`",
        f"- 分析指纹：`{report['analysis_fingerprint']}`",
        f"- 报告指纹：`{report['report_fingerprint']}`",
        f"- 数据描述哈希：`{config['data_spec_hash']}`",
        "",
        "## 摘要",
        "",
        f"- 有效方案：{summary['valid_scheme_count']}",
        f"- 评估任务：{summary['evaluated_task_count']}",
        f"- 失败记录：{summary['failure_count']}",
        f"- 精度敏感样本：{summary['precision_sensitive_sample_count']}",
        f"- 边界候选：{summary['boundary_candidate_count']}，其中敏感翻转：{summary['boundary_sensitive_count']}",
        f"- 缓存命中/失效/未命中/快照复用：{summary['cache'].get('hit', 0)}/{summary['cache'].get('invalidated', 0)}/{summary['cache'].get('miss', 0)}/{summary['cache'].get('snapshot_reuse', 0)}",
        f"- 快照恢复点：`{summary.get('snapshot_recovery_point')}`",
        "",
    ]
    execution = report.get("execution", {})
    lines.extend(
        [
            "## 执行",
            "",
            f"- 配置并发上限：{execution.get('max_workers_requested')}",
            f"- 本次有效并发上限：{execution.get('max_workers_effective')}",
            f"- 稳定任务键（前 5 个）：`{[item.get('stable_sort_key') for item in execution.get('tasks', [])[:5]]}`",
            "",
        ]
    )

    if report["failures"]:
        lines.extend(["## 失败清单", "", "| 业务码 | 方案 | 字段 | 原因 | 说明 |", "|---|---|---|---|---|"])
        for failure in report["failures"]:
            lines.append(
                f"| `{failure['code']}` | `{failure['scope']}` | `{failure['field']}` | `{failure['reason']}` | {failure['message']} |"
            )
        lines.append("")

    if report["precision_sensitive"]:
        lines.extend(
            [
                "## 精度敏感样本",
                "",
                "| 方案 | 样本 | 阶数 | 路径 | float32 | float64 | 阈值 | 主误差环节 | 来源 |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for item in report["precision_sensitive"]:
            lines.append(
                "| {scenario} | {sample} | {order} | {path} | {v32} | {v64} | {threshold} | {stage} | {origin} |".format(
                    scenario=item["scenario_id"],
                    sample=item["sample_id"],
                    order=item["order_limit"],
                    path=item["path"],
                    v32=_format_float(item["metric_float32"]),
                    v64=_format_float(item["metric_float64"]),
                    threshold=_format_float(item["threshold"]),
                    stage=item["primary_error_stage"],
                    origin=item["error_origin"],
                )
            )
        lines.append("")

    lines.extend(
        [
            "## 方案指标与逐级误差",
            "",
            "| 方案 | n | 阶数 | 路径 | 维度 | 首越界样本 | 交叉误差 | 标准化误差 | 投影误差 | 最终误差 |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["scheme_metrics"]:
        first_counts = Counter(
            sample["first_exceeded_stage"] for sample in row["stage_deltas"] if sample["first_exceeded_stage"]
        )
        max_errors = row["max_stage_error"]
        retained = report["reduction_diagnostics"]
        retained_dim = next(
            (
                item["retained_dimension"]
                for item in retained
                if item["scenario_id"] == row["scenario_id"]
                and item["sample_size"] == row["sample_size"]
                and item["order_limit"] == row["order_limit"]
                and item["path"] == row["path"]
                and item["precision"] == "float64"
            ),
            "-",
        )
        lines.append(
            "| {scenario} | {size} | {order} | {path} | {dim} | {first} | {cross} | {std} | {proj} | {final} |".format(
                scenario=row["scenario_id"],
                size=row["sample_size"],
                order=row["order_limit"],
                path=row["path"],
                dim=retained_dim,
                first=sum(first_counts.values()),
                cross=_format_float(max_errors["cross_encoding"]),
                std=_format_float(max_errors["standardization"]),
                proj=_format_float(max_errors["projection"]),
                final=_format_float(max_errors["final_metric"]),
            )
        )
    lines.append("")

    boundary = report["boundary_diagnostics"]
    lines.extend(
        [
            "## 近并列边界诊断",
            "",
            f"- 记录数：{len(boundary)}",
            f"- 业务码计数：`{summary['boundary_status_counts']}`",
            f"- 翻转来源计数：`{summary['boundary_origin_counts']}`",
            "",
            "| 类型 | 方案 | n | 阶数 | 路径 | 左 | 右 | float64 差值 | float32 判定 | float64 判定 | 业务码 | 来源 |",
            "|---|---|---:|---:|---|---|---|---:|---|---|---|---|",
        ]
    )
    for item in boundary[:100]:
        lines.append(
            "| {kind} | {scenario} | {size} | {order} | {path} | {left} | {right} | {gap} | {d32} | {d64} | {code} | {origin} |".format(
                kind=item["kind"],
                scenario=item["scenario_id"],
                size=item["sample_size"],
                order=item["order_limit"],
                path=item["path"] or "-",
                left=item["left"],
                right=item["right"],
                gap=_format_float(item["reference_values"]["gap"]),
                d32=item["decisions"]["float32"],
                d64=item["decisions"]["float64"],
                code=item["business_code"] or "-",
                origin=item["flip_origin"] or "-",
            )
        )
    if len(boundary) > 100:
        lines.extend(["", f"仅展示前 100 条，完整记录见 JSON（共 {len(boundary)} 条）。"])
    lines.append("")

    ledger = report["cache_ledger"]
    lines.extend(
        [
            "## 缓存增量账本",
            "",
            f"- 汇总：`{ledger.get('summary', {})}`",
            "",
            "| 阶段 | 方案 | n | 阶数 | 路径 | 精度 | 状态 | 原因 |",
            "|---|---|---:|---:|---|---|---|---|",
        ]
    )
    for entry in ledger.get("entries", [])[:120]:
        lines.append(
            "| {kind} | {scenario} | {size} | {order} | {path} | {precision} | {status} | {reason} |".format(
                kind=entry.get("artifact_kind") or "-",
                scenario=entry.get("scenario_id") or "-",
                size=entry.get("sample_size") or "-",
                order=entry.get("order_limit") or "-",
                path=entry.get("path") or "-",
                precision=entry.get("precision") or "-",
                status=entry.get("status"),
                reason=entry.get("reason"),
            )
        )
    if len(ledger.get("entries", [])) > 120:
        lines.extend(["", f"仅展示前 120 条，完整账本见 JSON（共 {len(ledger.get('entries', []))} 条）。"])
    lines.append("")

    snapshot = report["snapshot"]
    lines.extend(
        [
            "## 阶段快照",
            "",
            f"- 全局版本：`{snapshot.get('global_version')}`",
            f"- 配置哈希：`{snapshot.get('config_hash')}`",
            f"- 精度指纹：`{snapshot.get('precision_fingerprints')}`",
            f"- 已恢复阶段：`{snapshot.get('resumed_stages')}`",
            f"- 本次提交阶段：`{snapshot.get('computed_stages')}`",
            f"- 恢复点：`{snapshot.get('recovery_point')}`",
            f"- 旧快照拒绝事件：`{snapshot.get('stale_event')}`",
        ]
    )
    lines.append("")

    lines.extend(
        [
            "## 复现",
            "",
            "```bash",
            "python -m venv --without-pip .venv",
            "curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py",
            ".venv/bin/python /tmp/get-pip.py",
            ".venv/bin/python -m pip install -r requirements-dev.txt",
            ".venv/bin/python -m feature_diag run --config sample/experiment.yaml --out artifacts/reports/main",
            "```",
            "",
            f"报告分析指纹：`{report['analysis_fingerprint']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "report.json"
    markdown_path = destination / "report.md"
    atomic_write_text(json_path, dump_json(report, pretty=True))
    atomic_write_text(markdown_path, render_markdown(report))
    return {"json": str(json_path), "markdown": str(markdown_path)}
