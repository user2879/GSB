"""Human- and machine-readable experiment report assembly."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .diagnostics import STAGE_ORDER
from .models import AppConfig, Dataset
from .runner import RunOutcome
from .snapshot import SnapshotStore, atomic_write_json, atomic_write_text, utc_now


def _load_stage_payload(snapshot: SnapshotStore, stage: str) -> dict[str, Any]:
    artifact = snapshot.load_stage(stage)
    if artifact is None:
        return {"records": []}
    return artifact.get("payload", {"records": []})


def _stage_summary(payload: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    records = payload.get("records", [])
    channel_counts: dict[str, list[int]] = {name: [] for name in ("lexical", "collaborative", "popularity")}
    for record in records:
        for name, channel in record["channels"].items():
            channel_counts[name].append(channel["candidate_count"])
    return {
        "users": len(records),
        "mean_candidates_per_channel": {
            name: (sum(values) / len(values) if values else 0.0) for name, values in channel_counts.items()
        },
        "cold_start_users": sum(1 for record in records if record["history_item_count"] == 0),
    }


def _error_summary(fusion_records: list[dict[str, Any]]) -> dict[str, Any]:
    stage_max = {stage: 0.0 for stage in STAGE_ORDER}
    stage_exceeding = {stage: 0 for stage in STAGE_ORDER}
    first_exceeding = Counter()
    rank_change_causes = Counter()
    changed_users = 0
    candidates_with_any_exceed = 0
    for record in fusion_records:
        comparison = record["comparison"]
        if comparison["rank_changed_count"]:
            changed_users += 1
        for change in comparison["rank_changes"]:
            rank_change_causes[change["cause"]] += 1
        for candidate in comparison["candidates"]:
            if candidate["first_exceeds_tolerance"]:
                candidates_with_any_exceed += 1
                first_exceeding[candidate["first_exceeds_tolerance"]] += 1
            for stage in STAGE_ORDER:
                difference = candidate["stage_differences"].get(stage)
                if difference is None:
                    continue
                stage_max[stage] = max(stage_max[stage], float(difference))
                if difference > comparison["tolerance"]:
                    stage_exceeding[stage] += 1
    return {
        "users_with_rank_changes": changed_users,
        "rank_changes": sum(rank_change_causes.values()),
        "rank_change_causes": dict(sorted(rank_change_causes.items())),
        "candidates_with_any_tolerance_exceedance": candidates_with_any_exceed,
        "stage_max_difference": stage_max,
        "stage_exceeding_candidate_counts": stage_exceeding,
        "first_exceeding_stage_counts": dict(sorted(first_exceeding.items())),
    }


def build_report(
    dataset: Dataset,
    config: AppConfig,
    snapshot: SnapshotStore,
    outcome: RunOutcome,
    *,
    execution_mode: str,
) -> dict[str, Any]:
    recall = _load_stage_payload(snapshot, "recall")
    fusion = _load_stage_payload(snapshot, "fusion")
    evaluation = _load_stage_payload(snapshot, "evaluation")
    fusion_records = fusion.get("records", [])
    evaluation_records = evaluation.get("records", [])

    reallocation_users = 0
    reallocation_examples: list[dict[str, Any]] = []
    for record in fusion_records:
        primary = record["primary"]
        if primary["zero_channels"]:
            reallocation_users += 1
            if len(reallocation_examples) < 6:
                reallocation_examples.append(
                    {
                        "user_id": record["user_id"],
                        "zero_channels": primary["zero_channels"],
                        "weight_decomposition": primary["weight_decomposition"],
                    }
                )
    boundary_sensitive = [
        entry
        for record in fusion_records
        for entry in record.get("boundary_sensitive", [])
    ]
    failure_counts = Counter(failure.code for failure in dataset.failures)
    failure_payload = {
        "count": len(dataset.failures),
        "code_counts": dict(sorted(failure_counts.items())),
        "records": [failure.to_dict() for failure in dataset.failures],
    }
    evaluation_summary = evaluation.get("summary", {})
    report = {
        "report_schema_version": 1,
        "generated_at_utc": utc_now(),
        "reproducibility": {
            "seed": config.seed,
            "config_hash": config.config_hash,
            "run_version": snapshot.run_version,
            "input_sha256": dataset.file_hashes,
            "command": "python run.py run --input <samples> --run-dir <run-dir> --version <n> [--serial]",
            "deterministic_scope": "all scores, rankings, failure ordering, and stage artifacts; generated_at timestamps are metadata only",
        },
        "execution": {
            "mode": execution_mode,
            "status": outcome.status,
            "exit_code": outcome.exit_code,
            "stats": outcome.stats,
            "configured_max_workers": config.max_workers,
            "observed_concurrency_bound": 1 if execution_mode == "serial" else config.max_workers,
        },
        "precision": {
            "primary_bits": config.primary_bits,
            "reference_bits": config.reference_bits,
            "primary_quantization_step": config.primary_format.quantum,
            "reference_quantization_step": config.reference_format.quantum,
            "rounding": config.rounding,
            "tolerance": config.tolerance,
        },
        "validation": failure_payload,
        "business_error_codes": {
            "E_RECORD_INVALID": "invalid JSONL record or truth entry; processing continues",
            "E_CONFIG_INVALID": "configuration failed validation; no stage is started",
            "E_CONFIG_MISMATCH": "snapshot version or configuration hash is incompatible; resume is refused",
            "E_SNAPSHOT_INVALID": "snapshot artifact is structurally corrupt or internally inconsistent",
            "E_INPUT_NOT_FOUND": "required local input file is missing",
        },
        "recall": {
            **_stage_summary(recall, config),
            "channel_order": ["lexical", "collaborative", "popularity"],
            "zero_candidate_users": reallocation_users,
            "weight_reallocation_examples": reallocation_examples,
        },
        "error_propagation": {
            "summary": _error_summary(fusion_records),
            "users": [record["comparison"] for record in fusion_records],
            "boundary_sensitive_candidates": boundary_sensitive,
        },
        "evaluation": {
            "summary": evaluation_summary,
            "users": evaluation_records,
            "precision_results_differ": evaluation_summary.get("primary") != evaluation_summary.get("reference"),
        },
        "snapshot": snapshot.manifest,
        "artifacts": {
            "manifest": str(snapshot.manifest_path),
            "recall": str(snapshot.stage_path("recall")),
            "fusion": str(snapshot.stage_path("fusion")),
            "evaluation": str(snapshot.stage_path("evaluation")),
        },
    }
    digest_material = {
        "config_hash": config.config_hash,
        "run_version": snapshot.run_version,
        "precision": report["precision"],
        "validation": report["validation"],
        "recall": report["recall"],
        "error_propagation": report["error_propagation"],
        "evaluation": report["evaluation"],
        "stage_payloads": {
            "recall": recall,
            "fusion": fusion,
            "evaluation": evaluation,
        },
    }
    canonical = json.dumps(digest_material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["result_digest"] = hashlib.sha256(canonical).hexdigest()
    report["result_digest_scope"] = "substantive results and stage payloads; excludes timestamps, paths, and execution mode"
    return report


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    precision = report["precision"]
    validation = report["validation"]
    error_summary = report["error_propagation"]["summary"]
    evaluation = report["evaluation"]["summary"]
    primary_eval = evaluation.get("primary", {})
    reference_eval = evaluation.get("reference", {})
    recall = report["recall"]
    lines = [
        "# 多路召回与定点融合误差诊断实验报告",
        "",
        "## 可复现元数据",
        "",
        f"- 全局版本号：`{report['reproducibility']['run_version']}`",
        f"- 配置哈希：`{report['reproducibility']['config_hash']}`",
        f"- 固定种子：`{report['reproducibility']['seed']}`",
        f"- 主位宽 / 参考位宽：`{precision['primary_bits']}` / `{precision['reference_bits']}` 位小数",
        f"- 量化步长：`{precision['primary_quantization_step']}` / `{precision['reference_quantization_step']}`",
        f"- 舍入方式：`{precision['rounding']}`；容差：`{precision['tolerance']}`",
        f"- 结果摘要哈希：`{report['result_digest']}`",
        "",
        "## 执行与快照",
        "",
        f"- 执行模式：`{report['execution']['mode']}`，状态：`{report['execution']['status']}`，退出码：`{report['execution']['exit_code']}`",
        f"- 配置最大并发：`{report['execution']['configured_max_workers']}`；并发上界：`{report['execution']['observed_concurrency_bound']}`",
        f"- 用户数：`{report['execution']['stats'].get('users')}`",
        "",
        "| 阶段 | 产物版本 | 已完成用户数 | 是否完成 | 游标 |",
        "|---|---:|---:|---|---|",
    ]
    for stage, detail in report["snapshot"]["stages"].items():
        lines.append(
            f"| {stage} | {detail['artifact_version']} | {len(detail['completed_users'])} | {detail['complete']} | {detail.get('cursor')} |"
        )
    lines.extend(
        [
            "",
            "## 输入校验与失败清单",
            "",
            f"- 非法记录数：`{validation['count']}`；业务码分布：`{json.dumps(validation['code_counts'], ensure_ascii=False)}`",
            f"- 业务码目录：`{', '.join(report['business_error_codes'])}`",
            "",
            "| 业务码 | 来源 | 行 | 字段 | 原因 |",
            "|---|---|---:|---|---|",
        ]
    )
    if validation["records"]:
        for failure in validation["records"]:
            lines.append(
                f"| {failure['code']} | {failure['source']} | {failure['line']} | {failure['field']} | {failure['message']} |"
            )
    else:
        lines.append("| - | - | - | - | 无失败记录 |")
    lines.extend(
        [
            "",
            "## 三路召回",
            "",
            f"- 用户数：`{recall['users']}`；冷启动用户：`{recall['cold_start_users']}`；零候选权重重分配用户：`{recall['zero_candidate_users']}`",
            "- 通道候选均值：" + "；".join(
                f"`{name}`={_fmt(value, 2)}" for name, value in recall["mean_candidates_per_channel"].items()
            ),
            "",
            "## 逐级误差传播",
            "",
            f"- 排序位置变化用户数：`{error_summary['users_with_rank_changes']}`；位置变化数：`{error_summary['rank_changes']}`",
            f"- 排序变化归因：`{json.dumps(error_summary['rank_change_causes'], ensure_ascii=False)}`",
            "",
            "| 环节 | 最大差值 | 超过容差候选数 | 首次超差计数 |",
            "|---|---:|---:|---:|",
        ]
    )
    for stage in STAGE_ORDER:
        lines.append(
            f"| {stage} | {_fmt(error_summary['stage_max_difference'].get(stage))} | {error_summary['stage_exceeding_candidate_counts'].get(stage, 0)} | {error_summary['first_exceeding_stage_counts'].get(stage, 0)} |"
        )
    rank_changes = [
        {"user_id": user["user_id"], **change}
        for user in report["error_propagation"]["users"]
        for change in user["rank_changes"]
    ]
    lines.extend(["", "### 位宽切换导致的排序变化", "", "| 用户 | 物品 | 主位宽名次 | 参考位宽名次 | 归因 |", "|---|---|---:|---:|---|"])
    if rank_changes:
        for change in rank_changes[:40]:
            lines.append(
                f"| {change['user_id']} | {change['item_id']} | {change['primary_rank']} | {change['reference_rank']} | {change['cause']} |"
            )
    else:
        lines.append("| - | - | - | - | 本样本未发生排序翻转 |")
    boundary = report["error_propagation"]["boundary_sensitive_candidates"]
    lines.extend(["", "### 边界敏感候选", "", "| 用户 | 物品 | 位宽角色 | 位宽 | 名次 | 相邻分差 | 原因 |", "|---|---|---|---:|---:|---:|---|"])
    if boundary:
        for entry in boundary[:50]:
            lines.append(
                f"| {entry['user_id']} | {entry['item_id']} | {entry['precision_role']} | {entry['bits']} | {entry['rank']} | {_fmt(entry['next_rank_margin'])} | {', '.join(entry['reasons'])} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | 未发现边界敏感候选 |")
    lines.extend(
        [
            "",
            "## 真值评估",
            "",
            "| 位宽 | 用户数 | 命中率 | Precision@K | 平均位置偏差 | 平均 Top-K 重合 | 未命中真值候选数 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {primary_eval.get('bits')} | {primary_eval.get('users_evaluated')} | {_fmt(primary_eval.get('mean_hit_rate'))} | {_fmt(primary_eval.get('mean_precision_at_k'))} | {_fmt(primary_eval.get('mean_position_deviation'))} | {_fmt(primary_eval.get('mean_top_k_overlap'))} | {primary_eval.get('missed_truth_candidate_count')} |",
            f"| {reference_eval.get('bits')} | {reference_eval.get('users_evaluated')} | {_fmt(reference_eval.get('mean_hit_rate'))} | {_fmt(reference_eval.get('mean_precision_at_k'))} | {_fmt(reference_eval.get('mean_position_deviation'))} | {_fmt(reference_eval.get('mean_top_k_overlap'))} | {reference_eval.get('missed_truth_candidate_count')} |",
            "",
            f"两档位评估是否真实不同：`{report['evaluation']['precision_results_differ']}`（评估使用各自位宽的实际排序结果）。",
            "",
            "## 产物",
            "",
            f"- 清单：`{report['artifacts']['manifest']}`",
            f"- 召回快照：`{report['artifacts']['recall']}`",
            f"- 融合快照：`{report['artifacts']['fusion']}`",
            f"- 评估快照：`{report['artifacts']['evaluation']}`",
            "- 完整报告 JSON 内含每用户逐候选误差、通道贡献和未命中归因。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report: dict[str, Any], report_dir: Path, *, temp_parent: Path | None = None) -> dict[str, str]:
    report_dir = report_dir.resolve()
    json_path = report_dir / "experiment_report.json"
    markdown_path = report_dir / "experiment_report.md"
    failures_path = report_dir / "failures.json"
    atomic_write_json(json_path, report, temp_parent=temp_parent)
    atomic_write_text(markdown_path, render_markdown(report), temp_parent=temp_parent)
    atomic_write_json(
        failures_path,
        {
            "code_counts": report["validation"]["code_counts"],
            "failures": report["validation"]["records"],
        },
        temp_parent=temp_parent,
    )
    return {"json": str(json_path), "markdown": str(markdown_path), "failures": str(failures_path)}
