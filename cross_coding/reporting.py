from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .store import atomic_write_json


def write_report(report: dict[str, Any], json_path: str | Path) -> Path:
    path = Path(json_path)
    atomic_write_json(path, report)
    markdown_path = path.with_suffix(".md")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return path


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# 类别交叉编码复核报告", ""]
    lines.append(f"- 状态：`{report.get('status', 'unknown')}`")
    if report.get("config_fingerprint"):
        lines.append(f"- 配置/表指纹：`{report['config_fingerprint']}`")
    if report.get("width_profile"):
        lines.append(f"- 位宽档位：`{json.dumps(report['width_profile'], ensure_ascii=False)}`")
    if report.get("manifest"):
        manifest = report["manifest"]
        lines.append(
            f"- 快照版本：global_version=`{manifest.get('global_version')}`, "
            f"command_cursor=`{manifest.get('completed_command_cursor')}`"
        )
    counts = report.get("business_code_counts", {})
    lines.append(f"- 业务码计数：`{json.dumps(counts, ensure_ascii=False, sort_keys=True)}`")
    lines.append("")

    failures = report.get("failures") or report.get("failure_list") or []
    if failures:
        lines.append("## 失败清单")
        lines.append("")
        for item in failures:
            lines.append(
                f"- `{item.get('code')}` / `{item.get('detail_code')}` "
                f"{item.get('path')}: {item.get('message')}"
            )
        lines.append("")

    lines.append("## 逐组合裁决")
    lines.append("")
    lines.append("| # | 组合 | 阶数 | 唯一码 | 碰撞 | 点预算 | 裁决 | 指纹 |")
    lines.append("|---:|---|---:|---:|---:|---|---|---|")
    for combo in report.get("combinations", []):
        encoding = combo.get("encoding") or {}
        decision = combo.get("budget_decision") or {}
        lines.append(
            "| {rank} | `{key}` | {order} | {unique} | {collision} | `{point}` | `{verdict}` | `{fp}` |".format(
                rank=combo.get("rank"),
                key=combo.get("canonical_key"),
                order=combo.get("order"),
                unique=encoding.get("unique_count", "-"),
                collision=encoding.get("collision_count", "-"),
                point=combo.get("point_status"),
                verdict=decision.get("combined_verdict") or decision.get("verdict", "-"),
                fp=str(combo.get("combo_fingerprint", ""))[:24],
            )
        )
    lines.append("")

    pending = report.get("pending_queue", [])
    if pending:
        lines.append("## 待办队列")
        lines.append("")
        for item in pending:
            lines.append(
                f"{item.get('rank')}. `{item.get('canonical_key')}` -> `{item.get('code')}`"
            )
        lines.append("")

    reuse = report.get("reuse_summary")
    if reuse:
        lines.append("## 失效与复用统计")
        lines.append("")
        lines.append(f"- 总计：`{json.dumps(reuse.get('totals', {}), ensure_ascii=False, sort_keys=True)}`")
        for stage, result_counts in reuse.get("by_stage", {}).items():
            lines.append(f"- `{stage}`: `{json.dumps(result_counts, ensure_ascii=False, sort_keys=True)}`")
        lines.append("")

    flips = []
    for combo in report.get("combinations", []):
        attribution = combo.get("flip_attribution") or {}
        if attribution.get("changed_scenario_count"):
            flips.append((combo["canonical_key"], attribution.get("source_counts", {})))
    if flips:
        lines.append("## 翻转归因")
        lines.append("")
        for key, sources in flips:
            lines.append(f"- `{key}`: `{json.dumps(sources, ensure_ascii=False, sort_keys=True)}`")
        lines.append("")

    return "\n".join(lines)


def write_generated_table(table_payload: dict[str, Any], json_path: str | Path) -> None:
    atomic_write_json(Path(json_path), table_payload)
    csv_path = Path(json_path).with_suffix(".csv")
    import csv

    columns = table_payload["order"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row_index in range(table_payload["row_count"]):
            writer.writerow(
                [
                    "" if table_payload["columns"][name]["values"][row_index] is None
                    else table_payload["columns"][name]["values"][row_index]
                    for name in columns
                ]
            )
