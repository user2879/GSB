"""诊断报告：汇总证据链校准、失效边界、预算、并发与快照诊断信息。"""
from __future__ import annotations

import json
from pathlib import Path

from .pipeline import PipelineResult


def build_report(result: PipelineResult, config, sample_dir: str | Path,
                 scenario: str) -> dict:
    failures = result.errors.sorted_dict_list()
    failure_codes: dict[str, int] = {}
    for item in failures:
        failure_codes[item["code"]] = failure_codes.get(item["code"], 0) + 1

    emotion_trace = [
        {"round": rnd,
         "emotion": result.calibrations[rnd].emotion,
         "calib_q": result.calibrations[rnd].calib_q,
         "link": result.calibrations[rnd].link,
         "link_weight_q": result.calibrations[rnd].link_weight_q}
        for rnd in sorted(result.calibrations)
    ]
    speaker_trace = [
        {"round": rnd,
         "speaker": result.attributions[rnd].speaker,
         "segment_id": result.attributions[rnd].segment_id,
         "break_reason": result.attributions[rnd].break_reason,
         "calib_round": result.attributions[rnd].calib_round,
         "evidence_component": result.attributions[rnd].evidence_component}
        for rnd in sorted(result.attributions)
    ]

    total_consumed = sum(b["consumed"] for b in result.batches)
    total_budget = sum(b["budget"] for b in result.batches)
    report = {
        "scenario": scenario,
        "sample_dir": str(sample_dir),
        "fixed_point": {
            "frac_bits": config.frac_bits,
            "quant_gear": config.quant_gear,
            "gain_q": config.gain_q,
            "decay_q": config.decay_q,
        },
        "rounds_total": len(result.valid_rounds),
        "rounds_completed": len(result.calibrations),
        "emotion_trace": emotion_trace,
        "speaker_trace": speaker_trace,
        "trajectories": [t.to_dict() for t in result.trajectories],
        "incremental": {
            "reuse_hits": result.reuse_hits,
            "recompute_rounds": result.recompute_rounds,
            "pending_rounds": result.pending_rounds,
            "decisions": result.ledger.decisions,
        },
        "budget": {
            "batches": result.batches,
            "total_consumed": total_consumed,
            "total_budget": total_budget,
        },
        "snapshot": result.snapshot_manifest,
        "snapshot_restored": result.snapshot_restored,
        "injections_hit": result.injections_hit,
        "failure_summary": failure_codes,
        "failures": failures,
    }
    return report


def write_report(report: dict, output_dir: str | Path,
                 name: str = "diagnostic_report.json") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(json.dumps(report, sort_keys=True, indent=2,
                               ensure_ascii=False), encoding="utf-8")
    return path


def render_markdown(report: dict) -> str:
    lines = [
        f"# 诊断报告：{report['scenario']}",
        "",
        f"- 轮次总数/完成：{report['rounds_total']} / {report['rounds_completed']}",
        f"- 定点位宽/档位：{report['fixed_point']['frac_bits']} / "
        f"{report['fixed_point']['quant_gear']}",
        f"- 复用命中：{report['incremental']['reuse_hits']}，"
        f"重算轮次：{report['incremental']['recompute_rounds']}",
        f"- 预算待办：{report['incremental']['pending_rounds']}",
        f"- 快照恢复：{report['snapshot_restored']}",
        f"- 注入命中：{[i['injection_id'] for i in report['injections_hit']]}",
        f"- 失败码统计：{report['failure_summary']}",
        "",
        "## 情感与说话人轨迹",
        "",
        "| 轮次 | 情感 | 说话人 | 轨迹段 | 断开原因 |",
        "| --- | --- | --- | --- | --- |",
    ]
    emotions = {row["round"]: row for row in report["emotion_trace"]}
    for row in report["speaker_trace"]:
        emo = emotions.get(row["round"], {}).get("emotion", "-")
        lines.append(
            f"| {row['round']} | {emo} | {row['speaker']} | "
            f"{row['segment_id'] or '-'} | {row['break_reason'] or '-'} |")
    lines.extend(["", "## 批次预算", ""])
    for batch in report["budget"]["batches"]:
        lines.append(
            f"- 批次 {batch['batch_no']}：消耗 {batch['consumed']}/"
            f"{batch['budget']}，复用 {batch['reused_hits']}，"
            f"待办 {batch['pending_rounds']}")
    lines.extend(["", "## 失效/复用判定（前 20 条）", ""])
    for decision in report["incremental"]["decisions"][:20]:
        lines.append(
            f"- R{decision['round']} {decision['stage']}: "
            f"{decision['action']} ({decision['reason']})")
    if report["failures"]:
        lines.extend(["", "## 失败清单", ""])
        for failure in report["failures"]:
            lines.append(
                f"- {failure['code']} @ R{failure['round']} "
                f"{failure['stage']}: {failure['message']}"
                + (f" [{failure['injection_id']}]"
                   if failure.get("injection_id") else ""))
    return "\n".join(lines) + "\n"
