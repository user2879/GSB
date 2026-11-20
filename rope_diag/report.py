"""Diagnostic report assembly (JSON + Markdown)."""
from __future__ import annotations

import json
from pathlib import Path


def canonical(result: dict, sections: tuple[str, ...]) -> str:
    """Canonical JSON of selected result sections for equality checks."""
    return json.dumps({k: result[k] for k in sections}, sort_keys=True, default=str)


COMPARE_SECTIONS = ("stratified", "traces", "failures", "budget")


def trace_summary(traces_by_scheme: dict) -> dict:
    summary = {}
    for scheme, traces in traces_by_scheme.items():
        counts = {"position_vector": 0, "score": 0, "weight": 0, "none": 0}
        n_keys = 0
        for trace in traces.values():
            n_keys += trace["n_extrapolation_keys"]
            for stage, n in trace["stage_exceed_counts"].items():
                counts[stage] += n
            passed = trace["n_extrapolation_keys"] - sum(trace["stage_exceed_counts"].values())
            counts["none"] += passed
        summary[scheme] = {"n_extrapolation_keys": n_keys, "first_exceed_counts": counts}
    return summary


def build_report(
    final_run: dict,
    sensitivity: dict,
    checks: dict,
    decision_record: dict,
) -> dict:
    return {
        "title": "RoPE extrapolation diagnostic report",
        "manifest": {
            "global_version": final_run["manifest"]["global_version"],
            "config_hash": final_run["manifest"]["config_hash"],
            "injection_hash": final_run["manifest"]["injection_hash"],
            "pruning_fingerprint": final_run["manifest"]["pruning_fingerprint"],
            "completed": final_run["manifest"]["completed"],
        },
        "stratified_scores": final_run["stratified"],
        "error_trace_summary": trace_summary(final_run["traces"]),
        "error_traces": final_run["traces"],
        "sensitivity": sensitivity,
        "pruning_decision": decision_record,
        "budget": final_run["budget"],
        "failures": final_run["failures"],
        "invalidation": {
            "reused_count": len(final_run["reused"]),
            "recomputed_count": len(final_run["recomputed"]),
            "invalidated_key_count": len(final_run["invalidated_keys"]),
        },
        "consistency_checks": checks,
    }


def write_reports(report: dict, report_dir: str | Path) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path = report_dir / "report.md"
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, md_path


def to_markdown(report: dict) -> str:
    lines = [
        "# RoPE 外推诊断报告",
        "",
        f"- 全局版本: {report['manifest']['global_version']}",
        f"- 配置哈希: `{report['manifest']['config_hash']}`",
        f"- 注入清单哈希: `{report['manifest']['injection_hash']}`",
        f"- 剪枝口径指纹: `{report['manifest']['pruning_fingerprint']}`",
        "",
        "## 一致性校验",
    ]
    for name, ok in report["consistency_checks"].items():
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
    lines += ["", "## 误差追踪汇总（首次超差环节计数）"]
    for scheme, summ in report["error_trace_summary"].items():
        lines.append(
            f"- {scheme}: 外推键 {summ['n_extrapolation_keys']} 个, "
            f"首次超差分布 {summ['first_exceed_counts']}"
        )
    lines += ["", "## 分层打分（距离档位数）"]
    for scheme, strat in report["stratified_scores"].items():
        lines.append(
            f"- {scheme}: in_train 距离档 {len(strat['in_train'])} 个, "
            f"extrapolation 距离档 {len(strat['extrapolation'])} 个"
        )
    lines += ["", "## 敏感性不可用档位"]
    for scheme, diag in report["sensitivity"].items():
        for tier in diag["unusable_tiers"]:
            lines.append(
                f"- {scheme}: {tier['axis']} x{tier['factor']} -> {','.join(tier['reasons'])}"
            )
    lines += ["", "## 预算与失败"]
    lines.append(
        f"- 点数预算 {report['budget']['point_budget']}, "
        f"已消耗 {report['budget']['spent_points']}, "
        f"待算 {len(report['budget']['pending'])} 项 (E_POINT_BUDGET_PENDING)"
    )
    for failure in report["failures"]:
        lines.append(
            f"- 失败 {failure['scheme']}/{failure['sample_id']}/{failure['stage']}: "
            f"{failure['code']} (注入 {failure['injection_id']})"
        )
    lines += [
        "",
        "## 剪枝决策",
        f"- 选中策略: {report['pruning_decision']['selected_policy']}",
        f"- 依据: {report['pruning_decision']['rationale']}",
        "",
    ]
    return "\n".join(lines)
