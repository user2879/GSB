"""Summarize persisted reports into a reproducible validation artifact."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "artifacts" / "demo"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    report = _json(DEMO / "run" / "diagnostic_report.json")
    snapshot_meta = _json(DEMO / "snapshot" / "snapshot.json")
    injection_report = _json(DEMO / "injection_case" / "run" / "diagnostic_report.json")
    injection_violation = injection_report["metrics"]["post_compensation"]["first_tolerance_violation"]
    test_log_path = DEMO / "test_results.txt"
    test_log = test_log_path.read_text(encoding="utf-8") if test_log_path.is_file() else ""
    test_ok = "OK" in test_log.splitlines()[-1] if test_log.splitlines() else False
    test_count = test_log.count(" ... ok") if test_log else 0
    post = report["metrics"]["post_compensation"]
    pre = report["metrics"]["pre_compensation"]
    summary = {
        "format_version": 1,
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "numpy": __import__("numpy").__version__,
        },
        "cube": {
            "shape": report["cube"]["shape"],
            "content_hash": report["cube"]["content_hash"],
            "injections": report["cube"]["injections"],
        },
        "run": {
            "run_hash": report["run_hash"],
            "calibration": report["calibration"],
            "condition_number": report["unmix"]["condition_number"],
            "drift": {
                key: report["drift"][key]
                for key in ("max_estimate_residual_px", "rms_estimate_residual_px", "max_estimated_shift_px")
            },
            "pre_compensation_mae": pre["overall_absolute_error_mean"],
            "post_compensation_mae": post["overall_absolute_error_mean"],
            "compensation_evaluation": report["metrics"]["compensation_evaluation"],
            "post_fluorophores": post["per_fluorophore"],
            "first_tolerance_violation": post["first_tolerance_violation"],
            "closure": report["closure"],
            "warnings": report["warnings"],
            "stage_reuse": report["incremental"]["stage_reuse"],
        },
        "snapshot": {
            "version": snapshot_meta["version"],
            "content_hash": snapshot_meta["content_hash"],
            "source_run_hash": snapshot_meta["source_run_hash"],
        },
        "injection_case": {
            "run_hash": injection_report["run_hash"],
            "condition_number": injection_report["unmix"]["condition_number"],
            "first_tolerance_violation": injection_violation,
            "closure": injection_report["closure"],
        },
        "tests": {
            "passed": bool(test_ok),
            "count": int(test_count),
            "log": str(test_log_path.relative_to(ROOT)) if test_log_path.is_file() else None,
        },
        "artifact_bytes": _tree_size(DEMO),
    }
    (DEMO / "validation_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 实际验证报告",
        "",
        "本文件由 `scripts/make_validation_report.py` 从持久化诊断报告和测试日志生成。",
        "",
        "## 环境",
        "",
        f"- Python: `{summary['environment']['python']}`",
        f"- 平台: `{summary['environment']['platform']}`",
        f"- NumPy: `{summary['environment']['numpy']}`",
        "",
        "## 默认全量样本",
        "",
        f"- 形状: `{summary['cube']['shape']}`",
        f"- 立方体内容哈希: `{summary['cube']['content_hash']}`",
        f"- 运行哈希: `{summary['run']['run_hash']}`",
        f"- 端元条件数: `{summary['run']['condition_number']:.12g}`",
        f"- 量化误差上界最大值: `{summary['run']['calibration']['quantization_upper_bound_max_dn']:.9g} DN`",
        f"- 漂移残差最大值: `{summary['run']['drift']['max_estimate_residual_px']:.9g} px`",
        f"- 漂移残差 RMS: `{summary['run']['drift']['rms_estimate_residual_px']:.9g} px`",
        f"- 补偿前 MAE: `{summary['run']['pre_compensation_mae']:.9g}`",
        f"- 补偿后 MAE: `{summary['run']['post_compensation_mae']:.9g}`",
        f"- 改善量: `{summary['run']['compensation_evaluation']['absolute_error_reduction']:.9g}`",
        f"- 闭合校验: `{summary['run']['closure']['verified']}`",
        "",
        "### 逐荧光团后补偿指标",
        "",
        "| 荧光团 | MAE | 95 分位 | 超容差比例 | 平均区间 | 覆盖率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["run"]["post_fluorophores"]:
        lines.append(
            f"| {item['name']} | {item['absolute_error_mean']:.9g} | {item['absolute_error_p95']:.9g} | "
            f"{item['over_tolerance_fraction']:.6g} | {item['uncertainty_mean']:.9g} | "
            f"{item['interval_coverage_fraction']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## 快照",
            "",
            f"- 版本: `{summary['snapshot']['version']}`",
            f"- 内容哈希: `{summary['snapshot']['content_hash']}`",
            f"- 来源 run hash: `{summary['snapshot']['source_run_hash']}`",
            "",
            "## 浓度偏差注入案例",
            "",
            f"- 运行哈希: `{summary['injection_case']['run_hash']}`",
            f"- 条件数: `{summary['injection_case']['condition_number']:.12g}`",
            f"- 首越界带/像素: `{injection_violation['frame']}` / `{injection_violation['pixel']}`",
            f"- 荧光团 0 绝对误差: `{injection_violation['absolute_errors'][0]:.9g}`",
            "- 误差分量: "
            + ", ".join(
                f"`{key}={value:.9g}`"
                for key, value in injection_violation["fluorophores"][0]["components"].items()
            ),
            f"- 闭合校验: `{summary['injection_case']['closure']['verified']}`",
            "",
            "## 自动化测试",
            "",
            f"- 通过: `{summary['tests']['passed']}`",
            f"- 用例数: `{summary['tests']['count']}`",
            f"- 日志: `{summary['tests']['log']}`",
            "",
            "## 复算命令",
            "",
            "```powershell",
            "python -m spectral_workbench generate --out artifacts\\demo\\cube",
            "python -m spectral_workbench run --cube artifacts\\demo\\cube --out artifacts\\demo\\run --snapshot artifacts\\demo\\snapshot --version 1 --verify-closure",
            "python -m unittest discover -v",
            "python scripts\\make_validation_report.py",
            "```",
            "",
            f"当前 `artifacts/demo` 占用 `{summary['artifact_bytes']}` 字节。",
            "",
        ]
    )
    (DEMO / "VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"validation_summary": str(DEMO / "validation_summary.json"), "tests": summary["tests"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
