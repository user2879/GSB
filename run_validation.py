from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from dense_index.build import run_full_build
from dense_index.diagnostics import build_failure_manifest, run_fault_injections
from dense_index.replay import run_stream_equivalence
from dense_index.sample import generate_sample


def run_unittest_suite() -> dict:
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).parent / "tests"))
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "output_tail": stream.getvalue()[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="sample_workspace")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)

    generation = generate_sample(root)
    build = run_full_build(root)
    equivalence = run_stream_equivalence(root)
    faults = run_fault_injections(root)
    failure_manifest = build_failure_manifest(root)
    tests = {"successful": True, "skipped": True} if args.skip_tests else run_unittest_suite()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "generation": generation,
        "full_build": {"blocks": len(build["view"]["blocks"]), "buckets": len(build["buckets"])},
        "equivalence": {
            "all_fields_match": equivalence["all_fields_match"],
            "command_count": equivalence["command_count"],
            "budget_pending_seen": equivalence["budget_pending_seen"],
            "failed_steps": [step for step in equivalence["steps"] if not step["match"]],
        },
        "faults": {"all_hit": faults["all_hit"], "hit_count": faults["hit_count"], "total": faults["total"]},
        "failure_manifest": {"all_hit": failure_manifest["all_hit"], "cases": len(failure_manifest["cases"])},
        "tests": tests,
        "overall_success": equivalence["all_fields_match"] and faults["all_hit"] and failure_manifest["all_hit"] and tests["successful"],
    }
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")

    report = [
        "# 实际验证报告",
        "",
        f"- 生成时间（UTC）：{summary['generated_at_utc']}",
        f"- 平台：`{summary['platform']}`",
        f"- Python：`{summary['python']}`",
        f"- 总结果：{'通过' if summary['overall_success'] else '未通过'}",
        "",
        "## 样本生成",
        "",
        f"- 段落数：{generation['paragraph_count']}",
        f"- 初始块数：{generation['initial_block_count']}",
        f"- 命令数：{equivalence['command_count']}",
        "",
        "## 增量与全量等价",
        "",
        f"- 逐字段一致：`{equivalence['all_fields_match']}`",
        f"- 预算待办已命中：`{equivalence['budget_pending_seen']}`",
        "- 每条命令均比较块完整向量/定点向量、摘要、分桶成员、召回、账本与快照视图。",
        "",
        "## 故障注入",
        "",
        f"- 命中：{faults['hit_count']}/{faults['total']}；全部命中：`{faults['all_hit']}`",
        "- 业务码：`E_CORPUS_INVALID`、`E_CONFIG_INVALID`、`E_RECEIPT_INVALID`、`E_VECTOR_NONFINITE`、`E_SNAPSHOT_RECOVERY`、`E_BOUNDARY_FLIP`、`E_POINT_BUDGET_PENDING`。",
        "",
        "## 自动化测试",
        "",
    ]
    windows_summary_path = root / "outputs" / "windows_validation_summary.json"
    if windows_summary_path.exists():
        windows_summary = json.loads(windows_summary_path.read_text(encoding="utf-8"))
        report.extend(
            [
                "## Windows 复核",
                "",
                f"- 平台：`{windows_summary.get('platform')}`",
                f"- Python：`{windows_summary.get('python')}`",
                f"- 总结果：`{windows_summary.get('overall_success')}`；明细见 `docs/WINDOWS_VALIDATION_REPORT.md`。",
                "",
            ]
        )
    if args.skip_tests:
        report.append("- 已按参数跳过。")
    else:
        report.extend(
            [
                f"- 用例数：{tests['tests_run']}",
                f"- 失败：{tests['failures']}；错误：{tests['errors']}；跳过：{tests['skipped']}",
                f"- 结果：{'通过' if tests['successful'] else '未通过'}",
                "",
                "```text",
                tests["output_tail"],
                "```",
            ]
        )
    report_path = Path("docs") / "VALIDATION_REPORT.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"overall_success": summary["overall_success"], "summary": str(out_dir / "validation_summary.json"), "report": str(report_path)}, ensure_ascii=False, indent=2))
    return 0 if summary["overall_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
