"""可重复演示：三个场景顺序执行，输出稳定顺序的总结。

场景 1：首次全量扫描（宽松预算，全列投影）；
场景 2：预算收紧后的重调度（版本递增、旧轨迹标记陈旧、整体替换）；
场景 3：快照恢复重放（整体替换恢复 + 只读备份）；
附加：三类异常注入业务码 E_PAYLOAD / E_ENCODING / E_BUDGET。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from colbench.errors import BenchError
from colbench.sample_data import generate
from colbench.versioning import BudgetParams
from colbench.workbench import Workbench

ROOT = Path(__file__).parent
WORKSPACE = ROOT / "data" / "demo"


def reset_workspace() -> Workbench:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    spec_paths = generate(WORKSPACE / "samples")
    bench = Workbench(WORKSPACE)
    bench.build_from_specs(spec_paths)
    return bench


def load_query(name: str) -> str:
    return json.loads((WORKSPACE / "samples" / name)
                      .read_text(encoding="utf-8"))["sql"]


def main() -> int:
    summary: list[dict] = []

    # ---- 场景 1：首次全量扫描 ------------------------------------------
    bench = reset_workspace()
    full_query = load_query("q_full.json")
    loose = BudgetParams(global_parallelism=4, per_column_memory=8192,
                         per_segment_concurrency=2)
    model_one = bench.run_scan(full_query, loose, business_codes=["OK"])
    summary.append({
        "scene": "1 首次全量扫描",
        "version": model_one.version.to_dict(),
        "tasks": len(model_one.schedule.tasks),
        "max_running": model_one.schedule.max_running_observed,
        "statistics": model_one.schedule.statistics(),
        "verify": model_one.verify_report["status"],
    })

    # ---- 场景 2：预算收紧后的重调度 -------------------------------------
    tight = BudgetParams(global_parallelism=1, per_column_memory=1024,
                         per_segment_concurrency=1)
    range_query = load_query("q_range.json")
    model_two = bench.run_scan(range_query, tight, business_codes=["OK"])
    report = model_two.replacement.to_dict()
    summary.append({
        "scene": "2 预算收紧重调度",
        "version": model_two.version.to_dict(),
        "tasks": len(model_two.schedule.tasks),
        "max_running": model_two.schedule.max_running_observed,
        "cleared_trajectories": report["cleared_trajectory_count"],
        "affected_blocks": len(report["affected_block_keys"]),
        "stale": report["stale_count"],
        "candidate_blocks": len(model_two.plan.candidate_blocks),
        "skipped_blocks": len(model_two.plan.skipped_blocks),
        "verify": model_two.verify_report["status"],
    })

    # 括号/空白等价性
    paren_query = load_query("q_paren.json")
    paren_plan = bench.plan(paren_query)
    assert paren_plan.plan_digest == bench.plan(range_query).plan_digest
    summary.append({
        "scene": "2b 括号空白等价性",
        "plan_digest": paren_plan.plan_digest,
        "equal_digest": True,
    })

    # ---- 场景 3：快照恢复重放 -------------------------------------------
    restore_report = bench.restore(tight)
    reopen = Workbench(WORKSPACE)
    reopen.reopen_segments()
    model_three = reopen.run_scan(range_query, tight,
                                  business_codes=["OK"])
    summary.append({
        "scene": "3 快照恢复重放",
        "restore_action": restore_report["action"],
        "backup": Path(restore_report.get("backup", "")).name
        if restore_report.get("backup") else None,
        "version": model_three.version.to_dict(),
        "tasks": len(model_three.schedule.tasks),
        "verify": model_three.verify_report["status"],
    })

    # ---- 附加：异常注入业务码 -------------------------------------------
    fault_codes = _fault_injection_demo()
    summary.append({"scene": "4 异常注入业务码", "codes": fault_codes})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n演示视图（场景 2 稳定渲染）：")
    print(model_two.render_text())
    return 0


def _fault_injection_demo() -> list[dict]:
    results = []
    query = load_query("q_eq_in.json")

    # E_PAYLOAD：载荷截断（注入后重新从磁盘打开段以读取损坏文件）
    bench = Workbench(WORKSPACE)
    bench.reopen_segments()
    bench.inject_fault("truncate", segment_id="seg_a", row_group=0,
                       column="region")
    try:
        bench.reopen_segments()
        bench.run_scan(query, BudgetParams(2, 8192, 2))
    except BenchError as exc:
        results.append({"injection": "truncate", "code": exc.code})

    # E_ENCODING：编码与载荷不匹配（重建干净段后注入）
    bench = reset_workspace()
    bench.inject_fault("encoding_mismatch", segment_id="seg_a",
                       row_group=0, column="region")
    try:
        bench.reopen_segments()
        bench.run_scan(query, BudgetParams(2, 8192, 2))
    except BenchError as exc:
        results.append({"injection": "encoding_mismatch", "code": exc.code})

    # E_BUDGET：解码内存预算为零
    bench = reset_workspace()
    try:
        bench.run_scan(query, BudgetParams(2, 0, 2))
    except BenchError as exc:
        results.append({"injection": "zero_memory", "code": exc.code})
    return results


if __name__ == "__main__":
    raise SystemExit(main())
