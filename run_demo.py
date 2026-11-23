"""端到端演示：多格式回执归一、热更新版本隔离、灰度归并、快照恢复重放。

用法: python run_demo.py [样本目录] [账本路径]
"""

import os
import sys

from gsb.ledger import Ledger
from gsb.pipeline import (
    SimulatedCrash,
    recover_session,
    resume_session,
    run_session,
)
from gsb.spec import input_content_digest

SAMPLE_DIR = sys.argv[1] if len(sys.argv) > 1 else "samples"
DB_PATH = sys.argv[2] if len(sys.argv) > 2 else "gsb_ledger.db"
HOT_UPDATES = {2: [("b-gray-a", 777)]}  # 第 3 批前整体替换灰度后端参数


def show_ledger(title):
    ledger = Ledger.open(DB_PATH, input_content_digest(SAMPLE_DIR))
    print(f"\n== {title} ==")
    print("账本元信息:", ledger.meta())
    print("灰度去向分布:", ledger.route_distribution())
    print("参数版本隔离 (批次, 版本, 请求数):")
    for row in ledger.param_version_isolation():
        print(f"  {row['batch_id']}  v{row['param_version']}  x{row['n']}")
    for batch_id in ledger.committed_batches():
        conclusion = ledger.conclusion_for(batch_id)
        rejected = len(conclusion["rejected"])
        print(
            f"  {batch_id}: 采纳 {conclusion['adopted_format']} "
            f"(参数v{conclusion['param_version']}) 来源格式 {conclusion['source_formats']} "
            f"否决 {rejected} 条"
        )
    waits = ledger.all_waits()
    total_wait = sum(w["wait_end_ms"] - w["wait_start_ms"] for w in waits)
    print(f"等待观测: {len(waits)} 条, 累计 {total_wait} ms")
    ledger.close()


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print("== 阶段 1: 正常会话（含热更新） ==")
    report = run_session(SAMPLE_DIR, DB_PATH, hot_updates=HOT_UPDATES)
    print(f"完成批次: {[r.batch_id for r in report.completed]}")
    show_ledger("正常会话账本")

    print("\n== 阶段 2: 恢复重放校验 ==")
    recovery = recover_session(SAMPLE_DIR, DB_PATH)
    print(f"重放校验通过批次: {recovery.replay_verified}, 未知批次: {recovery.unknown_batches}")

    print("\n== 阶段 3: 模拟强制终止（batch-3 提交前崩溃） ==")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    try:
        run_session(
            SAMPLE_DIR, DB_PATH, hot_updates=HOT_UPDATES, crash_before_commit={2}
        )
    except SimulatedCrash as exc:
        print(f"崩溃: {exc}")
    recovery = recover_session(SAMPLE_DIR, DB_PATH)
    print(
        f"已提交可恢复批次: {recovery.replay_verified}, "
        f"未知需重跑: {recovery.unknown_batches}, "
        f"可生成退出报告: {recovery.exit_report_available}"
    )

    print("\n== 阶段 4: 重跑未知批次并再次重放 ==")
    resumed = resume_session(SAMPLE_DIR, DB_PATH, hot_updates=HOT_UPDATES)
    print(f"重跑批次: {[r.batch_id for r in resumed.completed]}")
    recovery = recover_session(SAMPLE_DIR, DB_PATH)
    print(
        f"最终重放校验: {recovery.replay_verified} 批, "
        f"未知批次: {recovery.unknown_batches}, "
        f"可生成退出报告: {recovery.exit_report_available}"
    )
    show_ledger("最终账本")


if __name__ == "__main__":
    main()
