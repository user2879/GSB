"""可重复演示：生成确定性样本工作区，跑增量命令流与全量重建并逐字段比对。

用法::

    python demo.py [--workspace demo_workspace] [--samples 10] [--keep]

退出码 0 表示全部验收断言通过。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

from xhash_ledger.engine import Engine
from xhash_ledger.generator import generate_workspace
from xhash_ledger.state import write_json_atomic

COMPARE_FIELDS = ("signatures", "cluster_ledger")


def _final_ids(workspace: str, commands: list[dict]) -> list[str]:
    active: set[str] = set()
    for command in commands:
        cmd, sid = command.get("cmd"), command.get("sample_id")
        if cmd in ("add", "replace"):
            active.add(sid)
        elif cmd == "delete":
            active.discard(sid)
    sample_dir = os.path.join(workspace, "samples")
    return sorted(sid for sid in active if os.path.exists(os.path.join(sample_dir, f"{sid}.png")))


def compare_reports(incremental: dict, rebuild: dict) -> dict:
    result = {"field_equal": {}, "cache_key_equal": False, "config_hash_equal": False, "all_equal": True}
    for field in COMPARE_FIELDS:
        equal = incremental[field] == rebuild[field]
        result["field_equal"][field] = equal
        result["all_equal"] &= equal
    inc_keys = {sid: doc["cache_key"] for sid, doc in incremental["signatures"].items()}
    reb_keys = {sid: doc["cache_key"] for sid, doc in rebuild["signatures"].items()}
    result["cache_key_equal"] = inc_keys == reb_keys
    result["all_equal"] &= result["cache_key_equal"]
    result["config_hash_equal"] = (
        incremental["manifest"]["config_hash"] == rebuild["manifest"]["config_hash"]
    )
    result["all_equal"] &= result["config_hash_equal"]
    return result


def run_demo(workspace: str, samples: int, keep: bool) -> int:
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    info = generate_workspace(workspace, sample_count=samples)
    commands = info["commands"]

    engine = Engine(workspace, rebuild=False)
    can_resume, reason = engine.store.can_resume(engine.config.config_hash())
    print(f"[1] 恢复判定: {reason}")
    reports = engine.execute_stream(commands)
    write_json_atomic(os.path.join(workspace, "reports_incremental.json"), reports)
    incremental_final = reports[-1]
    print(f"[2] 增量运行: {len(reports)} 条命令，最终样本 {len(incremental_final['manifest']['completed_samples'])} 个")

    rebuild_engine = Engine(workspace, rebuild=True)
    rebuild_engine.apply_final_config(commands)
    rebuild_report = rebuild_engine.execute_rebuild(_final_ids(workspace, commands))
    write_json_atomic(os.path.join(workspace, "reports_rebuild.json"), rebuild_report)
    print(f"[3] 全量重建: 从空白一次性计算 {len(rebuild_report['manifest']['completed_samples'])} 个样本")

    comparison = compare_reports(incremental_final, rebuild_report)
    write_json_atomic(os.path.join(workspace, "comparison.json"), comparison)
    print("[4] 逐字段比对:")
    for field, equal in comparison["field_equal"].items():
        print(f"    - {field}: {'一致' if equal else '不一致'}")
    print(f"    - 缓存键集合: {'一致' if comparison['cache_key_equal'] else '不一致'}")
    print(f"    - 配置哈希: {'一致' if comparison['config_hash_equal'] else '不一致'}")

    reused = sum(1 for entry in incremental_final["reuse_log"] if entry["status"] == "reused")
    print(f"[5] 末轮复用清单: {len(incremental_final['reuse_log'])} 条，其中命中复用 {reused} 条")
    print(f"[6] 簇账本: {len(incremental_final['cluster_ledger'])} 个簇，"
          f"重指定记录 {len(incremental_final['cluster_reassignments'])} 条")
    if not keep:
        print(f"(演示工作区保留在 {workspace}，可手动查阅报告)")
    print("结果: " + ("全部验收断言通过 ✔" if comparison["all_equal"] else "存在不一致 �’"))
    return 0 if comparison["all_equal"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跨模态哈希账本可重复演示")
    parser.add_argument("--workspace", default="demo_workspace")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--keep", action="store_true", help="保留工作区（默认即保留）")
    args = parser.parse_args(argv)
    return run_demo(args.workspace, args.samples, args.keep)


if __name__ == "__main__":
    sys.exit(main())
