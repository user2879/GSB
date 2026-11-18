"""Repeatable three-round demonstration with stable-ordered output.

Round 1: cold start with the default pool parameters.
Round 2: adjust the quotas / acquisition timeout (new parameter version) and rerun.
Round 3: exercise the three startup-parameter fault injections and snapshot
         version handling, printing business codes in a fixed order.

Run:  python -m pool_workbench.demo
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .app import WorkbenchSession
from .errors import BusinessError
from .fault_injection import (
    inject_config_quota_exceed,
    inject_schema_missing_table,
    inject_snapshot_missing_version,
)
from .sample_db import INVENTORY_TABLE, build_sample_db

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = Path(__file__).resolve().parent / "samples"


def _line(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def _summarize(result) -> None:
    print("结局计数（固定顺序 成功/获取超时/持有超时/执行错误）:")
    for key in ("SUCCESS", "GET_TIMEOUT", "HOLD_TIMEOUT", "EXEC_ERROR"):
        print(f"  {key:<14} {result.outcome_counts[key]}")
    print("逐条轨迹（任务, 结局, 连接序号, 等待ms）:")
    for trace in result.traces:
        print(f"  {trace.task_id}  {trace.outcome:<12} slot={trace.slot_seq}  wait_ms={trace.wait_ms}")
    print("约束违反:", result.constraint_violations or "无")
    print("失效重建次数:", result.rebuild_count)


def main() -> int:
    workspace = PROJECT_ROOT / "workspace"
    # Cold start guarantee: clear snapshots and rebuild the sample library.
    shutil.rmtree(workspace, ignore_errors=True)
    db_path = workspace / "sample.sqlite"
    build_sample_db(db_path)
    print(f"工作区: {workspace}")
    print(f"样本库: {db_path}")

    # ----- Round 1: cold start ----------------------------------------------
    _line("第 1 轮：冷启动（默认参数，param_version=1）")
    cold = WorkbenchSession(workspace)
    cold.load(SAMPLES / "tasks_cold.json", db_path)
    result_cold = cold.execute()
    _summarize(result_cold)
    snapshot_path = cold.save_snapshot()
    print("已保存快照:", snapshot_path.name)
    replay_report = cold.replay(cold.store.load(snapshot_path))
    print("重放校验:", replay_report["status"],
          "| 比对字段", replay_report["compared_fields"],
          "| 样本表未改动:", replay_report["sample_tables_unchanged"])
    cold.close()

    # ----- Round 2: adjusted quotas, new parameter version ------------------
    _line("第 2 轮：调整配额后重跑（param_version=2）")
    adjusted = WorkbenchSession(workspace)
    adjusted.load(SAMPLES / "tasks_adjusted.json", db_path)
    result_adjusted = adjusted.execute()
    _summarize(result_adjusted)
    restored = adjusted.restore_snapshot(snapshot_path)
    print("快照恢复采用整体替换:", restored["replaced"])
    print("变化参数项（稳定顺序）:")
    for change in restored["diff"]["changed_params"]:
        print(f"  {change['param']}: 快照={change['snapshot_value']} -> 当前={change['current_value']}")
    print("受影响排队任务:", restored["diff"]["affected_queued_tasks"])
    try:
        adjusted.replay(restored["document"])
    except BusinessError as exc:
        print("版本漂移重放报告:", exc.code,
              "| 首个任务:", exc.details.get("task_id"),
              "| 字段:", exc.details.get("field"),
              "| 期望:", exc.details.get("expected"),
              "| 实际:", exc.details.get("actual"))
    adjusted.close()

    # ----- Round 3: fault injections ----------------------------------------
    _line("第 3 轮：异常注入（仅改输入文件/启动参数）")
    injections: list[tuple[str, str]] = []

    build_sample_db(db_path)
    schema_session = WorkbenchSession(workspace)
    inject_schema_missing_table(db_path, INVENTORY_TABLE)
    try:
        schema_session.load(SAMPLES / "tasks_cold.json", db_path)
    except BusinessError as exc:
        injections.append(("schema_missing_table", exc.code))

    build_sample_db(db_path)
    config_session = WorkbenchSession(workspace)
    injected_config = inject_config_quota_exceed(SAMPLES / "tasks_cold.json")
    try:
        config_session.load(injected_config, db_path)
    except BusinessError as exc:
        injections.append(("config_quota_exceed", exc.code))

    snapshot_session = WorkbenchSession(workspace)
    injected_snapshot = inject_snapshot_missing_version(snapshot_path)
    try:
        snapshot_session.store.load(injected_snapshot)
    except BusinessError as exc:
        injections.append(("snapshot_missing_version", exc.code))

    for name, code in injections:
        print(f"  {name:<28} -> {code}")
    expected = ["E_SCHEMA", "E_CONFIG", "E_SNAPSHOT"]
    assert [code for _, code in injections] == expected, injections

    summary = {
        "round1_outcome_counts": result_cold.outcome_counts,
        "round2_outcome_counts": result_adjusted.outcome_counts,
        "injection_codes": expected,
        "snapshot": snapshot_path.name,
    }
    (workspace / "demo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("\n演示总结已写入:", workspace / "demo_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
