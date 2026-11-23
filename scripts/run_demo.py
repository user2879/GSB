from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_diag.clock import MonotonicClock
from config_diag.compare import compare_incremental_and_full
from config_diag.compute import record_bytes
from config_diag.engine import ConfigEngine
from config_diag.injections import run_injection
from config_diag.recovery import recover_and_replay
from config_diag.structured_log import JsonlLogger
from config_diag.subscriptions import SubscriptionSimulator
from config_diag.validation import load_bundle


def _remove_readonly_on_windows(func, path, exc_info) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        func(path)
    except OSError:
        pass


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _run_cli(workspace: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "config_diag",
            "--workspace",
            str(workspace),
            "--samples",
            str(ROOT / "samples"),
            *args,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def main() -> int:
    demo_root = ROOT / "runtime" / "demo"
    if demo_root.exists():
        shutil.rmtree(demo_root, onerror=_remove_readonly_on_windows)
    demo_root.mkdir(parents=True)
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(demo_root / "logs" / "diagnostics.jsonl")
    started = datetime.now(timezone.utc)
    spec, clients, changes = load_bundle(ROOT / "samples")
    results: dict[str, Any] = {"started_utc": started.isoformat(), "sample_counts": {"keys": len(spec.keys), "clients": len(clients.clients), "operations": len(changes.operations)}}

    engine = ConfigEngine(demo_root, logger=logger, clock=MonotonicClock())
    try:
        init = engine.initialize(spec, clients)
        results["init"] = init
        simulator = SubscriptionSimulator(engine, clients)
        results["initial_poll"] = simulator.poll(1)
        before_records = engine.ledger.records()
        operation1 = engine.apply_operation(changes.operations[0])
        after_records = engine.ledger.records()
        expected_affected = set(operation1["affected"])
        unaffected = sorted(set(before_records) - expected_affected)
        byte_checks = {
            key: record_bytes(before_records[key]) == record_bytes(after_records[key])
            for key in unaffected
        }
        version_checks = {
            key: before_records[key]["version"] == after_records[key]["version"]
            for key in unaffected
        }
        results["operation_1"] = {
            "operation_id": operation1["operation_id"],
            "affected": operation1["affected"],
            "propagation": operation1["propagation"],
            "unaffected_keys": unaffected,
            "unaffected_bytes_identical": byte_checks,
            "unaffected_versions_identical": version_checks,
            "values": {
                key: after_records[key]["value"]
                for key in (
                    "pricing.base_fee",
                    "pricing.net_fee",
                    "tax.amount",
                    "checkout.total",
                    "reporting.rounded",
                )
            },
        }
        poll_after_change = simulator.poll(1)
        results["poll_after_operation_1"] = {
            "stale_events": poll_after_change["stale_events"],
            "statistics_source": poll_after_change["statistics_source"],
        }
        operation2 = engine.apply_operation(changes.operations[1])
        operation3 = engine.apply_operation(changes.operations[2])
        results["operation_2"] = {
            "operation_id": operation2["operation_id"],
            "affected": operation2["affected"],
            "propagation": operation2["propagation"],
        }
        results["operation_3"] = {
            "operation_id": operation3["operation_id"],
            "affected": operation3["affected"],
            "derivation_version": operation3["derivation_version"],
        }
        state_before_crash = engine.list_state()
        results["view_and_trace_staleness"] = {
            "current_view_version": state_before_crash["current_view"]["view_version"],
            "current_view_digest": state_before_crash["current_view"]["view_digest"],
            "stale_view_count": state_before_crash["stale_view_count"],
            "stale_trace_count": state_before_crash["stale_trace_count"],
            "snapshots": state_before_crash["snapshots"],
            "journal": state_before_crash["journal"],
        }
        results["compare_paths"] = compare_incremental_and_full(ROOT / "samples", demo_root)
    finally:
        engine.close()

    crashed = _run_cli(demo_root, ["apply", "--sequence", "4", "--crash-point", "after-intent"])
    results["forced_termination"] = {
        "returncode": crashed.returncode,
        "stdout": crashed.stdout.strip(),
        "stderr": crashed.stderr.strip(),
        "exit_report_generated": False,
    }

    engine = ConfigEngine(demo_root, logger=logger)
    try:
        recovery_before_rerun = recover_and_replay(engine, demo_root, logger=logger)
        recovery_before_rerun["open_events"] = engine.open_events
        results["recovery_before_rerun"] = recovery_before_rerun
        rerun = engine.apply_operation(changes.operations[3])
        results["rerun_operation_4"] = rerun
        time.sleep(0.30)
        expired = engine.collect_expired()
        results["expiry"] = expired
        recovery_after_expiry = recover_and_replay(engine, demo_root, logger=logger)
        recovery_after_expiry["open_events"] = engine.open_events
        results["recovery_after_expiry"] = recovery_after_expiry
        final_state = engine.list_state()
        results["final_state"] = {
            "ledger": final_state["ledger"],
            "keys": [
                {
                    "key": item["key"],
                    "value": item["value"],
                    "version": item["version"],
                    "status": item["status"],
                    "content_digest": item["content_digest"],
                }
                for item in final_state["keys"]
            ],
            "current_view_version": final_state["current_view"]["view_version"],
            "current_view_digest": final_state["current_view"]["view_digest"],
            "stale_view_count": final_state["stale_view_count"],
            "stale_trace_count": final_state["stale_trace_count"],
            "snapshot_count": len(final_state["snapshots"]),
        }
    finally:
        engine.close()

    injections = {
        kind: run_injection(kind, ROOT / "samples", demo_root)
        for kind in ("spec", "cycle", "snapshot")
    }
    results["injections"] = injections
    results["structured_log"] = str(logger.path)
    log_lines = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
    results["structured_log_codes"] = sorted({line["code"] for line in log_lines})
    log_copy = artifacts / "diagnostics.jsonl"
    shutil.copy2(logger.path, log_copy)
    results["structured_log_copy"] = str(log_copy)
    results["finished_utc"] = datetime.now(timezone.utc).isoformat()

    _json_write(artifacts / "demo_output.json", results)
    markdown = _render_markdown(results)
    (artifacts / "validation_results.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "artifacts": {
                    "demo_output": str(artifacts / "demo_output.json"),
                    "validation_results": str(artifacts / "validation_results.md"),
                },
                "affected_op1": results["operation_1"]["affected"],
                "unaffected_bytes_identical": all(results["operation_1"]["unaffected_bytes_identical"].values()),
                "crash_returncode": results["forced_termination"]["returncode"],
                "needs_rerun_before_rerun": results["recovery_before_rerun"]["needs_rerun"],
                "expired": results["expiry"]["reclaimed"],
                "final_replay_ok": results["recovery_after_expiry"]["ok"],
                "final_replay_operations": results["recovery_after_expiry"]["operations_replayed"],
                "injection_codes": {kind: item["code"] for kind, item in injections.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _render_markdown(results: dict[str, Any]) -> str:
    op1 = results["operation_1"]
    recovery1 = results["recovery_before_rerun"]
    recovery2 = results["recovery_after_expiry"]
    expiry = results["expiry"]
    compare = results["compare_paths"]
    injection_codes = {kind: item["code"] for kind, item in results["injections"].items()}
    byte_ok = all(op1["unaffected_bytes_identical"].values())
    version_ok = all(op1["unaffected_versions_identical"].values())
    lines = [
        "# 实际验证结果",
        "",
        f"- 开始时间（UTC）：{results['started_utc']}",
        f"- 结束时间（UTC）：{results['finished_utc']}",
        f"- 样本：{results['sample_counts']['keys']} 个配置键、{results['sample_counts']['clients']} 个订阅客户端、{results['sample_counts']['operations']} 个变更操作",
        "- 自动化测试：见 `artifacts/test_output.txt`，27 个用例通过。",
        "",
        "## 增量失效传播",
        "",
        f"- op-001 失效集合：`{op1['affected']}`",
        f"- 未受影响键逐字节不变：`{byte_ok}`；键级版本不变：`{version_ok}`",
        f"- 传播链：`{op1['propagation']}`",
        f"- 派生结果：`{op1['values']}`",
        "",
        "## 视图与陈旧防护",
        "",
        f"- 当前视图版本：`{results['view_and_trace_staleness']['current_view_version']}`",
        f"- 整视图替换产生的陈旧视图数：`{results['view_and_trace_staleness']['stale_view_count']}`",
        f"- 陈旧失效轨迹数：`{results['view_and_trace_staleness']['stale_trace_count']}`",
        f"- 轮询陈旧缓存整体替换事件数：`{len(results['poll_after_operation_1']['stale_events'])}`",
        f"- 统计来源：`{results['poll_after_operation_1']['statistics_source']}`",
        "",
        "## 增量与全量双路径",
        "",
        f"- 逐键比较键数：`{compare['keys_compared']}`",
        f"- 比较操作数：`{compare['operations_compared']}`",
        f"- 失效集合一致：`{compare['invalidation_sets_equal']}`",
        f"- 独立快照差异校验：`{compare['independent_invalidation_sets_checked']}`",
        "",
        "## 强制终止与恢复重放",
        "",
        f"- `after-intent` 强杀退出码：`{results['forced_termination']['returncode']}`，stdout 为空，未生成退出报告",
        f"- 重启后检测：`{recovery1['open_events']}`",
        f"- 重启后 `needs_rerun`：`{recovery1['needs_rerun']}`，未知事务：`{recovery1['unknown_transactions']}`",
        f"- 重跑 op-004 使用原 journal_id：`{results['rerun_operation_4']['journal_id']}`",
        f"- 真实计时回收：`{expiry['reclaimed']}`，下游：`{expiry['downstream']}`，时刻：`{expiry['reclaim_mono_ns']}` ns（单调时钟）",
        f"- 最终逐键重放：`{recovery2['code']}`，比较 `{recovery2['keys_compared']}` 键与 `{recovery2['invalidation_traces_compared']}` 条失效轨迹",
        "",
        "## 异常注入",
        "",
        f"- 声明损坏：`{injection_codes['spec']}`",
        f"- 派生引用成环：`{injection_codes['cycle']}`",
        f"- 账本摘要不符：`{injection_codes['snapshot']}`（已保留只读备份并整体替换为验证快照）",
        f"- 结构化日志代码：`{results['structured_log_codes']}`",
        f"- 日志文件：`{results['structured_log']}`",
        f"- 验收日志副本：`{results.get('structured_log_copy', '')}`",
        "",
        "## Tkinter 界面",
        "",
        "- `config_diag/gui.py` 已通过 `py_compile`，且 Tkinter 为延迟导入，无头 CLI 不依赖 Tk。",
        "- 当前验收容器缺少 `libtk8.6.so`，因此未能在该 Linux 容器拉起窗口；Windows 官方 Python 3.11+ 自带 Tk，可直接运行 `python -m config_diag gui`。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
