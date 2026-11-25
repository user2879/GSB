"""强制终止恢复验证（业务要求8）。

流程：
1. 正常跑一轮流水线并完整提交账本；
2. 第二轮写入 pending 批次后用 SIGKILL 强制终止进程（不生成退出报告）；
3. 重启执行 recover：只承认已提交记录，pending 标记 unknown 并回收租约配额；
4. 重跑 pending 后独立 replay 全部一致。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slo_burn.engine import run_pipeline  # noqa: E402
from slo_burn.ledger import Ledger  # noqa: E402
from slo_burn.lease_client import EvaluatorProcess, LeaseClient  # noqa: E402
from slo_burn.recovery import recover  # noqa: E402
from slo_burn.replay import verify_ledger  # noqa: E402
from slo_burn.scheduler import WindowScheduler  # noqa: E402


def main(workspace: str, runtime: str) -> dict:
    runtime = Path(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    ledger_path = runtime / "crash.db"
    socket_path = str(runtime / "crash.sock")

    with EvaluatorProcess(socket_path, 3, 2):
        with LeaseClient(socket_path) as client:
            scheduler = WindowScheduler(client, 3, 2)
            first = run_pipeline(workspace, ledger_path, scheduler=scheduler)
    assert first["commit_seq"] == 1

    # 第二轮：登记未提交批次后，在提交前被 SIGKILL（模拟强制终止）
    pid = os.getpid()

    def crash_worker() -> None:
        with Ledger(ledger_path) as ledger:
            ledger.register_pending(
                "checkout-api",
                777,
                {"minute": 777, "successes": 42, "failures": 7, "note": "uncommitted"},
            )
            time.sleep(0.2)
            os.kill(pid, signal.SIGKILL)

    import threading

    thread = threading.Thread(target=crash_worker, daemon=True)
    thread.start()
    time.sleep(1.0)
    raise RuntimeError("SIGKILL 未生效，测试无效")


def recover_and_rerun(workspace: str, runtime: str) -> dict:
    runtime = Path(runtime)
    ledger_path = runtime / "crash.db"
    report = recover(ledger_path)
    assert report.unknown_batches, "必须存在 unknown 未提交批次"
    assert report.unknown_batches[0]["minute"] == 777
    # 重跑前先回收租约配额（权威不持久化，重启即视为全部到期）
    socket_path = str(runtime / "crash2.sock")
    with EvaluatorProcess(socket_path, 3, 2):
        with Ledger(ledger_path) as ledger:
            for item in report.unknown_batches:
                ledger.clear_pending(item["service_id"], item["minute"])
            assert ledger.pending_batches() == []
        with LeaseClient(socket_path) as client:
            scheduler = WindowScheduler(client, 3, 2)
            run_pipeline(workspace, ledger_path, scheduler=scheduler)
    replay = verify_ledger(
        ledger_path,
        Path(workspace) / "services.json",
        Path(workspace) / "batches",
        Path(workspace) / "param_edits.json",
    )
    return {
        "recovery": report.__dict__,
        "replay": replay,
        "exit_report_generated": False,
    }


if __name__ == "__main__":
    mode = sys.argv[1]
    workspace, runtime = sys.argv[2], sys.argv[3]
    if mode == "crash":
        main(workspace, runtime)
    elif mode == "recover":
        print(json.dumps(recover_and_rerun(workspace, runtime), indent=2, ensure_ascii=False, default=str))
    else:
        raise SystemExit("mode must be crash|recover")
