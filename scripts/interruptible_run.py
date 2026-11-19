"""可中断训练驱动：在指定任务的指定轮次触发后等待外部信号或自杀。

环境变量：
  REGBENCH_CONFIG   配置路径
  REGBENCH_SNAPSHOT 快照目录
  REGBENCH_REPORT   报告输出路径
  REGBENCH_KILL     sigterm | sigkill | none
  REGBENCH_EPOCH    触发轮次
  REGBENCH_TASK     任务键子串
"""

from __future__ import annotations

import json
import os
import signal
import time

from regbench.config import load_config
from regbench.errors import FailureLog
from regbench.report import build_report
from regbench.runner import StopController, run_training
from regbench.snapshot import SnapshotStore


def main() -> int:
    config_path = os.environ["REGBENCH_CONFIG"]
    snapshot_dir = os.environ["REGBENCH_SNAPSHOT"]
    report_path = os.environ.get("REGBENCH_REPORT", "")
    kill_mode = os.environ.get("REGBENCH_KILL", "none")
    trigger_epoch = int(os.environ.get("REGBENCH_EPOCH", "5"))
    task_marker = os.environ.get("REGBENCH_TASK", "")

    config, config_failures = load_config(config_path)
    store = SnapshotStore(snapshot_dir)
    stop = StopController()

    def handler(signum, frame):  # noqa: ANN001
        stop.request_stop()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    triggered = {"done": False}

    def failpoint(stage: str, epoch: int, task_key: str) -> None:
        if (
            not triggered["done"]
            and stage == "train"
            and epoch == trigger_epoch
            and task_marker in task_key
        ):
            triggered["done"] = True
            if kill_mode == "sigterm":
                os.kill(os.getpid(), signal.SIGTERM)
            elif kill_mode == "sigkill":
                # 给文件系统一点时间后强制终止，模拟 kill -9。
                time.sleep(0.2)
                os.kill(os.getpid(), signal.SIGKILL)

    failure_log = FailureLog()
    failure_log.extend(config_failures)
    outcome = run_training(
        config, store, failure_log=failure_log,
        stop_controller=stop, failpoint=failpoint,
    )
    report = build_report(config, outcome, concurrency_label="interruptible")
    if report_path:
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({"graceful_stopped": outcome.graceful_stopped,
                      "completed": len(outcome.results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
